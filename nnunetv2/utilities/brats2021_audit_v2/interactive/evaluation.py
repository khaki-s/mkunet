"""No-LLM, no-training, GT-simulated interactive segmentation evaluation.

Ground truth is passed only to select_prompt and region_metrics. The actual
corrector receives MRI, the current predicted mask, and finite point/box prompts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
from pathlib import Path
import time
import traceback

import nibabel as nib
import numpy as np
import torch

from .backends import NNInteractiveBackend
from .evaluation_manifest import (MODALITIES, check_file, file_identity, load_protocol,
    object_digest, prepare_manifest, read_json, scientific_identity, write_json)
from .interaction_metrics import aggregate_results, prompt_budget_metadata, region_metrics
from .prompt_sampling import select_prompt
from .schemas import BoxPrompt, PointPrompt, SpatialROI, StructuredAction
from .session import CorrectionSession


def labels_from_logits(logits):
    positive = logits.detach().cpu().numpy() > 0
    labels = np.zeros(positive.shape[1:], dtype=np.uint8)
    for head in range(3):
        labels[positive[head]] = head + 1
    return labels


def labels_to_logits(labels):
    return np.stack([np.where(labels > head, 4.59512, -4.59512)
                     for head in range(3)]).astype(np.float32)


def read_labels(image):
    values = np.asanyarray(image.dataobj)
    if values.ndim != 3 or not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError('Require native-grid internal labels 0=BG,1=ED,2=NCR,3=ET')
    return values.astype(np.uint8)


def same_grid(image, reference):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError('MRI, prediction, and GT must have identical native shape and affine; do not auto-resample GT')


def verify_case_inputs(row):
    for identity in [row['prediction'], row['gt'], *row['modalities'].values()]:
        check_file(identity)


def load_case(row, modality_index, verify=True):
    if verify:
        verify_case_inputs(row)
    reference = nib.load(row['prediction']['path'])
    gt_image = nib.load(row['gt']['path'])
    same_grid(gt_image, reference)
    pred, gt = read_labels(reference), read_labels(gt_image)
    image = nib.load(row['modalities'][MODALITIES[modality_index]]['path'])
    same_grid(image, reference)
    raw = image.get_fdata(dtype=np.float32)[None]
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite MRI')
    return raw, pred, gt, reference


def load_session_mri(row, raw, reference, config):
    """Explicitly load every channel used by the frozen, predeclared head policy."""
    selected = config['modality_index']
    indices = {selected}
    if 'correction' in config:
        indices.update(config['correction']['modality_by_head'].values())
    mri = np.zeros((4, *reference.shape), dtype=np.float32)
    mri[selected] = raw[0]
    for index in indices - {selected}:
        image = nib.load(row['modalities'][MODALITIES[index]]['path'])
        same_grid(image, reference)
        mri[index] = image.get_fdata(dtype=np.float32)
    if not np.isfinite(mri).all():
        raise ValueError('Nonfinite selected MRI')
    return mri, [index in indices for index in range(4)]


def make_backend(config, raw, affine, *, device, inference_session=None):
    kwargs = dict(modality_index=config['modality_index'], device=device,
                  inference_session=inference_session, replay_box_history=True)
    if 'correction' in config:
        from .local_backend import SpatiallyConstrainedNNInteractiveBackend
        return SpatiallyConstrainedNNInteractiveBackend(config['model_folder'], raw, affine,
                                                        **kwargs, **config['correction'])
    return NNInteractiveBackend(config['model_folder'], raw, affine, **kwargs)


def prompt_roi(prompt, shape, requested_size):
    # Derived ONLY from the finite annotation, never the dense error mask.
    lo = np.array(prompt['voxel'] if prompt['kind'] == 'point' else prompt['box_start'])
    hi = lo + 1 if prompt['kind'] == 'point' else np.array(prompt['box_stop'])
    size = np.minimum(shape, np.maximum(requested_size, hi - lo + 16))
    start = np.maximum(0, np.minimum((lo + hi - size) // 2, np.array(shape) - size))
    return SpatialROI(tuple(int(v) for v in start), tuple(int(v) for v in start + size))


def register_prompt(session, prompt, number, roi_size):
    roi_id, prompt_id = f'roi_{number}', f'prompt_{number}'
    session.register_roi(roi_id, prompt_roi(prompt, session.shape, roi_size))
    kwargs = {}
    paired = prompt['kind'] == 'point_box'
    if prompt['kind'] in ('point', 'point_box'):
        point_id = prompt_id + '_point' if paired else prompt_id
        session.register_point(PointPrompt(point_id, tuple(prompt['voxel']), prompt['head'], prompt['positive']))
        kwargs['positive_point_ids' if prompt['positive'] else 'negative_point_ids'] = (point_id,)
    if prompt['kind'] in ('box', 'point_box'):
        box_id = prompt_id + '_box' if paired else prompt_id
        session.register_box(BoxPrompt(box_id, tuple(prompt['box_start']), tuple(prompt['box_stop']),
                                      prompt['head'], prompt['positive']))
        kwargs.update(box_prompt=True, box_id=box_id)
    return StructuredAction(prompt['head'], prompt['operation'], roi_id, **kwargs)


def stable_seed(seed, case_id, target, kind, step):
    text = f'{seed}|{case_id}|{target}|{kind}|{step}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(torch.device(device))


def evaluate_trajectory(session, gt, *, target, kind, config, device='cpu', save_snapshot=None):
    """Testable loop. GT never enters session or its corrector.

    A point/box budget counts annotations. A point_box budget counts rounds,
    each containing two new annotations (one point plus one box). Replayed
    history never increases annotation counts; one prediction runs per round.
    """
    budgets = config['budgets']
    design = prompt_budget_metadata(kind)
    volume = float(abs(np.linalg.det(np.asarray(session.affine)[:3, :3])))
    pred = labels_from_logits(session.logits)
    baseline = region_metrics(pred, gt, voxel_volume_mm3=volume)
    result = {'case_id': session.case_id, 'split': session.split, 'target': target,
              'prompt_kind': kind, 'status': 'complete', 'baseline': baseline,
              'budgets': {}, 'steps': [], 'stop_reason': 'budget_exhausted', **design}
    selection_seconds = inference_seconds = interaction_seconds = 0.0
    actual = 0

    def snapshot(budget, carried=False):
        result['budgets'][str(budget)] = {
            'metrics': region_metrics(pred, gt, voxel_volume_mm3=volume),
            'actual_prompts': actual * design['annotations_per_round'],
            'actual_tool_calls': actual, 'actual_rounds': actual,
            'actual_points': actual * design['points_per_round'],
            'actual_boxes': actual * design['boxes_per_round'],
            'budget_unit': design['budget_unit'],
            'cumulative_selection_seconds': selection_seconds,
            'cumulative_inference_seconds': inference_seconds,
            'cumulative_interaction_seconds': interaction_seconds,
            'carried_forward': carried,
        }
        if save_snapshot is not None:
            save_snapshot(budget, pred)

    snapshot(0)
    for step in range(1, max(budgets) + 1):
        begin = time.perf_counter()
        prompt = select_prompt(pred, gt, np.asarray(session.affine), target=target, kind=kind,
            seed=stable_seed(config['seed'], session.case_id, target, kind, step),
            box_axis=config['box_axis'], box_padding=config['box_padding'],
            box_max_extent=config['box_max_extent'])
        selection_seconds += time.perf_counter() - begin
        if prompt is None:
            interaction_seconds += time.perf_counter() - begin
            result['stop_reason'] = 'no_current_target_error'
            for budget in budgets:
                if str(budget) not in result['budgets']:
                    snapshot(budget, carried=True)
            break
        action = register_prompt(session, prompt, step, config['roi_size'])
        synchronize(device)
        inference_begin = time.perf_counter()
        action_record = session.apply_action(action)
        synchronize(device)
        elapsed = time.perf_counter() - inference_begin
        inference_seconds += elapsed
        pred = labels_from_logits(session.logits)
        interaction_seconds += time.perf_counter() - begin
        actual += 1
        result['steps'].append({'prompt_number': step, 'round_number': step,
                                'new_annotations': design['annotations_per_round'],
                                'prompt': prompt, 'action_result': action_record,
                                'backend_constraints': getattr(session.corrector, 'last_constraints', None),
                                'inference_seconds': elapsed,
                                'metrics': region_metrics(pred, gt, voxel_volume_mm3=volume)})
        # Batch evaluation never performs undo; retain finite prompt history but
        # free old dense ROI snapshots to bound CPU RAM across five interactions.
        session._undo_stack.clear()
        if step in budgets:
            snapshot(step)
    return result


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_run(folder):
    folder = Path(folder)
    run = read_json(folder / 'run.json')
    records = [read_json(p) for p in sorted((folder / 'records').glob('*/*.json'))]
    summary = aggregate_results(records, run['case_ids'], run['science']['protocol']['targets'],
                                budgets=run['science']['protocol']['budgets'],
                                tolerance=run['science']['protocol']['tolerance'])
    summary['run_identity'] = run['identity']
    summary['split'] = run['split']
    summary['pilot'] = run['pilot']
    summary['run_status'] = run.get('status', 'unknown')
    summary['valid_for_full_cohort_reporting'] = bool(
        run.get('status') == 'complete' and not run['pilot'] and summary['coverage']['full_cohort_complete'])
    summary['prompt_kind'] = run['science']['prompt_kind']
    summary.update(prompt_budget_metadata(summary['prompt_kind']))
    summary['gt_usage'] = 'oracle finite annotation selection and metric scoring; never a model input'
    write_json(folder / 'summary.json', summary)
    write_csv(folder / 'region_summary.csv', summary['groups'])
    write_csv(folder / 'timing_summary.csv', summary['timing_groups'])
    per_case = []
    for record in records:
        if record['status'] != 'complete':
            continue
        for budget, state in record['budgets'].items():
            for region, metrics in state['metrics'].items():
                before, after = record['baseline'][region]['dice'], metrics['dice']
                per_case.append({'case_id': record['case_id'], 'target': record['target'],
                    'prompt_kind': record['prompt_kind'], 'budget': int(budget), 'region': region,
                    'baseline_dice': before, 'dice': after,
                    'delta_pp': 100 * (after - before) if before is not None and after is not None else None,
                    'actual_prompts': state['actual_prompts'], 'carried_forward': state['carried_forward'],
                    'actual_points': state.get('actual_points'), 'actual_boxes': state.get('actual_boxes'),
                    'actual_rounds': state.get('actual_rounds'),
                    'actual_tool_calls': state['actual_tool_calls'],
                    'budget_unit': record.get('budget_unit', 'annotations'),
                    'interaction_seconds': state['cumulative_interaction_seconds'],
                    'inference_seconds': state['cumulative_inference_seconds'],
                    'fp_mm3': metrics['fp_mm3'], 'fn_mm3': metrics['fn_mm3']})
    write_csv(folder / 'per_case_metrics.csv', per_case)
    return summary


def assert_locked(lock, science):
    if lock['science'] != science:
        differences = [k for k in science if science[k] != lock['science'].get(k)]
        raise ValueError(f'Test protocol differs from frozen validation: {differences}')


def run_evaluation(args):
    torch.set_num_threads(4)
    manifest = read_json(args.manifest)
    config = load_protocol(args.protocol)
    if manifest['split'] not in ('validation', 'test'):
        raise ValueError('This entry point accepts validation/test only')
    # Verify the prepared split identities are still the same files.
    check_file(manifest['splits_file'])
    check_file(manifest['source_manifest'])
    check_file(manifest['prediction_provenance'])
    if manifest.get('test_whitelist'):
        check_file(manifest['test_whitelist'])
    split = read_json(manifest['splits_file']['path'])[manifest['fold']]
    case_ids = [row['case_id'] for row in manifest['cases']]
    source_ids = [row['case_id'] for row in read_json(manifest['source_manifest']['path'])['cases']]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != set(source_ids):
        raise ValueError('Prepared case IDs differ from source manifest or contain duplicates')
    if manifest['split'] == 'validation':
        if set(case_ids) != set(split['val']):
            raise ValueError('Prepared validation IDs differ from fold validation split')
    else:
        if not manifest.get('test_whitelist'):
            raise ValueError('Prepared test manifest lacks fixed whitelist')
        expected = {row['case_id'] for row in read_json(manifest['test_whitelist']['path'])['cases']}
        if set(case_ids) != expected or set(case_ids) & set(split['train'] + split['val']):
            raise ValueError('Prepared test IDs violate frozen split membership')
    science = scientific_identity(manifest, config, args.kind)
    science['runtime_versions'] = {name: version(name) for name in (
        'nninteractive', 'nnunetv2', 'dynamic-network-architectures', 'acvl-utils',
        'torch', 'numpy', 'scipy', 'nibabel')}
    if manifest['split'] == 'test':
        if not args.locked_protocol:
            raise ValueError('Test requires --locked-protocol exported after a complete val80 run')
        if not manifest['initialization']['verified']:
            raise ValueError('Test requires verified initial checkpoint provenance')
        assert_locked(read_json(args.locked_protocol), science)
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    rows = manifest['cases'][:args.limit]
    identity_data = {'manifest': file_identity(args.manifest), 'science': science,
                     'case_ids': [row['case_id'] for row in rows],
                     'save_predictions': args.save_predictions, 'device': args.device}
    run = {'schema_version': 1, 'identity': object_digest(identity_data), **identity_data,
           'split': manifest['split'], 'pilot': len(rows) != len(manifest['cases']),
           'verified_initialization': manifest['initialization']['verified'],
           'full_case_count': len(manifest['cases']), 'status': 'running'}
    run['hardware'] = {'device': args.device,
        'gpu_name': torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None}
    folder = Path(args.output)
    if folder.exists():
        if not args.resume or not (folder / 'run.json').is_file():
            raise FileExistsError('Use a fresh output, or --resume for an existing matching run')
        previous = read_json(folder / 'run.json')
        if previous['identity'] != run['identity']:
            raise ValueError('Resume identity mismatch: protocol/inputs/code/model/output options changed')
    else:
        folder.mkdir(parents=True)
    write_json(folder / 'run.json', run)
    engine = None
    try:
        for row in rows:
            # Even fully completed cases are checked on resume. Otherwise a
            # changed MRI/GT could silently keep an old score in the cohort.
            verify_case_inputs(row)
            pending = []
            for target in config['targets']:
                record_file = folder / 'records' / target / (row['case_id'] + '.json')
                if not record_file.exists() or read_json(record_file)['status'] != 'complete':
                    pending.append((target, record_file))
            if not pending:
                continue
            raw, pred, gt, reference = load_case(row, config['modality_index'], verify=False)
            # Exactly one raw MRI goes to the upstream binary model. The
            # multi-region session uses explicit flags and zero other channels.
            mri, mask = load_session_mri(row, raw, reference, config)
            for target, record_file in pending:
                try:
                    load_begin = time.perf_counter()
                    backend = make_backend(config, raw, reference.affine,
                        device=args.device, inference_session=engine)
                    if engine is None:
                        run['model_load_seconds'] = time.perf_counter() - load_begin
                        write_json(folder / 'run.json', run)
                    engine = backend.engine
                    backend.is_trained_checkpoint = True
                    session = CorrectionSession(mri, mask, labels_to_logits(pred), reference.affine,
                        backend, case_id=row['case_id'], split=manifest['split'],
                        enforce_clicks=True, taper_voxels=3)

                    def save_snapshot(budget, labels):
                        if args.save_predictions == 'none' or (args.save_predictions == 'final' and budget != 5):
                            return
                        path = folder / 'predictions' / target / f'budget_{budget}' / (row['case_id'] + '.nii.gz')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        header = reference.header.copy()
                        header.set_data_dtype(np.uint8)
                        nib.save(nib.Nifti1Image(labels, reference.affine, header), path)

                    condition_begin = time.perf_counter()
                    record = evaluate_trajectory(session, gt, target=target, kind=args.kind,
                        config=config, device=args.device, save_snapshot=save_snapshot)
                    record['condition_wall_seconds'] = time.perf_counter() - condition_begin
                    record['runtime_scope'] = 'interaction latency includes GT selection and correction; inference includes session merge; image IO/model load excluded'
                    write_json(record_file, record)
                    final = record['budgets']['5']
                    detail = (f', {final["actual_rounds"]} rounds '
                              f'({final["actual_points"]} points + {final["actual_boxes"]} boxes)'
                              if args.kind == 'point_box' else '')
                    print(f'{row["case_id"]} {target} {args.kind}: '
                          f'{final["actual_prompts"]} prompts{detail}, {record["stop_reason"]}', flush=True)
                    del session
                except Exception as exc:
                    write_json(record_file, {'case_id': row['case_id'], 'split': manifest['split'],
                        'target': target, 'prompt_kind': args.kind, 'status': 'failed',
                        **prompt_budget_metadata(args.kind),
                        'error': str(exc), 'traceback': traceback.format_exc()})
                    raise
            summarize_run(folder)
        run['status'] = 'complete'
    except BaseException as exc:
        run['status'] = 'incomplete'
        run['error'] = str(exc)
        raise
    finally:
        write_json(folder / 'run.json', run)
        summarize_run(folder)
    return run


def freeze_run(args):
    folder = Path(args.validation_run)
    run = read_json(folder / 'run.json')
    summary = summarize_run(folder)
    if (run['split'] != 'validation' or run['pilot'] or run['status'] != 'complete'
            or len(run['case_ids']) != 80 or not run['verified_initialization']):
        raise ValueError('Freeze requires a complete, non-pilot val80 with verified initial checkpoint')
    records = [read_json(p) for p in (folder / 'records').glob('*/*.json')]
    expected = {(key, target) for key in run['case_ids'] for target in run['science']['protocol']['targets']}
    complete = {(r['case_id'], r['target']) for r in records if r['status'] == 'complete'}
    if complete != expected:
        raise ValueError('Cannot freeze incomplete coverage')
    if Path(args.output).exists():
        raise FileExistsError('Keep protocol locks immutable; choose a new output')
    lock = {'schema_version': 1, 'science': run['science'], 'validation_run': str(folder.resolve()),
            'validation_run_identity': run['identity'], 'validation_summary_sha256': file_identity(folder / 'summary.json')['sha256'],
            'annotation': 'Simulated interactive test with finite GT oracle prompts; not autonomous test segmentation'}
    write_json(args.output, lock)
    return lock


def preflight(args):
    config = load_protocol(args.protocol)
    import nnunetv2
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    source = Path(nnunetv2.__file__).resolve()
    repository = Path(__file__).resolve().parents[4]
    if repository in source.parents:
        raise RuntimeError('nnInteractive imported the legacy training nnunetv2. Use a separate '
                           'environment with official dependencies; do not pip install -e the old repo there.')
    for method in ('set_image', 'set_target_buffer', 'add_initial_seg_interaction',
                   'add_point_interaction', 'add_bbox_interaction', 'initialize_from_trained_model_folder'):
        if not hasattr(nnInteractiveInferenceSession, method):
            raise RuntimeError(f'Installed nnInteractive lacks required API: {method}')
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this environment/device mapping')
        print('GPU:', torch.cuda.get_device_name(torch.device(args.device)))
    model_identity = scientific_identity({'initialization': {'checkpoint_sha256': None, 'tta': None}}, config, 'point')['nninteractive_model']
    print('Official nnunetv2:', source)
    print('nnInteractive:', version('nninteractive'), 'torch:', version('torch'))
    print('Model files verified:', ', '.join(model_identity))
    print('Qwen/Transformers not required. This check did not run patient inference.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Validate split and hash read-only source inputs')
    prepare.add_argument('--split', choices=('validation', 'test'), required=True)
    for option in ('source-manifest', 'splits-file', 'predictions', 'prediction-provenance', 'gt', 'output'):
        prepare.add_argument('--' + option, required=True)
    prepare.add_argument('--fold', type=int, default=0)
    prepare.add_argument('--test-manifest')
    prepare.add_argument('--allow-unverified-initialization', action='store_true')
    prepare.set_defaults(handler=prepare_manifest)
    run = commands.add_parser('run', help='Run a separate point, box, or paired point+box experiment (never Qwen)')
    for option in ('manifest', 'protocol', 'output'):
        run.add_argument('--' + option, required=True)
    run.add_argument('--kind', choices=('point', 'box', 'point_box'), required=True,
                     help='point_box budgets count rounds: each round has one point and one box')
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--limit', type=int)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--save-predictions', choices=('none', 'final', 'budgets'), default='budgets')
    run.add_argument('--locked-protocol')
    run.set_defaults(handler=run_evaluation)
    summary = commands.add_parser('summarize', help='Regenerate summaries without loading a model')
    summary.add_argument('--run', required=True)
    summary.set_defaults(handler=lambda args: summarize_run(args.run))
    freeze = commands.add_parser('freeze', help='Freeze a complete validation experiment before test')
    freeze.add_argument('--validation-run', required=True)
    freeze.add_argument('--output', required=True)
    freeze.set_defaults(handler=freeze_run)
    check = commands.add_parser('check', help='Check installed official API, local weights, and GPU without inference')
    check.add_argument('--protocol', required=True)
    check.add_argument('--device', default='cuda:0')
    check.set_defaults(handler=preflight)
    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
