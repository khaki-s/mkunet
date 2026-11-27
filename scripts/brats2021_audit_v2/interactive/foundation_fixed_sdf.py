#!/usr/bin/env python3
"""Reproduce the frozen SDF-only removal arm across the original BraTS splits."""
import argparse
import fcntl
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

# The frozen expert trainers come from the workspace. Native nnInteractive
# uses the installed official package and the existing private runtime.
if len(sys.argv) > 1 and sys.argv[1] in ('sam', 'respond'):
    from foundation_prompt import name as runtime
else:
    runtime = 'nnunetv2.utilities.brats2021_audit_v2.interactive'
base = runtime + '.foundation_prompt_v2'
common = import_module(base + '.common')
MODALITIES = common.MODALITIES
check_file, file_identity, manifest = common.check_file, common.file_identity, common.manifest
read_json, read_labels, same_grid = common.read_json, common.read_labels, common.same_grid
save_mask, write_json = common.save_mask, common.write_json
background, candidates, dense, matrix, expert_geometry, expert_prompts, residual, scoring = [
    import_module(base + '.sam_protocol.' + name) for name in
    ('background', 'candidates', 'dense', 'evaluation_matrix', 'expert_geometry', 'expert_prompts', 'residual', 'scoring')
]

STUDY = 'frozen_sam3_sdf_native192_remove_v1'
NAMES = ('BraTSAuditBaseline300', 'BraTSSemanticMoELoss300', 'BraTSAgentLocalV2Loss300')
TRAINERS = ('nnUNetTrainer_BraTSAudit200', 'nnUNetTrainer_BraTSAudit200', 'nnUNetTrainer_BraTSAgentV2Loss300')
MODULE = 'nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.'
MODELS = Path('/data02/chujingyi/data/nnUNet_results/Dataset137_BraTS2021')
THRESHOLD = 2.
WT_TARGET = .30


def prepare(args):
    root = Path(args.root)
    site = Path(args.output)
    look = root / 'sam_protocol'
    v9 = look / 'expanded_expert_support_v9' / 'train100_val25'
    baseline_score = v9 / 'margin2/directional_score.json'
    existing = read_json(baseline_score)
    selected = existing['comparison']['sdf/native_remove_only']
    gain = selected['WT']['gain_pp']
    candidate = read_json(args.comparison)
    variants = [candidate['comparison'][name]['WT'] for name in dense.VARIANTS]
    best = max(variants, key=lambda value: value['gain_pp'])
    cohort_path = look / 'balanced_errors_v6/cohort.json'
    if (candidate['cohort'] != file_identity(cohort_path) or candidate['n_cases'] != 25
            or any(value['n_paired'] != 25 for value in variants) or existing['n_cases'] != 25):
        raise ValueError('Fallback selection requires complete matching val25 scoring')
    if best['gain_pp'] > gain:
        raise ValueError(f'The newly calibrated pipeline reached {best["gain_pp"]:+.4f} pp and exceeded the fallback')
    train = root / 'manifests/train320.json'
    validation = root / 'manifests/val80.deployment.json'
    test = root / 'manifests/test100.deployment.json'
    target = {'train': file_identity(train), 'validation': file_identity(validation), 'test': file_identity(test)}
    groups = {}
    for section, amount in (('train', 320), ('validation', 80), ('test', 100)):
        source = manifest(target[section]['path'], deployment=section != 'train', allowed=(section,))
        if len(source['cases']) != amount:
            raise ValueError('Expected the original train320, val80, and test100 splits')
        groups[section] = [r['case_id'] for r in source['cases']]
    cohort = site / 'cohort.json'
    selection_doc = {'schema': STUDY, 'sources': target, 'cases': groups}
    if cohort.is_file() and read_json(cohort) != selection_doc:
        raise ValueError('The fixed-method cohort changed')
    if not cohort.is_file():
        write_json(cohort, selection_doc)
    experts = []
    for name, trainer in zip(NAMES, TRAINERS):
        home = MODELS / f'nnUNetTrainer_{name}__nnUNetPlans__3d_fullres'
        artifacts = {k: file_identity(home / k) for k in ('plans.json', 'dataset.json', 'fold_0/checkpoint_final.pth')}
        experts.append({'name': name, 'trainer': MODULE + trainer, 'folder': str(home), 'files': artifacts})
    permit = args.permit
    record = {'schema': STUDY, 'cohort': file_identity(cohort), 'pilot_cohort': file_identity(cohort_path),
              'variant': 'sdf',
              'pilot_score': file_identity(baseline_score), 'paired_voxel_comparison': file_identity(args.comparison),
              'sources': target, 'models': candidates.model_files(), 'experts': experts,
              'sam': background.contract(read_json(train)), 'margin': THRESHOLD, 'wt_target_pp': WT_TARGET,
              'pilot_wt_gain_pp': gain, 'new_method_wt_gain_pp': best['gain_pp'],
              'code': [file_identity(path) for path in (__file__, expert_prompts.__file__, expert_geometry.__file__)],
              'pilot_expert_config': file_identity(v9 / 'experts.json'), 'pilot_cache': str(v9)}
    if Path(permit).is_file() and read_json(permit) != record:
        raise ValueError('Fixed method or cohort changed. Use a new output directory.')
    if not Path(permit).is_file():
        write_json(permit, record)
    root_test = site / 'test100_15_modalities'
    result = matrix.prepare_matrix(SimpleNamespace(manifest=str(test), baselines=args.baselines,
        permit=permit, output=str(root_test / 'manifest.json')), record)
    print(result, flush=True)


def method(args):
    spec = read_json(args.permit)
    if spec['schema'] != STUDY or spec['margin'] != THRESHOLD or spec['variant'] != 'sdf':
        raise ValueError('Expected the frozen SDF negative interaction method')
    for identity in (*spec['code'], spec['cohort'], spec['pilot_cohort'], spec['paired_voxel_comparison']):
        check_file(identity)
    return spec


def directory(args):
    root = Path(args.output)
    if args.split == 'test':
        field = f'mask_{args.mask_code:02d}'
        return root / 'test100_15_modalities' / 'work' / field, root / 'test100_15_modalities' / 'predictions' / field
    name = 'train320' if args.split == 'train' else 'val80'
    return root / name / 'work', root / name / 'predictions'


def selection(args, spec):
    if args.split == 'test':
        path = Path(args.output) / 'test100_15_modalities/manifest.json'
        return matrix.test_rows(path, args.mask_code, args.permit)
    source = spec['sources'][args.split]
    check_file(source)
    entry = manifest(source['path'], deployment=args.split != 'train', allowed=(args.split,))
    return [{**{k: v for k, v in row.items() if k != 'gt'}, 'mask_code': 15} for row in entry['cases']]


def expert(args, spec, selected, work):
    from predict_prompt_expert import initialize
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import MissingModalityPredictor, preprocess_available_case

    network = spec['experts'][args.expert_index]
    site = work / 'experts' / network['name']
    site.mkdir(parents=True, exist_ok=True)
    for file in network['files'].values():
        check_file(file)
    prior = read_json(spec['pilot_expert_config']['path']).get(args.split, ())
    predictor = preprocessor = None
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(3)
    for row in selected:
        key = row['case_id']
        if 'gt' in row or len(row['modalities']) != sum(row['availability']):
            raise ValueError('Expert requires label-free available-only MRI')
        receipt = site / f'{key}.json'
        if not receipt.is_file() and prior:
            cached = Path(prior[args.expert_index]) / f'{key}.json'
            if cached.is_file() and read_json(cached)['binding']['row'] == row:
                receipt.symlink_to(cached)
        if receipt.is_file():
            record = read_json(receipt)
            if (record['binding']['row'] != row or record['binding']['model'] != network['files']
                    or record['gt_used'] is not False):
                raise ValueError('Cached expert mask has mismatched inputs')
            check_file(record['mask'])
            continue
        original = row['modalities']
        for identity in (row['prediction'], *original.values()):
            check_file(identity)
        full = all(row['availability'])
        if predictor is None:
            cls = nnUNetPredictor if full else MissingModalityPredictor
            predictor = cls(device=torch.device(args.device), perform_everything_on_device=False,
                            use_mirroring=True, verbose=False, allow_tqdm=False)
            initialize(predictor, network['folder'], 'checkpoint_final.pth', network['trainer'])
            preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
        if full:
            array, _, geometry = preprocessor.run_case([original[name]['path'] for name in MODALITIES], None,
                predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
            prediction = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(array)).cpu()
        else:
            available = {name: identity['path'] for name, identity in original.items()}
            tensor, geometry, mask = preprocess_available_case(available, predictor.plans_manager,
                predictor.configuration_manager, predictor.dataset_json)
            prediction, _ = predictor.predict_with_mask(tensor, mask, case_id=key, collect_routes=False)
        temporary = site / f'{key}.partial'
        export_prediction_from_logits(prediction, geometry, predictor.configuration_manager, predictor.plans_manager,
                                      predictor.dataset_json, str(temporary), False, num_threads_torch=3)
        staged = temporary.with_suffix('.partial.nii.gz')
        same_grid(nib.load(staged), nib.load(row['prediction']['path']))
        path = site / f'{key}.nii.gz'
        staged.replace(path)
        bind = {'row': row, 'model': network['files'], 'mirroring': True}
        write_json(receipt, {'binding': bind, 'mask': file_identity(path), 'gt_used': False})
        print(network['name'], key, flush=True)


def pilot_receipt(spec, row):
    root = Path(spec['pilot_cache'])
    key = row['case_id']
    saved = root / 'margin2' / 'responses' / key / 'responses.json'
    if not saved.is_file():
        return None
    state = read_json(saved)
    if state['row'] != row:
        return None
    _, _, sam = dense.read_masks(root / 'dense_expanded', row)
    executor = state['executor']
    if (state['gt_used'] is not False or state['sam'] != sam['arrays']
            or executor['native_margin']['tolerance'] != THRESHOLD
            or executor['nninteractive_model'] != spec['models']
            or sam['contract']['expert_config'] != spec['pilot_expert_config']):
        raise ValueError('Archived pilot response cannot be reused')
    check_file(state['arrays'])
    return file_identity(saved)


def sam(args, spec, selected, work):
    from foundation_sam_protocol import prepare_sam_import

    uncached = [row for row in selected if pilot_receipt(spec, row) is None]
    if not uncached:
        return
    setup = work / 'experts.json'
    expected = {args.split: [str(work / 'experts' / name) for name in NAMES]}
    work.mkdir(parents=True, exist_ok=True)
    with setup.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not setup.is_file():
            write_json(setup, expected)
        elif read_json(setup) != expected:
            raise ValueError('Stored expert folders differ')
    prepare_sam_import(SimpleNamespace(stage='expert-sam', device=args.device))
    cfg = SimpleNamespace(cohort=spec['pilot_cohort']['path'], split=args.split, selected_rows=uncached, shard_index=0,
        num_shards=1, experts=str(setup), output=str(work / 'paired'), device=args.device,
        stream_blocks=16, ready_only=False, profile=expert_prompts.MULTISLICE)
    expert_prompts.expert_sam(cfg)
    cfg.dense = cfg.output
    cfg.output = str(work / 'expanded')
    expert_prompts.expert_expand(cfg)


@torch.inference_mode()
def respond(args, spec, selected, work, output):
    torch.set_num_threads(3)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < candidates.MIN_FREE_MIB * 2**20:
        raise RuntimeError('Insufficient free CUDA memory for native 192³ responses')
    torch.cuda.reset_peak_memory_stats(device)
    backend = None
    for row in selected:
        key = row['case_id']
        folder = work / 'predictions' / key
        receipt = folder / 'prediction.json'
        replay = pilot_receipt(spec, row)
        source = replay or file_identity(work / 'expanded' / key / 'sam.json')
        binding = {'row': row, 'models': spec['models'], 'selected': 'sdf', 'support': source,
                   'method': file_identity(args.permit)}
        if receipt.is_file():
            stored = read_json(receipt)
            if stored['binding'] != binding or stored['gt_used'] is not False:
                raise ValueError('Stored fixed-method prediction differs')
            check_file(stored['result']['sdf']['mask'])
            continue
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        labels = read_labels(reference)
        if replay:
            published = read_json(replay['path'])
            with np.load(published['arrays']['path'], allow_pickle=False) as array:
                packed = array['packed'][0, :, 2]
                heads = np.unpackbits(packed, axis=-1, count=labels.size).reshape(3, *labels.shape).astype(bool)
        else:
            mri, _, _ = dense.load_native(row)
            _, _, cue = dense.read_masks(work / 'expanded', row)
            with np.load(cue['arrays']['path'], allow_pickle=False) as array:
                components = array['components']
            heads = np.stack([labels > head for head in range(len(dense.HEADS))])
            for h, base in enumerate(heads):
                moves = [a for a in cue['actions'] if a['head'] == h and not a['positive'] and a['variants'][0]['accept']]
                if not moves:
                    continue
                if backend is None:
                    backend = candidates.NativeExecutor(str(device))
                channel = next(i for i in dense.PREFERRED_MODALITIES[h] if row['availability'][i])
                backend.engine.set_image(np.ascontiguousarray(mri[channel][None], dtype=np.float32))
                original = base.copy()
                for action in moves:
                    spot = components[h] == action['id']
                    result = expert_prompts._margin_finite(backend, original, spot, action['variants'][0]['prompt'], THRESHOLD)
                    base[spot] = result[spot]
        if any(np.any(head & ~(labels > i)) for i, head in enumerate(heads)):
            raise ValueError('Frozen treatment may only remove foreground')
        result = residual.merge_heads(heads)
        identity = save_mask(output / f'{key}.nii.gz', result, reference)
        write_json(receipt, {'binding': binding, 'result': {'sdf': {'mask': identity}}, 'gt_used': False,
                            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30})
        print(key, 'pilot replay' if replay else 'native removal', flush=True)


def score(args, spec, selected, work, output):
    test = Path(args.output) / 'test100_15_modalities'
    if args.split == 'test':
        cfg = SimpleNamespace(matrix=str(test / 'manifest.json'), truth=args.truth, permit=args.permit,
                              output=str(test), mask_code=args.mask_code)
        return matrix.score_matrix_mask(cfg, spec, wt_target=WT_TARGET)
    truth = {r['case_id']:r for r in manifest(args.truth, allowed=(args.split,))['cases']}
    combined = []
    for row in selected:
        key = row['case_id']
        receipt = scoring._validate_prediction(row, work / 'predictions' / key / 'prediction.json', spec['models'], 'sdf')
        result = receipt['result']['sdf']['mask']
        check_file(result)
        old, gt, ref, volume = scoring._image_pair(row, truth[key])
        image = nib.load(result['path'])
        same_grid(image, ref)
        new = read_labels(image)
        visible = (old > 0) | (gt > 0) | (new > 0)
        area = residual._window(visible, padding=0) if visible.any() else np.s_[:, :, :]
        combined.extend({'case_id': key, **r} for r in scoring._measure(old[area], new[area], gt[area], volume))
    flat = scoring.summary(combined)
    path = work.parent / 'summary.json'
    write_json(path, {'split': args.split, 'n_cases': len(selected), 'method': file_identity(args.permit),
                      'regions': flat, 'per_case': combined})
    scoring._csv(work.parent / 'per_case_metrics.csv', combined)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'expert', 'sam', 'respond', 'score', 'summary'))
    for field in ('root', 'output', 'permit'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--comparison')
    p.add_argument('--baselines')
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    p.add_argument('--mask-code', type=int, choices=matrix.CODES, default=15)
    p.add_argument('--truth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--expert-index', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    if args.stage == 'prepare':
        return prepare(args)
    spec = method(args)
    if args.stage == 'summary':
        test = Path(args.output) / 'test100_15_modalities'
        cfg = SimpleNamespace(matrix=str(test / 'manifest.json'), permit=args.permit, output=str(test))
        print(matrix.summarize_matrix(cfg, spec, wt_target=WT_TARGET), flush=True)
        return
    selected = selection(args, spec)
    work, output = directory(args)
    if args.stage == 'score':
        print(score(args, spec, selected, work, output), flush=True)
    else:
        chosen = selected[args.shard_index::args.num_shards]
        if args.stage == 'respond':
            respond(args, spec, chosen, work, output)
        else:
            globals()[args.stage](args, spec, chosen, work)


if __name__ == '__main__':
    main()
