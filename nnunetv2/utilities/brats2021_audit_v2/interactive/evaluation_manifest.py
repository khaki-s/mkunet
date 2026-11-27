"""Explicit split, input provenance, and frozen protocol for oracle evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path)}


def source_cases(path):
    cases = read_json(path)['cases']
    ids = [c['case_id'] for c in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Source manifest has no cases or duplicated case IDs')
    for case in cases:
        case_id = case['case_id']
        if Path(case_id).name != case_id or case_id in ('.', '..'):
            raise ValueError('Unsafe case ID')
    return cases


def prepare_manifest(args):
    """Hash inputs without modifying them. Training IDs cannot become evaluation IDs."""
    if Path(args.output).exists():
        raise FileExistsError(f'Manifest exists; use a new output: {args.output}')
    cases = source_cases(args.source_manifest)
    fold = read_json(args.splits_file)[args.fold]
    train, val = set(fold['train']), set(fold['val'])
    ids = {c['case_id'] for c in cases}
    if train & val:
        raise ValueError('Training/validation split overlaps')
    if args.split == 'validation':
        if ids != val:
            raise ValueError('Validation cases must exactly match splits_final[fold].val')
    else:
        if not args.test_manifest:
            raise ValueError('Test preparation requires --test-manifest (fixed test whitelist)')
        expected = {c['case_id'] for c in source_cases(args.test_manifest)}
        if ids != expected or ids & (train | val):
            raise ValueError('Test cases must match the fixed whitelist and exclude train/val')
    if ids & train:
        raise ValueError('Training cases cannot be evaluated by this entry point')
    provenance = read_json(args.prediction_provenance)
    if provenance.get('status') != 'complete':
        raise ValueError('Initial prediction provenance is not marked complete')
    contract = provenance.get('contract', provenance)
    origin = contract.get('provenance', contract)
    checkpoint_sha = origin.get('checkpoint_sha256')
    checkpoint = origin.get('checkpoint')
    verified = bool(checkpoint and checkpoint_sha and len(checkpoint_sha) == 64)
    if not verified and not args.allow_unverified_initialization:
        raise ValueError('Initial predictions need checkpoint name + SHA256. Re-export best val80, '
                         'or explicitly allow unverified initialization for a pilot only.')
    declared_ids = contract.get('case_ids')
    if declared_ids is not None and set(declared_ids) != ids:
        raise ValueError('Initial prediction provenance case IDs do not match the source split')
    if 'mask_codes' in contract:
        expected_predictions = Path(args.prediction_provenance).resolve().parent / 'predictions' / 'mask_15'
        if 15 not in contract['mask_codes'] or Path(args.predictions).resolve() != expected_predictions:
            raise ValueError('Use the mask_15 prediction folder belonging to this evaluation manifest')
    output_cases = []
    for case in sorted(cases, key=lambda c: c['case_id']):
        key = case['case_id']
        modalities = case['modalities']
        if set(modalities) != set(MODALITIES):
            raise ValueError(f'{key}: first protocol requires all four source MRIs')
        row = {'case_id': key, 'modalities': {name: file_identity(modalities[name]) for name in MODALITIES},
               'prediction': file_identity(Path(args.predictions) / f'{key}.nii.gz'),
               'gt': file_identity(Path(args.gt) / f'{key}.nii.gz')}
        output_cases.append(row)
        print(f'Prepared {len(output_cases)}/{len(cases)} {key}', flush=True)
    manifest = {
        'schema_version': 1, 'split': args.split, 'fold': args.fold,
        'source_manifest': file_identity(args.source_manifest),
        'splits_file': file_identity(args.splits_file),
        'prediction_provenance': file_identity(args.prediction_provenance),
        'test_whitelist': file_identity(args.test_manifest) if args.test_manifest else None,
        'initialization': {'verified': verified, 'checkpoint': checkpoint,
                           'checkpoint_sha256': checkpoint_sha,
                           'tta': origin.get('tta', contract.get('tta', contract.get('mirroring'))),
                           'mirror_axes': contract.get('mirror_axes'),
                           'tile_step': contract.get('tile_step'),
                           'input_protocol': contract.get('protocol'),
                           'network_sha256': contract.get('network_sha256')},
        'cases': output_cases,
        'annotation_protocol': 'GT supplies finite prompts and evaluation only; no GT to model',
    }
    write_json(args.output, manifest)
    return manifest


def check_file(identity):
    if digest(identity['path']) != identity['sha256']:
        raise ValueError(f'Input changed since preparation: {identity["path"]}')


def load_protocol(path):
    config = read_json(path)
    required = {'budgets', 'targets', 'seed', 'roi_size', 'modality_index', 'model_folder',
                'box_axis', 'box_padding', 'box_max_extent', 'tolerance'}
    if not required.issubset(config) or set(config) - required - {'correction'}:
        raise ValueError(f'Protocol requires {sorted(required)}; only optional key is correction')
    if config['budgets'] != [0, 1, 3, 5]:
        raise ValueError('This experiment uses budgets 0,1,3,5')
    if config['targets'] != ['ED', 'ET', 'NCR', 'joint']:
        raise ValueError('Keep all four independent target conditions: ED,ET,NCR,joint')
    if (len(config['roi_size']) != 3 or any(type(v) is not int or v < 8 for v in config['roi_size'])
            or type(config['modality_index']) is not int or config['modality_index'] not in range(4)
            or type(config['seed']) is not int or config['seed'] < 0
            or type(config['box_axis']) is not int or config['box_axis'] not in range(3)
            or type(config['box_padding']) is not int or config['box_padding'] < 0
            or type(config['box_max_extent']) is not int or config['box_max_extent'] < 1
            or not 0 <= config['tolerance'] < 1):
        raise ValueError('Invalid protocol values')
    config['model_folder'] = str(Path(config['model_folder']).resolve())
    if 'correction' in config:
        correction = config['correction']
        if not isinstance(correction, dict) or set(correction) != {
            'full_image_context', 'radius_mm', 'directional', 'same_source_class', 'modality_by_head'}:
            raise ValueError('correction requires full_image_context/radius_mm/directional/same_source_class/modality_by_head')
        if any(type(correction[k]) is not bool for k in ('full_image_context', 'directional', 'same_source_class')):
            raise ValueError('Correction flags must be bool')
        if (isinstance(correction['radius_mm'], bool) or not isinstance(correction['radius_mm'], (int, float))
                or not 0 < correction['radius_mm'] <= 64):
            raise ValueError('Correction radius must be in (0,64] mm')
        mapping = correction['modality_by_head']
        if (not isinstance(mapping, dict) or set(mapping) != {'WT', 'TC', 'ET'}
                or any(type(v) is not int or v not in range(4) for v in mapping.values())):
            raise ValueError('modality_by_head must map WT/TC/ET to available channel indices 0..3')
    return config


def implementation_identity():
    folder = Path(__file__).parent
    return {name: digest(folder / name) for name in (
        'evaluation.py', 'evaluation_manifest.py', 'prompt_sampling.py', 'interaction_metrics.py',
        'backends.py', 'local_backend.py', 'session.py', 'schemas.py', 'prompts.py')}


def model_identity(folder):
    folder = Path(folder)
    paths = [folder / 'fold_0' / 'checkpoint_final.pth', folder / 'plans.json', folder / 'dataset.json']
    return {str(p.relative_to(folder)): digest(p) for p in paths}


def scientific_identity(manifest, config, kind):
    initial = manifest['initialization']
    return {'protocol': config, 'prompt_kind': kind,
            'initial_checkpoint_sha256': initial['checkpoint_sha256'],
            'initial_tta': initial['tta'],
            'initial_mirror_axes': initial.get('mirror_axes'),
            'initial_tile_step': initial.get('tile_step'),
            'initial_input_protocol': initial.get('input_protocol'),
            'initial_network_sha256': initial.get('network_sha256'),
            'nninteractive_model': model_identity(config['model_folder']),
            'implementation': implementation_identity(),
            'semantics': 'gt_oracle_hierarchical_transition_v1',
            'hard_click_constraints': True, 'replay_box_history': True,
            'initial_logits': 'hard_mask_to_logit_pm4.59512'}
