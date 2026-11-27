"""Frozen patient selections and per-modality baseline binding."""
from pathlib import Path

from ..common import MODALITIES, check_file, file_identity, manifest, read_json, write_json

SCHEMA = 'sam_selfprompt_study_v1'
TRAIN_COUNT = {'pilot': 100, 'formal': 320}
VAL_COUNT = {'pilot': 25, 'formal': 80}
VAL_OFFSET = {'pilot': 18, 'formal': 0}


def prepare(args):
    root = Path(args.root)
    train_path = root / 'manifests/train320.json'
    val_path = root / 'manifests/val80.deployment.json'
    train = manifest(train_path, allowed=('train',))
    val = manifest(val_path, deployment=True, allowed=('validation',))
    if train['splits_file'] != val['splits_file'] or train['fold'] != val['fold']:
        raise ValueError('Different patient folds')
    count, offset = VAL_COUNT[args.study], VAL_OFFSET[args.study]
    if args.val_offset is not None:
        if args.study != 'pilot' or args.val_offset < 0 or args.val_offset + count > len(val['cases']):
            raise ValueError('A custom held-out block is permitted for pilot studies only')
        offset = args.val_offset
    train_cases = [r['case_id'] for r in train['cases'][:TRAIN_COUNT[args.study]]]
    custom_train = getattr(args, 'train_list', None)
    if custom_train:
        if args.study != 'pilot':
            raise ValueError('Only a pilot may use a subset list from the original training fold')
        train_cases = read_json(custom_train)
        allowed = {row['case_id'] for row in train['cases']}
        if (not isinstance(train_cases, list) or any(not isinstance(key, str) for key in train_cases)
                or len(set(train_cases)) != TRAIN_COUNT['pilot'] or not set(train_cases) <= allowed):
            raise ValueError('The pilot list must contain 100 distinct cases from the original training split')
    val_cases = [r['case_id'] for r in val['cases'][offset:offset + count]]
    if len(train_cases) != TRAIN_COUNT[args.study] or len(val_cases) != count or set(train_cases) & set(val_cases):
        raise ValueError('Wrong cohort size or overlapping patients')
    binding = {'schema': SCHEMA, 'study': args.study, 'train': train_cases, 'validation': val_cases,
               'train_manifest': file_identity(train_path), 'validation_manifest': file_identity(val_path)}
    if custom_train:
        binding['train_list'] = file_identity(custom_train)
    if args.study == 'formal':
        from .scoring import require_training_permit
        require_training_permit(args.pilot_permit)
        binding['pilot_permit'] = file_identity(args.pilot_permit)
    path = Path(args.output)
    if path.exists():
        if read_json(path) != binding:
            raise ValueError('Existing cohort differs. Choose a new run directory')
    else:
        write_json(path, binding)
    return path


def read_cohort(path):
    value = read_json(path)
    mode = value.get('study')
    if value.get('schema') != SCHEMA or mode not in TRAIN_COUNT:
        raise ValueError('Invalid SAM study')
    for split, size in (('train', TRAIN_COUNT[mode]), ('validation', VAL_COUNT[mode])):
        if len(set(value[split])) != size or len(value[split]) != size:
            raise ValueError(f'Incorrect {split} selection')
    if set(value['train']) & set(value['validation']):
        raise ValueError('Patient sets overlap')
    if 'train_list' in value:
        check_file(value['train_list'])
        if mode != 'pilot' or read_json(value['train_list']['path']) != value['train']:
            raise ValueError('The frozen pilot training list changed')
    if mode == 'formal':
        check_file(value['pilot_permit'])
    return value


def rows(cohort_path, split, *, truth=False):
    if split not in ('train', 'validation') or (truth and split != 'train'):
        raise ValueError('Training is the only stage allowed to request labels')
    binding = read_cohort(cohort_path)
    source = binding[f'{split}_manifest']
    check_file(source)
    value = manifest(source['path'], deployment=split != 'train', allowed=(split,))
    index = {r['case_id']:r for r in value['cases']}
    return [{**{k:v for k,v in index[key].items() if truth or k != 'gt'}, 'mask_code':15}
            for key in binding[split]]


def calibration_rows(cohort_path, split, case_list=None):
    selected = rows(cohort_path, split)
    if case_list is None:
        return selected
    keys = read_json(case_list)
    by_id = {row['case_id']: row for row in selected}
    if (split != 'train' or not isinstance(keys, list) or not keys
            or any(not isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys) or not set(keys) <= by_id.keys()):
        raise ValueError('Calibration must select distinct members of the frozen training cohort')
    return [by_id[key] for key in keys]


def sam_cache_spec(train_source, approved=None):
    from . import dense

    profile = approved.get('prompt_profile') if approved else None
    if profile is None:
        config = {'sam': dense.sam_contract(train_source), 'paired_prompt_code': file_identity(dense.__file__)['sha256'],
                  'variants': list(dense.VARIANTS)}
        factory = dense.PairedSamRefiner
    else:
        from . import background

        if profile != background.PROFILE:
            raise ValueError(f'Unknown SAM prompt profile: {profile}')
        config, factory = background.contract(train_source), background.BackgroundPairedRefiner
    if approved is not None and config != approved:
        raise ValueError('The approved SAM profile, weights or source changed')
    return config, factory


def cache_formal(args):
    from . import dense
    from .scoring import require_training_permit

    group = read_cohort(args.cohort)
    if group['study'] != 'formal':
        raise ValueError('This SAM stage requires a formal cohort')
    permit = require_training_permit(group['pilot_permit']['path'])
    config, factory = sam_cache_spec(read_json(group['train_manifest']['path']), permit.get('sam_contract'))
    if getattr(args, 'vision_only', False):
        from .image_memory import vision_factory

        factory = vision_factory(factory)
    chosen = rows(args.cohort, 'train') + rows(args.cohort, 'validation')
    chosen = chosen[args.shard_index::args.num_shards]
    root = Path(args.output)
    stats = dense.build_cache(chosen, args, root, config, factory=factory)
    path = root / f'both_shard{args.shard_index}.json'
    if stats['new_cases'] or not path.exists():
        write_json(path, stats)
    return path


def reuse_pilot_cache(args):
    """Resume the identical unlabelled candidate work for overlapping cases."""
    from . import candidates, dense
    from .scoring import require_training_permit

    group = read_cohort(args.cohort)
    if group['study'] != 'formal':
        raise ValueError('Candidate reuse is for the approved formal cohort')
    permit = require_training_permit(group['pilot_permit']['path'])
    pilot_report = read_json(permit['pilot_score']['path'])
    cache_root = Path(pilot_report['cohort']['path']).parent
    output = Path(args.run_dir)
    sam_config, _ = sam_cache_spec(read_json(group['train_manifest']['path']), permit.get('sam_contract'))
    executor_config = candidates.executor_configuration()
    count = {'dense': 0, 'responses': 0}
    for split in ('train', 'validation'):
        for row in rows(args.cohort, split):
            case = row['case_id']
            sam_path = cache_root / 'dense' / case / 'sam.json'
            if not sam_path.is_file():
                continue
            sam = read_json(sam_path)
            if sam['row'] != row or sam.get('gt_used') is not False or sam['contract'] != sam_config:
                continue
            available = [('dense', sam_path.parent)]
            response_path = cache_root / 'responses' / case / 'responses.json'
            if response_path.is_file():
                response = read_json(response_path)
                if (response['row'] == row and response['sam'] == sam['arrays'] and response.get('gt_used') is False
                        and candidates.compatible_executor(response['executor'], executor_config)):
                    available.append(('responses', response_path.parent))
            for name, source in available:
                target = output / name / case
                if target.exists() or target.is_symlink():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve(), target_is_directory=True)
                count[name] += 1
    return count
