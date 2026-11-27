"""Cache frozen SAM candidates on the original volume grid, without opening GT."""
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch

from .. import sam_dense
from ..common import check_file, file_identity, load_case, manifest, read_json, read_labels, same_grid, write_json
from ..sam_fusion_pipeline import contract


def cohort(args):
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    train = manifest(args.train_manifest, allowed=('train',))
    validation = manifest(args.val_manifest, deployment=True, allowed=('validation',))
    if train['splits_file'] != validation['splits_file'] or train['fold'] != validation['fold']:
        raise ValueError('Training and validation must originate from the same fold')
    train_rows = train['cases'][:args.train_cases]
    val_rows = validation['cases'][args.val_start:args.val_start + args.val_cases]
    if len(train_rows) != args.train_cases or len(val_rows) != args.val_cases:
        raise ValueError('Cohort exceeds manifest')
    result = {'schema': 'sam_component_cohort_v1', 'train_manifest': file_identity(args.train_manifest),
              'val_manifest': file_identity(args.val_manifest), 'train': [row['case_id'] for row in train_rows],
              'validation': [row['case_id'] for row in val_rows]}
    write_json(root, result)
    return root


def rows_for(path, splits, *, truth=False):
    value = read_json(path)
    if value.get('schema') != 'sam_component_cohort_v1':
        raise ValueError('Expected frozen cohort')
    records = []
    for split in splits:
        identity = value[{'train': 'train_manifest', 'validation': 'val_manifest'}[split]]
        check_file(identity)
        selected = manifest(identity['path'], deployment=split == 'validation', allowed=(split,))
        cases = {row['case_id']: row for row in selected['cases']}
        for key in value[split]:
            row = cases[key]
            records.append(row if truth and split == 'train' else {k:v for k,v in row.items() if k != 'gt'})
    return records


def cache_masks(generated):
    return np.stack([[generated[variant][head] for head in sam_dense.HEADS] for variant in sam_dense.VARIANTS])


def load_masks(root, row):
    folder = Path(root) / row['case_id']
    entry = read_json(folder / 'candidate.json')
    if entry['row'] != row:
        raise ValueError(f'Dense candidate inputs differ for {row["case_id"]}')
    check_file(entry['masks'])
    with np.load(entry['masks']['path'], allow_pickle=False) as arrays:
        masks = arrays['masks']
    if masks.shape[:2] != (len(sam_dense.VARIANTS), len(sam_dense.HEADS)):
        raise ValueError('Wrong SAM candidate variant/head ordering')
    return masks.astype(bool, copy=False), entry


def dense_cache(args):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index')
    subset = rows_for(args.cohort, ('train', 'validation'))[args.shard_index::args.num_shards]
    frozen = read_json(args.cohort)
    source = manifest(frozen['train_manifest']['path'], allowed=('train',))
    method = contract(source)
    reuse = read_json(args.reuse) if args.reuse else None
    if reuse and reuse['binding']['contract'] != method:
        raise ValueError('Collection reuse requires identical frozen SAM contract')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    engine = None
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = monotonic()
    for row in subset:
        key = row['case_id']
        folder = root / key
        receipt = folder / 'candidate.json'
        if receipt.exists():
            _, old = load_masks(root, row)
            if old['sam'] != method:
                raise ValueError(f'{key}: Existing SAM cache is bound to a different model')
            print(f'SAM cached {key}', flush=True)
            continue
        folder.mkdir(exist_ok=True)
        if reuse and key in reuse['completed']:
            check_file(row['prediction'])
            reference = nib.load(row['prediction']['path'])
            candidates = reuse['completed'][key]['pairs']
            masks = []
            for variant in sam_dense.VARIANTS:
                heads = sorted(candidates[variant], key=lambda item: item['head'])
                result = []
                for h in heads:
                    check_file(h['candidate'])
                    image = nib.load(h['candidate']['path'])
                    same_grid(image, reference)
                    result.append(np.asarray(image.dataobj, dtype=bool))
                masks.append(np.stack(result))
            masks = np.stack(masks)
            calls = reuse['completed'][key]['sam_calls']
        else:
            mri, baseline, _ = load_case(row)
            if engine is None:
                engine = sam_dense.SamDenseRefiner(args.device)
            generated, details = engine.generate(mri, baseline)
            masks = cache_masks(generated)
            calls = details['sam_predict_calls']
            del mri, baseline, generated
        temporary = folder / 'masks.tmp.npz'
        np.savez_compressed(temporary, masks=masks)
        masks_path = folder / 'masks.npz'
        temporary.replace(masks_path)
        write_json(receipt, {'row': row, 'masks': file_identity(masks_path), 'sam': method,
                             'native_shape': list(masks.shape[2:]), 'calls': calls, 'gt_used': False})
        print(f'SAM ready {key} calls={calls} elapsed={monotonic()-started:.1f}s', flush=True)
    result = {'cases': [r['case_id'] for r in subset], 'seconds': monotonic() - started}
    if device.type == 'cuda':
        result.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    stats = root / f'dense_shard{args.shard_index}.json'
    write_json(stats, result)
    return stats


def diagnose_dense(args):
    """Compare identical point/box prompts with or without the soft mask on train only."""
    output = []
    for row in rows_for(args.cohort, ('train',), truth=True):
        clean = {k: v for k, v in row.items() if k != 'gt'}
        masks, evidence = load_masks(args.dense, clean)
        check_file(row['prediction'])
        check_file(row['gt'])
        before, truth = (nib.load(row[k]['path']) for k in ('prediction', 'gt'))
        same_grid(before, truth)
        baseline, gt = read_labels(before), read_labels(truth)
        for h, head in enumerate(sam_dense.HEADS):
            b, y = baseline > h, gt > h
            original_size = np.count_nonzero(b) + np.count_nonzero(y)
            original = 200. * np.count_nonzero(b & y) / original_size if original_size else 100.
            for v, variant in enumerate(sam_dense.VARIANTS):
                prediction = masks[v, h]
                dice_size = np.count_nonzero(prediction) + np.count_nonzero(y)
                dice = 200. * np.count_nonzero(prediction & y) / dice_size if dice_size else 100.
                add = prediction & ~b
                remove = b & ~prediction
                output.append({'case': row['case_id'], 'head': head, 'sam_prompt': variant, 'gain_pp': dice - original,
                               'false_negative_recovered': int(np.count_nonzero(add & y)),
                               'false_positive_added': int(np.count_nonzero(add & ~y)),
                               'false_positive_removed': int(np.count_nonzero(remove & ~y)),
                               'true_positive_removed': int(np.count_nonzero(remove & y)),
                               'sam_model': evidence['sam']['sam_checkpoint']})
    summary = {}
    for head in sam_dense.HEADS:
        summary[head] = {}
        for variant in sam_dense.VARIANTS:
            selected = [r for r in output if r['head'] == head and r['sam_prompt'] == variant]
            summary[head][variant] = {'mean_raw_dice_gain_pp': float(np.mean([r['gain_pp'] for r in selected])),
                **{field: sum(r[field] for r in selected) for field in ('false_negative_recovered',
                    'false_positive_added', 'false_positive_removed', 'true_positive_removed')}}
    write_json(args.output, {'train_only': True, 'summary': summary, 'per_case': output})
    return args.output
