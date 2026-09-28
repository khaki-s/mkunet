"""Case-wise internal-label Dice, empty-NCR FP and paired bootstrap.

Use nnU-Net exported NIfTI labels 0/1/2/3, not raw BraTS labels 0/1/2/4.
Prediction IDs define the evaluated cohort; optional --split guards fold 0.
"""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


REGIONS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,), 'ED': (1,), 'NCR': (2,)}


def score_case(gt_path, pred_path):
    gi, pi = nib.load(str(gt_path)), nib.load(str(pred_path))
    gt, pred = np.asarray(gi.dataobj), np.asarray(pi.dataobj)
    if gt.shape != pred.shape or not np.allclose(gi.affine, pi.affine, atol=1e-4):
        raise ValueError(f'Shape/affine mismatch: {pred_path.name}')
    if not np.isin(gt, (0, 1, 2, 3)).all() or not np.isin(pred, (0, 1, 2, 3)).all():
        raise ValueError('Expected internal labels 0/1/2/3; raw BraTS labels need conversion first')
    voxel_mm3 = float(abs(np.linalg.det(gi.affine[:3, :3])))
    result = {'case': pred_path.name.removesuffix('.nii.gz')}
    for name, labels in REGIONS.items():
        ref, out = np.isin(gt, labels), np.isin(pred, labels)
        nr, npred = int(ref.sum()), int(out.sum())
        result[f'{name}_dice'] = 2 * np.count_nonzero(ref & out) / (nr + npred) if nr + npred else 1.
        result[f'{name}_ref_mm3'] = nr * voxel_mm3
        result[f'{name}_fp_mm3'] = int(np.count_nonzero(out & ~ref)) * voxel_mm3
    for name, label in (('ED', 1), ('ET', 3)):
        result[f'NCR_to_{name}_mm3'] = int(np.count_nonzero((gt == 2) & (pred == label))) * voxel_mm3
    return result


def main(args):
    pred, gt = Path(args.pred), Path(args.gt)
    files = sorted(pred.glob('*.nii.gz'))
    if not files:
        raise ValueError('No prediction NIfTI files')
    ids = {p.name.removesuffix('.nii.gz') for p in files}
    if args.split:
        wanted = set(json.loads(Path(args.split).read_text())[args.fold]['val'])
        if ids != wanted:
            raise ValueError(f'Prediction IDs differ from val split: missing={wanted-ids}, extra={ids-wanted}')
    rows, baseline_rows = [], []
    for index, path in enumerate(files):
        rows.append(score_case(gt / path.name, path))
        if args.baseline:
            baseline_rows.append(score_case(gt / path.name, Path(args.baseline) / path.name))
        if (index + 1) % 20 == 0:
            print(f'Evaluated {index + 1}/{len(files)}', flush=True)
    summary = {'n_cases': len(rows), 'empty_both_dice': 1.0, 'regions': {}}
    rng = np.random.RandomState(20260913)
    for name in REGIONS:
        dice = np.array([r[f'{name}_dice'] for r in rows])
        nonempty = np.array([r[f'{name}_ref_mm3'] > 0 for r in rows])
        item = {'mean': float(dice.mean()), 'p10': float(np.quantile(dice, .1)),
                'median': float(np.median(dice)), 'n_below_0_5': int((dice < .5).sum()),
                'nonempty_mean': float(dice[nonempty].mean()) if nonempty.any() else None,
                'n_empty_gt': int((~nonempty).sum()),
                'empty_gt_fp_mm3_mean': float(np.mean([r[f'{name}_fp_mm3'] for r, ne in zip(rows, nonempty) if not ne])) if (~nonempty).any() else None}
        if baseline_rows:
            delta = dice - np.array([r[f'{name}_dice'] for r in baseline_rows])
            means = delta[rng.randint(0, len(delta), size=(2000, len(delta)))].mean(1)
            item['paired_delta'] = float(delta.mean())
            item['paired_bootstrap_ci95'] = np.quantile(means, [.025, .975]).tolist()
        summary['regions'][name] = item
    summary['NCR_volume_groups'] = {}
    for name, lower, upper in [('small_0_to_1000', 0, 1000), ('medium_1000_to_10000', 1000, 10000), ('large_gt10000', 10000, float('inf'))]:
        group = [r['NCR_dice'] for r in rows if lower < r['NCR_ref_mm3'] <= upper]
        summary['NCR_volume_groups'][name] = {'n': len(group), 'mean': float(np.mean(group)) if group else None}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cases.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--baseline')
    parser.add_argument('--split')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
