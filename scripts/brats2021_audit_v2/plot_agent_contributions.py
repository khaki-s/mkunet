#!/usr/bin/env python
"""Render actual CSV values; never substitutes illustrative routing numbers."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nnunetv2.utilities.brats2021_audit.missing_modality import MASK_CODES, MODALITIES


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def grid(rows, region, modalities, field, modality_key='modality'):
    result = np.full((15, 4), np.nan)
    for i, code in enumerate(MASK_CODES):
        for j, name in enumerate(modalities):
            values = [float(r[field]) for r in rows if int(r['mask_code']) == code and r['region'] == region
                      and r[modality_key] == name and r[field] not in ('', None)]
            if values:
                result[i, j] = np.mean(values)
    return result


def panels(matrices, titles, columns, output, *, signed=False, caption='', note=''):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 9), squeeze=False, constrained_layout=True)
    max_abs = max((np.nanmax(np.abs(a)) for a in matrices if np.isfinite(a).any()), default=1)
    bound = max(max_abs, 1e-6)
    for ax, matrix, title in zip(axes[0], matrices, titles):
        cmap = plt.get_cmap('RdBu_r' if signed else 'Greens').copy(); cmap.set_bad('#dddddd')
        image = ax.imshow(matrix, cmap=cmap, vmin=-bound if signed else 0, vmax=bound if signed else 1, aspect='auto')
        ax.set_xticks(range(len(columns)), columns, rotation=30)
        ax.set_yticks(range(15), [f'{c:02d}: ' + ''.join('●' if (c >> MODALITIES.index(m)) & 1 else '○' for m in columns[:4]) for c in MASK_CODES])
        ax.set_title(title)
        for i in range(15):
            for j in range(matrix.shape[1]):
                v = matrix[i, j]
                ax.text(j, i, 'N/A' if not np.isfinite(v) else (f'{v:+.2f}' if signed else f'{v:.2f}'),
                        ha='center', va='center', fontsize=8, color='white' if np.isfinite(v) and abs(v) > (.6 * bound if signed else .65) else 'black')
    if note:
        fig.suptitle(note, fontsize=14)
    fig.colorbar(image, ax=list(axes[0]), shrink=.7)
    fig.supxlabel(caption, fontsize=9)
    for ending in ('png', 'pdf'):
        fig.savefig(str(output) + '.' + ending, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--reference-column-order', action='store_true', help='FLAIR/T1/T1ce/T2 as in supplied figure')
    args = parser.parse_args()
    modalities = ('FLAIR', 'T1', 'T1ce', 'T2') if args.reference_column_order else MODALITIES
    manifest = args.evaluation / 'manifest.json'
    metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
    fixture = metadata.get('contract', {}).get('provenance', {}).get('fixture', '')
    note = 'Synthetic verification fixture — not research results' if 'synthetic' in fixture.lower() else ''
    rows = read_csv(args.evaluation / 'routing_per_case.csv')
    if rows:
        keys = [(r['case_id'], r['mask_code'], r['region'], r['modality']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Expected one row per case/mask/region/modality, not raw window records')
        panels([grid(rows, r, modalities, 'weight') for r in ('WT', 'TC', 'ET')], ('WT routing', 'TC routing', 'ET routing'),
               modalities, args.evaluation / 'routing_heatmap', caption='Case-equal means of patch/TTA routing; single-modality 1 is imposed by the mask.', note=note)
        panels([grid(rows, r, modalities, 'effective_weight') for r in ('WT', 'TC', 'ET')], ('WT beta × weight', 'TC beta × weight', 'ET beta × weight'),
               modalities, args.evaluation / 'effective_weights', caption='Direct logit fusion coefficients; these are not causal modality contributions.', note=note)
        beta = np.array([[np.mean([float(r['beta']) for r in rows if int(r['mask_code']) == code and r['region'] == region])
                          if any(int(r['mask_code']) == code and r['region'] == region for r in rows) else np.nan
                          for region in ('WT', 'TC', 'ET')] for code in MASK_CODES])
        fig, ax = plt.subplots(figsize=(4.5, 8), constrained_layout=True)
        im = ax.imshow(beta, vmin=0, vmax=1, cmap='Greens', aspect='auto')
        ax.set_xticks(range(3), ('WT', 'TC', 'ET')); ax.set_yticks(range(15), MASK_CODES); ax.set_ylabel('mask_code'); ax.set_title('Case-equal beta')
        if note:
            fig.suptitle(note, fontsize=9)
        fig.colorbar(im, ax=ax); fig.savefig(args.evaluation / 'beta_heatmap.pdf'); plt.close(fig)
    delta = read_csv(args.evaluation / 'input_removal_delta.csv')
    matrices = [100 * grid(delta, r, modalities, 'delta_dice', 'removed') for r in ('WT', 'TC', 'ET', 'ED', 'NCR')]
    panels(matrices, ('WT Δ Dice (pp)', 'TC Δ Dice (pp)', 'ET Δ Dice (pp)', 'ED Δ Dice (pp)', 'NCR Δ Dice (pp)'),
           modalities, args.evaluation / 'contribution_delta_heatmap', signed=True,
           caption='Actual input removal: Dice(S) − Dice(S without modality); negative values retained; empty input is N/A.', note=note)


if __name__ == '__main__':
    main()
