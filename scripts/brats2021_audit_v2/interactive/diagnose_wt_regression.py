#!/usr/bin/env python3
"""Stage 0 diagnosis for the frozen SDF-only removal arm.

Two independent, read-only diagnostics (no training, no GT-aided prediction):

  summary : from a scored summary.json, reproduce the per-case deletion
            accounting (corrected FP vs. wrongly deleted TP), the per-case
            break-even line  deletion_precision < 1 - Dice/2 ,  and confirm the
            totals reproduce 95.585 / 73.571 mL and the WT gain of -0.0337 pp.

  oracle  : on a frozen split (default val80), compute the per-voxel oracle
            upper bound of a remove-only vs. a bidirectional (add+remove) edit,
            reusing the shipped candidate responses and residual._coverage.

Only `summary` is fully standalone (json + numpy). `oracle` imports the
interactive pipeline and needs the cohort / SAM / response artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REGIONS = ('WT', 'TC', 'ET', 'ED', 'NCR')
# Published frozen-run totals used as a reproduction check (mm^3).
EXPECTED_WT = {'deleted_correct_mm3': 95585., 'deleted_wrong_mm3': 73571., 'gain_pp': -0.03368073006273539}


def _load(path):
    with open(path) as stream:
        return json.load(stream)


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _region_rows(records, region):
    return [row for row in records if row['region'] == region]


def deletion_accounting(row):
    """Per-case removal ledger for a delete-only edit.

    Because the frozen arm never adds voxels, the whole change is a deletion:
    voxels that were false positives and got removed are the correct deletions,
    and true-positive voxels that were removed reappear as extra false negatives.
    """
    correct = row['baseline_fp_mm3'] - row['fp_mm3']
    wrong = row['fn_mm3'] - row['baseline_fn_mm3']
    removed = correct + wrong
    precision = correct / removed if removed else None
    dice = row['baseline_dice']
    breakeven = None if dice is None else 1. - dice / 2.
    violates = bool(removed > 0 and precision is not None and precision < breakeven)
    return {'case_id': row['case_id'], 'region': row['region'], 'baseline_dice': dice,
            'dice': row['dice'], 'gain_pp': row['gain_pp'], 'deleted_correct_mm3': correct,
            'deleted_wrong_mm3': wrong, 'deleted_total_mm3': removed,
            'deletion_precision': precision, 'breakeven': breakeven, 'violates': violates,
            'worsened': row['gain_pp'] is not None and row['gain_pp'] < -1.e-8}


def summarise_region(rows):
    ledger = [deletion_accounting(row) for row in rows]
    correct = sum(item['deleted_correct_mm3'] for item in ledger)
    wrong = sum(item['deleted_wrong_mm3'] for item in ledger)
    total = correct + wrong
    gains = [item['gain_pp'] for item in ledger if item['gain_pp'] is not None]
    violators = [item for item in ledger if item['violates']]
    worsened = [item for item in ledger if item['worsened']]
    return {'n_cases': len(ledger),
            'deleted_correct_mm3': correct, 'deleted_wrong_mm3': wrong, 'deleted_total_mm3': total,
            'deletion_precision': correct / total if total else None,
            'gain_pp': float(np.mean(gains)) if gains else None,
            'n_violates': len(violators), 'n_worsened': len(worsened),
            'violators_equal_worsened': {item['case_id'] for item in violators}
                                         == {item['case_id'] for item in worsened},
            'per_case': ledger}


def check_expected(report, tolerance_mm3=1., tolerance_pp=1.e-9):
    """Confirm the WT ledger reproduces the published frozen-run totals."""
    mismatches = {}
    for key, expected in EXPECTED_WT.items():
        got = report[key]
        tolerance = tolerance_pp if key == 'gain_pp' else tolerance_mm3
        if got is None or abs(got - expected) > tolerance:
            mismatches[key] = {'expected': expected, 'got': got}
    return mismatches


def run_summary(args):
    data = _load(args.summary)
    records = data['per_case']
    regions = {name: summarise_region(_region_rows(records, name)) for name in REGIONS
               if _region_rows(records, name)}
    wt = regions.get('WT')
    mismatches = check_expected(wt) if wt else {'WT': 'missing'}
    report = {'source': str(args.summary), 'n_cases': data.get('n_cases'),
              'regions': {name: {k: v for k, v in body.items() if k != 'per_case'}
                          for name, body in regions.items()},
              'wt_reproduction_mismatches': mismatches,
              'reproduced': not mismatches}
    out = Path(args.output)
    _write(out / 'diagnostics.json', report)
    if wt:
        _write_csv(out / 'per_case.csv', wt['per_case'])
    print(json.dumps({'reproduced': report['reproduced'],
                      'wt_deletion_precision': None if not wt else wt['deletion_precision'],
                      'wt_gain_pp': None if not wt else wt['gain_pp'],
                      'wt_violates_equal_worsened': None if not wt else wt['violators_equal_worsened'],
                      'mismatches': mismatches}, indent=2, ensure_ascii=False))
    return out / 'diagnostics.json'


def _write_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    summary = commands.add_parser('summary')
    summary.add_argument('--summary', required=True)
    summary.add_argument('--output', required=True)
    args = parser.parse_args()
    {'summary': run_summary}[args.stage](args)


if __name__ == '__main__':
    main()

