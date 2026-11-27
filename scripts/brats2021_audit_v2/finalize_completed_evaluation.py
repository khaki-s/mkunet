#!/usr/bin/env python3
"""Finish summaries from verified completion receipts; never load/run a model.

Use only after the original exporter has exited. The original inference
contract is preserved, and the summary-only recovery has separate provenance.
Missing or changed predictions/inputs are errors, never re-inferred here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import (
    REGIONS, REMOVAL_DELTA_FIELDS, file_sha256, removal_deltas, summarize,
    write_csv, write_json)


def finalize(folder, *, check_only=False):
    folder = Path(folder)
    path = folder / 'manifest.json'
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    if original.get('status') not in ('running', 'complete'):
        raise ValueError('Require an existing running/complete evaluation manifest')
    contract = original['contract']
    keys, codes = contract['case_ids'], contract['mask_codes']
    if (not keys or len(set(keys)) != len(keys) or not codes or len(set(codes)) != len(codes)
            or any(Path(key).name != key or key in ('.', '..') for key in keys)):
        raise ValueError('Invalid case/mask contract')
    origin = contract['provenance']
    if file_sha256(origin['checkpoint']) != origin['checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    metrics, routes, receipts = [], [], {}
    for key in keys:
        gt = Path(contract['gt_folder']) / f'{key}.nii.gz'
        if file_sha256(gt) != contract['gt_sha256'][key]:
            raise ValueError(f'GT changed: {key}')
        for identity in origin.get('inputs', {}).get(key, {}).values():
            if file_sha256(identity['path']) != identity['sha256']:
                raise ValueError(f'MRI changed: {key}')
        for code in codes:
            done = folder / 'completed' / f'mask_{code:02d}' / f'{key}.json'
            prediction = folder / 'predictions' / f'mask_{code:02d}' / f'{key}.nii.gz'
            record = json.loads(done.read_text())
            if file_sha256(prediction) != record['prediction_sha256']:
                raise ValueError(f'Prediction changed: {prediction}')
            rows = record['metrics']
            if (len(rows) != len(REGIONS) or {r['region'] for r in rows} != set(REGIONS)
                    or any(r['case_id'] != key or r['mask_code'] != code for r in rows)):
                raise ValueError(f'Incomplete/mismatched score receipt: {done}')
            if any(r['case_id'] != key or r['mask_code'] != code for r in record['routing']):
                raise ValueError(f'Mismatched routing receipt: {done}')
            metrics.extend(rows)
            routes.extend(record['routing'])
            receipts[str(done.relative_to(folder))] = file_sha256(done)
        print(f'Checked {key}', flush=True)
    by_mask, by_count = summarize(metrics, codes)
    summary = {'n_cases': len(keys), 'n_masks': len(codes),
               'by_mask': by_mask, 'by_cardinality': by_count}
    if check_only:
        return summary
    if path.read_bytes() != original_bytes:
        raise RuntimeError('Source manifest changed during verification; do not finalize a running exporter')
    if original['status'] == 'complete':
        if json.loads((folder / 'summary.json').read_text()) != summary:
            raise ValueError('Complete evaluation summary differs from its receipts')
        return summary
    backup = folder / 'manifest.before_summary_finalization.json'
    if backup.exists() and backup.read_bytes() != original_bytes:
        raise ValueError('Existing finalization backup differs; inspect before retrying')
    if not backup.exists():
        with backup.open('xb') as stream:
            stream.write(original_bytes)
    write_csv(folder / 'per_case_metrics.csv', metrics)
    write_csv(folder / 'summary_by_mask.csv', by_mask)
    write_csv(folder / 'summary_by_cardinality.csv', by_count)
    write_csv(folder / 'routing_per_case.csv', routes,
              fields=None if routes else ['case_id', 'mask_code', 'region', 'modality', 'weight'])
    write_csv(folder / 'input_removal_delta.csv', removal_deltas(metrics), fields=REMOVAL_DELTA_FIELDS)
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'summary_finalization_audit.json', {
        'utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'finish summaries after all case/mask predictions already completed',
        'original_manifest_backup': backup.name,
        'original_manifest_sha256': file_sha256(backup),
        'script_sha256': file_sha256(__file__),
        'summary_module_sha256': file_sha256(
            Path(__file__).parents[2] / 'nnunetv2/utilities/brats2021_audit_v2/agent_evaluate.py'),
        'verified_receipt_sha256': receipts,
        'predictions_recomputed': False, 'model_loaded': False,
        'inference_contract_modified': False,
    })
    if path.read_bytes() != original_bytes:
        raise RuntimeError('Source manifest changed before final commit')
    write_json(path, dict(original, status='complete'))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = finalize(args.output, check_only=args.check_only)
    print(json.dumps({k: result[k] for k in ('n_cases', 'n_masks')}))
