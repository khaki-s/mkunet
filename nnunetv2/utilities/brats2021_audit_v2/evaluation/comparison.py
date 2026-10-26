"""Strict, read-only comparison of completed missing-modality evaluations."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REGIONS = ('WT', 'TC', 'ET', 'ED', 'NCR')
MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')
CODES = (1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15)
GROUPS = {'all15': CODES, 'missing14': CODES[:-1], 'full': (15,),
          **{f'k{k}': tuple(c for c in CODES if c.bit_count() == k) for k in range(1, 5)},
          'without_T1ce': tuple(c for c in CODES if not c & 2)}
PROTOCOL_KEYS = ('protocol', 'modalities', 'regions', 'tile_step', 'mirroring',
                 'mirror_axes', 'gt_sha256', 'empty_both_dice', 'hd95_empty_policy', 'accumulation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def mean_defined(a, axis=None):
    valid = np.isfinite(a)
    count = valid.sum(axis=axis)
    return np.divide(np.where(valid, a, 0).sum(axis=axis), count,
                     out=np.full(np.shape(count), np.nan), where=count > 0)


def load_run(folder, expected_ids):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    if manifest.get('status') != 'complete':
        raise ValueError(f'Incomplete evaluation: {folder}')
    contract = manifest['contract']
    if any(key not in contract for key in PROTOCOL_KEYS):
        raise ValueError(f'Incomplete protocol contract: {folder}')
    ids = contract['case_ids']
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError(f'Case IDs differ from the fixed cohort: {folder}')
    if len(contract['mask_codes']) != 15 or set(contract['mask_codes']) != set(CODES):
        raise ValueError(f'Expected all 15 masks: {folder}')
    if tuple(contract['modalities']) != MODALITIES or contract['regions'] != {
            'WT': [1, 2, 3], 'TC': [2, 3], 'ET': [3], 'ED': [1], 'NCR': [2]}:
        raise ValueError(f'Modality/label semantics differ: {folder}')
    provenance = contract['provenance']
    if provenance.get('split') != 'test' or provenance.get('fold') != 0:
        raise ValueError(f'This report requires fold-0 test outputs: {folder}')
    if not provenance.get('checkpoint_sha256') or not provenance.get('inputs'):
        raise ValueError(f'Missing checkpoint/input provenance: {folder}')
    if set(provenance['inputs']) != set(expected_ids) or set(contract['gt_sha256']) != set(expected_ids):
        raise ValueError(f'Incomplete input/GT fingerprints: {folder}')
    values = {}
    rows = []
    with (folder / 'per_case_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = row['case_id'], int(row['mask_code']), row['region']
            if key in values:
                raise ValueError(f'Duplicate metric key: {key}')
            if int(row['n_modalities']) != key[1].bit_count():
                raise ValueError(f'Incorrect cardinality: {key}')
            value = float(row['dice']) if row['dice'] else np.nan
            if np.isfinite(value) and not 0 <= value <= 1:
                raise ValueError(f'Invalid Dice: {key}')
            if not np.isfinite(value) and not (float(row['ref_mm3']) == float(row['pred_mm3']) == 0):
                raise ValueError(f'Undefined Dice without both-empty segmentation: {key}')
            values[key] = value
            rows.append(row)
    expected = {(c, m, r) for c in expected_ids for m in CODES for r in REGIONS}
    if set(values) != expected:
        raise ValueError(f'Missing/unexpected metric keys: {folder}; expected {len(expected)}, got {len(values)}')
    array = np.array([[[values[c, m, r] for r in REGIONS] for m in CODES] for c in expected_ids])
    if not np.isfinite(array[:, :, :3]).all():
        raise ValueError('Primary WT/TC/ET joint metric requires all scores to be defined')
    training = {name: read_json(folder.parent / name) for name in
                ('audit_experiment.json', 'agent_experiment.json') if (folder.parent / name).exists()}
    return {'folder': str(folder.resolve()), 'contract': contract, 'values': array, 'rows': rows,
            'training': training, 'metric_sha256': digest(folder / 'per_case_metrics.csv'),
            'manifest_sha256': digest(folder / 'manifest.json')}


def check_protocol(current, reference):
    a, b = current['contract'], reference['contract']
    checks = {k: a.get(k) == b.get(k) for k in PROTOCOL_KEYS}
    checks['source_input_fingerprints'] = a['provenance']['inputs'] == b['provenance']['inputs']
    checks['checkpoint_selection_filename'] = (Path(a['provenance']['checkpoint']).name ==
                                                 Path(b['provenance']['checkpoint']).name)
    if not all(checks.values()):
        raise ValueError(f'Evaluation protocol mismatch: {[k for k,v in checks.items() if not v]}')
    return checks


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(folders, expected_ids, output, *, reference, repetitions=10000, seed=20260930):
    if len(expected_ids) != len(set(expected_ids)) or not expected_ids:
        raise ValueError('Cohort must contain unique IDs')
    if repetitions < 2:
        raise ValueError('At least two bootstrap repetitions required')
    expected_ids = sorted(expected_ids)
    runs = {name: load_run(folder, expected_ids) for name, folder in folders.items()}
    if reference not in runs:
        raise ValueError('Reference is not among the provided runs')
    protocols = {name: check_protocol(run, runs[reference]) for name, run in runs.items()}
    summaries, by_mask, paired = [], [], []
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(expected_ids), size=(repetitions, len(expected_ids)))
    for name, run in runs.items():
        a = run['values']
        mask_means = mean_defined(a, axis=0)
        for j, code in enumerate(CODES):
            by_mask.append({'model': name, 'mask_code': code, 'modalities': '+'.join(
                m for i,m in enumerate(MODALITIES) if code >> i & 1),
                **{f'{r}_dice': finite_or_none(mask_means[j,i]) for i,r in enumerate(REGIONS)}})
        for group, codes in GROUPS.items():
            js = [CODES.index(c) for c in codes]
            s = mean_defined(mask_means[js], axis=0)
            row = {'model': name, 'group': group, 'n_cases': len(expected_ids), 'n_masks': len(js),
                   **{f'{r}_dice': finite_or_none(s[i]) for i,r in enumerate(REGIONS)},
                   'joint_WT_TC_ET': float(s[:3].mean())}
            summaries.append(row)
            if name == reference:
                continue
            # Paired case bootstrap: retain every mask/region of each sampled case.
            delta = a[:,js,:3] - runs[reference]['values'][:,js,:3]
            for i, region in enumerate(('WT', 'TC', 'ET', 'joint_WT_TC_ET')):
                per_case = delta[:,:,i].mean(1) if i < 3 else delta.mean((1,2))
                boot = per_case[indices].mean(1)
                low, high = np.quantile(boot, [.025, .975])
                paired.append({'model': name, 'reference': reference, 'group': group, 'region': region,
                               'delta_pp': float(per_case.mean()*100), 'ci95_low_pp': float(low*100),
                               'ci95_high_pp': float(high*100), 'n_improved': int((per_case>0).sum()),
                               'n_worse': int((per_case<0).sum()), 'n_tied': int((per_case==0).sum())})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new report directory; refusing to overwrite: {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'summary.csv', summaries)
    write_csv(output/'by_mask.csv', by_mask)
    if paired:
        write_csv(output/'paired_vs_reference.csv', paired)
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'n_cases': len(expected_ids),
                  'case_ids': expected_ids, 'reference': reference, 'repetitions': repetitions, 'seed': seed,
                  'resampling_unit': 'case; masks and regions kept together', 'protocol_checks': protocols,
                  'scope': 'Observed test-set differences; case CIs do not include training-seed uncertainty.',
                  'runs': {name: {k:v for k,v in run.items() if k not in ('values','rows')} for name,run in runs.items()}}
    write_json(output/'comparison_provenance.json', provenance)
    lines = ['# 完整 test100×15 比较', '', '单位：Dice ×100。以下按组合等权，病例配对；不代表通信因果收益。', '',
             '| 模型 | 分组 | WT | TC | ET | 三区域均值 |', '|---|---|---:|---:|---:|---:|']
    for row in summaries:
        if row['group'] in ('missing14','all15','full'):
            lines.append('| '+ ' | '.join([row['model'],row['group']]+[f"{100*row[k]:.4f}" for k in
                         ('WT_dice','TC_dice','ET_dice','joint_WT_TC_ET')])+' |')
    lines += ['', f'参照模型：`{reference}`；病例 bootstrap {repetitions:,} 次，seed={seed}。', '',
              '| 比较 | 分组 | 三区域均值差（百分点） | 95% CI |', '|---|---|---:|---|']
    for row in paired:
        if row['region']=='joint_WT_TC_ET' and row['group'] in ('missing14','all15','full'):
            lines.append(f"| {row['model']} − {reference} | {row['group']} | {row['delta_pp']:+.4f} | [{row['ci95_low_pp']:+.4f}, {row['ci95_high_pp']:+.4f}] |")
    lines += ['', '完整逐组合表见 `by_mask.csv`；各区域/模态数量的差值及区间见 `paired_vs_reference.csv`。',
              '协议相同不等于训练方法完全相同：ExclusiveLoss、专家分支与通信必须按训练元数据分别解释。',
              '多个次要比较未做多重比较校正；本报告没有依据 test 重新选择 checkpoint。', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    return summaries, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='NAME=/path/to/completed/test100_15_modalities')
    parser.add_argument('--reference', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260930)
    args = parser.parse_args()
    folders = {}
    for value in args.run:
        name, path = value.split('=', 1)
        if not name or name in folders:
            parser.error('Run names must be nonempty and unique')
        folders[name] = path
    ids = [c['case_id'] for c in read_json(args.manifest)['cases']]
    split = read_json(args.splits)[0]
    if len(ids) != 100 or set(ids) & set(split['train'] + split['val']):
        parser.error('Expected 100 internal test cases disjoint from train and validation')
    compare_runs(folders, ids, args.output, reference=args.reference,
                 repetitions=args.repetitions, seed=args.seed)
    print(f'Comparison written to {args.output}')


if __name__ == '__main__':
    main()
