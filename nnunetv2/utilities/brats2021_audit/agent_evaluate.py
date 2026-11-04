"""Full-case all-subset evaluation, case-equal routing and paired comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

from nnunetv2.inference.export_prediction import export_prediction_from_logits
from .missing_modality import MASK_CODES, MODALITIES, mask_from_code
from .missing_predictor import MissingModalityPredictor, preprocess_available_case
from .reproducibility import state_digest

REGIONS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,), 'ED': (1,), 'NCR': (2,)}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preprocessed_input_fingerprints(folder, keys):
    folder = Path(folder)
    result = {}
    for key in keys:
        files = [folder / (key + suffix) for suffix in ('.b2nd', '.npz', '.npy', '.pkl')]
        present = [p for p in files if p.is_file()]
        if not any(p.suffix in ('.b2nd', '.npz', '.npy') for p in present) or not (folder / (key + '.pkl')).is_file():
            raise FileNotFoundError(f'Incomplete preprocessed case: {key}')
        result[key] = {p.name: file_sha256(p) for p in present}
    return result


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def hd95(ref, pred, spacing):
    if not ref.any() or not pred.any():
        return None, 'both_empty' if not ref.any() and not pred.any() else 'one_empty'
    positions = np.where(ref | pred)
    crop = tuple(slice(max(0, int(v.min()) - 1), min(ref.shape[i], int(v.max()) + 2)) for i, v in enumerate(positions))
    r, p = ref[crop], pred[crop]
    sr, sp = r ^ binary_erosion(r), p ^ binary_erosion(p)
    distances = np.concatenate((distance_transform_edt(~sr, sampling=spacing)[sp],
                                distance_transform_edt(~sp, sampling=spacing)[sr]))
    return float(np.percentile(distances, 95)), 'finite'


def score_case(gt_path, prediction_path, case_id, code):
    gi, pi = nib.load(str(gt_path)), nib.load(str(prediction_path))
    if gi.shape != pi.shape or not np.allclose(gi.affine, pi.affine, rtol=0, atol=1e-4):
        raise ValueError(f'GT/prediction geometry mismatch: {case_id}')
    ref, pred = np.asanyarray(gi.dataobj), np.asanyarray(pi.dataobj)
    if not np.isin(ref, (0, 1, 2, 3)).all() or not np.isin(pred, (0, 1, 2, 3)).all():
        raise ValueError('Evaluation requires internal labels 0=BG,1=ED,2=NCR,3=ET; convert raw labels first')
    voxel = float(abs(np.linalg.det(gi.affine[:3, :3])))
    spacing = np.sqrt((gi.affine[:3, :3] ** 2).sum(0))
    result = []
    for region, labels in REGIONS.items():
        r, p = np.isin(ref, labels), np.isin(pred, labels)
        nr, npred, tp = int(r.sum()), int(p.sum()), int((r & p).sum())
        distance, status = hd95(r, p, spacing)
        result.append({'case_id': case_id, 'mask_code': code, 'n_modalities': int(code).bit_count(),
            'region': region, 'dice': 2 * tp / (nr + npred) if nr + npred else None,
            'ref_mm3': nr * voxel, 'pred_mm3': npred * voxel, 'fp_mm3': (npred - tp) * voxel,
            'fn_mm3': (nr - tp) * voxel, 'hd95_mm': distance, 'hd95_status': status,
            'NCR_to_ED_mm3': float(np.count_nonzero((ref == 2) & (pred == 1)) * voxel),
            'NCR_to_ET_mm3': float(np.count_nonzero((ref == 2) & (pred == 3)) * voxel)})
    return result


def aggregate_case_routes(routes, case_id, code):
    valid = [r for r in routes if 'weights' in r]
    if not valid:
        return []
    weights = np.stack([r['weights'][0] for r in valid])
    betas = np.stack([r['beta'][0] for r in valid])
    local = np.stack([np.asarray(r['local_logit_rms'])[:, 0] for r in valid])
    refined = np.stack([np.asarray(r['refined_logit_rms'])[:, 0] for r in valid])
    mask = mask_from_code(code).numpy()
    names = MODALITIES
    # Physical inputs still have four slots. The fifth node (independent
    # Generalist or shared Joint) is present for every nonempty input subset.
    if weights.shape[-1] == 5:
        names = (*MODALITIES, 'Generalist')
        mask = np.concatenate((mask, np.ones(1, dtype=bool)))
    elif weights.shape[-1] != 4:
        raise ValueError(f'Unsupported expert count: {weights.shape[-1]}')
    declared_names = [tuple(r['expert_names']) for r in valid if 'expert_names' in r]
    if declared_names:
        candidate = declared_names[0]
        if (len(candidate) != weights.shape[-1] or candidate[:4] != tuple(MODALITIES)
                or any(not isinstance(name, str) or not name for name in candidate)
                or len(set(candidate)) != len(candidate)
                or any(other != candidate for other in declared_names)):
            raise ValueError('Inconsistent expert names in evaluation routes')
        names = candidate
    if not np.all(weights[:, :, ~mask] == 0) or not np.allclose(weights.sum(-1), 1, atol=1e-6):
        raise ValueError('Invalid routing weights in evaluation')
    result = []
    for c, region in enumerate(('WT', 'TC', 'ET')):
        for m, name in enumerate(names):
            result.append({'case_id': case_id, 'mask_code': code, 'region': region, 'modality': name,
                'present': int(mask[m]), 'weight': float(weights[:, c, m].mean()),
                'beta': float(betas[:, c].mean()), 'joint_weight': float((1 - betas[:, c]).mean()),
                'effective_weight': float((weights[:, c, m] * betas[:, c]).mean()),
                'local_logit_rms': float(local[:, m, c].mean()), 'refined_logit_rms': float(refined[:, m, c].mean()),
                'n_window_tta_forwards': len(valid), 'scope': 'equal-window/TTA mean within one case'})
    return result


def summarize(rows, codes):
    summaries = []
    for code in codes:
        for region in REGIONS:
            selected = [r for r in rows if r['mask_code'] == code and r['region'] == region]
            values = [r['dice'] for r in selected if r['dice'] is not None]
            nonempty = [r['dice'] for r in selected if r['ref_mm3'] > 0]
            empty = [r['fp_mm3'] for r in selected if r['ref_mm3'] == 0]
            distances = [r['hd95_mm'] for r in selected if r['hd95_mm'] is not None]
            summaries.append({'mask_code': code, 'n_modalities': int(code).bit_count(), 'region': region,
                'n_cases': len(selected), 'n_valid_dice': len(values), 'dice_mean': float(np.mean(values)) if values else None,
                'dice_median': float(np.median(values)) if values else None, 'dice_p10': float(np.quantile(values, .1)) if values else None,
                'nonempty_gt_dice_mean': float(np.mean(nonempty)) if nonempty else None,
                'empty_gt_fp_mm3_mean': float(np.mean(empty)) if empty else None,
                'n_empty_gt': len(empty), 'n_dice_below_0_5': sum(v < .5 for v in values),
                'hd95_finite_mean_mm': float(np.mean(distances)) if distances else None,
                'n_hd95_finite': len(distances), 'n_hd95_one_empty': sum(r['hd95_status'] == 'one_empty' for r in selected),
                'n_hd95_both_empty': sum(r['hd95_status'] == 'both_empty' for r in selected)})
    groups = []
    for count in (1, 2, 3, 4, 'requested_all', 'missing_only'):
        for region in REGIONS:
            selected = [r for r in summaries if r['region'] == region and
                        (count == 'requested_all' or (count == 'missing_only' and r['mask_code'] != 15) or r['n_modalities'] == count)]
            values = [r['dice_mean'] for r in selected if r['dice_mean'] is not None]
            if selected:
                groups.append({'n_modalities': count, 'region': region, 'n_masks': len(selected),
                               'n_masks_valid': len(values), 'dice_mean': float(np.mean(values)) if values else None})
    return summaries, groups


def removal_deltas(rows):
    lookup = {(r['case_id'], r['mask_code'], r['region']): r for r in rows}
    result = []
    for r in rows:
        for m, name in enumerate(MODALITIES):
            if not (r['mask_code'] >> m) & 1:
                continue
            smaller = r['mask_code'] & ~(1 << m)
            other = lookup.get((r['case_id'], smaller, r['region']))
            if smaller and other is None:
                continue
            delta = r['dice'] - other['dice'] if other is not None and r['dice'] is not None and other['dice'] is not None else None
            result.append({'case_id': r['case_id'], 'mask_code': r['mask_code'], 'removed': name,
                           'smaller_mask_code': smaller, 'region': r['region'], 'delta_dice': delta,
                           'status': 'empty_input_not_evaluated' if smaller == 0 else ('undefined_dice' if delta is None else 'valid')})
    return result


def paired_comparison(current_csv, baseline_csv, output_json, *, seed=0, repetitions=2000):
    def read(path):
        with Path(path).open() as stream:
            return {(r['case_id'], int(r['mask_code']), r['region']): float(r['dice']) if r['dice'] else np.nan
                    for r in csv.DictReader(stream)}
    current, baseline = read(current_csv), read(baseline_csv)
    if current.keys() != baseline.keys():
        raise ValueError('Paired comparison requires identical cases, masks and regions')
    cases = sorted({k[0] for k in current})
    codes = [c for c in MASK_CODES if any(k[1] == c for k in current)]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cases), size=(repetitions, len(cases)))
    result = {'resampling_unit': 'case; all masks of a case are resampled together', 'seed': seed,
              'repetitions': repetitions, 'n_cases': len(cases), 'regions': {}}
    for region in REGIONS:
        delta = np.array([[current[c, m, region] - baseline[c, m, region] for m in codes] for c in cases])
        items = {}
        for j, code in enumerate(codes):
            values = delta[:, j]; valid = np.isfinite(values)
            samples = values[indices]; n = np.isfinite(samples).sum(1)
            means = np.divide(np.nansum(samples, 1), n, out=np.full(repetitions, np.nan), where=n > 0)
            boots = means[np.isfinite(means)]
            items[str(code)] = {'n_valid_pairs': int(valid.sum()), 'mean_delta': float(values[valid].mean()) if valid.any() else None,
                'ci95': np.quantile(boots, [.025, .975]).tolist() if len(boots) else None,
                'n_improved': int((values > 0).sum()), 'n_worse': int((values < 0).sum())}
        sampled = delta[indices]
        n = np.isfinite(sampled).sum(1)
        per_mask = np.divide(np.nansum(sampled, 1), n, out=np.full(n.shape, np.nan), where=n > 0)
        denom = np.isfinite(per_mask).sum(1)
        macro = np.divide(np.nansum(per_mask, 1), denom, out=np.full(repetitions, np.nan), where=denom > 0)
        finite = macro[np.isfinite(macro)]
        means = [v['mean_delta'] for v in items.values() if v['mean_delta'] is not None]
        result['regions'][region] = {'by_mask': items, 'macro_delta': float(np.mean(means)) if means else None,
                                    'macro_ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}
    write_json(output_json, result)
    return result


def evaluate_cases(predictor, keys, source, gt_folder, output, *, codes=MASK_CODES,
                   save_probabilities=False, provenance=None, protocol='controlled_preprocessed'):
    """source(case, code) returns only that case/subset's tensor and properties.

    Completed case/subset records permit safe resume only with identical
    model, code, cohort and inference protocol. Existing unrelated dirs fail.
    """
    output, gt_folder = Path(output), Path(gt_folder)
    keys, codes = list(keys), tuple(int(c) for c in codes)
    if not keys or len(set(keys)) != len(keys) or not codes or len(set(codes)) != len(codes):
        raise ValueError('Nonempty unique case IDs and masks required')
    for code in codes:
        mask_from_code(code)
    if any(not (gt_folder / (k + '.nii.gz')).is_file() for k in keys):
        raise FileNotFoundError('Missing internal-label reference segmentation')
    source_dir = Path(__file__).parent
    contract = {'version': 1, 'protocol': protocol, 'case_ids': keys, 'mask_codes': codes,
                'modalities': MODALITIES, 'regions': REGIONS, 'provenance': provenance or {},
                'network_sha256': state_digest(predictor.network), 'tile_step': predictor.tile_step_size,
                'mirroring': predictor.use_mirroring, 'mirror_axes': predictor.allowed_mirroring_axes,
                'save_probabilities': save_probabilities, 'gt_folder': str(gt_folder.resolve()),
                'gt_sha256': {k: file_sha256(gt_folder / (k + '.nii.gz')) for k in keys},
                'code_sha256': {p.name: file_sha256(p) for p in source_dir.glob('*.py')},
                'empty_both_dice': None, 'hd95_empty_policy': 'undefined; one-empty failures counted separately',
                'accumulation': 'FP32 Gaussian sliding windows; case-equal aggregate routing'}
    contract = json.loads(json.dumps(contract))
    manifest = output / 'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['contract'] != contract:
            raise ValueError('Output belongs to different inputs/model/code; choose a new directory')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to mix evaluation with an existing nonempty directory')
    write_json(manifest, {'status': 'running', 'contract': contract})
    metrics, aggregated = [], []
    for key in keys:
        for code in codes:
            done = output / 'completed' / f'mask_{code:02d}' / (key + '.json')
            prefix = output / 'predictions' / f'mask_{code:02d}' / key
            pred = Path(str(prefix) + predictor.dataset_json['file_ending'])
            if done.exists() and pred.exists():
                record = json.loads(done.read_text())
                if record['prediction_sha256'] != file_sha256(pred):
                    raise ValueError(f'Prediction changed after completion: {pred}')
            else:
                data, properties = source(key, code)
                logits, routes = predictor.predict_with_mask(data, mask_from_code(code), case_id=key)
                prefix.parent.mkdir(parents=True, exist_ok=True)
                export_prediction_from_logits(logits, properties, predictor.configuration_manager,
                    predictor.plans_manager, predictor.dataset_json, str(prefix), save_probabilities, num_threads_torch=4)
                record = {'metrics': score_case(gt_folder / (key + '.nii.gz'), pred, key, code),
                          'routing': aggregate_case_routes(routes, key, code), 'raw_routing': routes,
                          'prediction_sha256': file_sha256(pred)}
                write_json(done, record)
            metrics.extend(record['metrics']); aggregated.extend(record['routing'])
            print(f'Evaluated {key}, mask={code:02d}', flush=True)
    by_mask, by_count = summarize(metrics, codes)
    write_csv(output / 'per_case_metrics.csv', metrics)
    write_csv(output / 'summary_by_mask.csv', by_mask)
    write_csv(output / 'summary_by_cardinality.csv', by_count)
    write_csv(output / 'routing_per_case.csv', aggregated, fields=None if aggregated else ['case_id', 'mask_code', 'region', 'modality', 'weight'])
    write_csv(output / 'input_removal_delta.csv', removal_deltas(metrics))
    summary = {'n_cases': len(keys), 'n_masks': len(codes), 'by_mask': by_mask, 'by_cardinality': by_count}
    write_json(output / 'summary.json', summary)
    write_json(manifest, {'status': 'complete', 'contract': contract})
    return summary


def evaluate_preprocessed(network, plans_manager, configuration_manager, dataset_json, dataset_class,
                          data_folder, keys, gt_folder, output, *, device, mirror_axes,
                          codes=MASK_CODES, save_probabilities=False, provenance=None):
    predictor = MissingModalityPredictor(device=device, tile_step_size=.5, use_gaussian=True,
        use_mirroring=True, perform_everything_on_device=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans_manager, configuration_manager, None, dataset_json,
                                    type(network).__name__, mirror_axes)
    dataset = dataset_class(str(data_folder), list(keys))
    cached = {}
    def source(key, code):
        if cached.get('key') != key:
            data, _, _, props = dataset.load_case(key)
            cached.update(key=key, data=torch.from_numpy(np.asarray(data[:], dtype=np.float32)), properties=props)
        return cached['data'], cached['properties']
    provenance = dict(provenance or {}, data_folder=str(Path(data_folder).resolve()),
                      input_sha256=preprocessed_input_fingerprints(data_folder, keys))
    return evaluate_cases(predictor, keys, source, gt_folder, output, codes=codes,
                          save_probabilities=save_probabilities, provenance=provenance)


def evaluate_available(predictor, cases, gt_folder, output, *, codes=MASK_CODES, save_probabilities=False, provenance=None):
    lookup = {c['case_id']: c for c in cases}
    if len(lookup) != len(cases):
        raise ValueError('Duplicate case IDs')
    for case in cases:
        for code in codes:
            if any(((code >> m) & 1) and name not in case['modalities'] for m, name in enumerate(MODALITIES)):
                raise ValueError(f'{case["case_id"]} cannot supply mask {code}; all15 evaluation needs four source modalities')
    def source(key, code):
        selected = {name: path for name, path in lookup[key]['modalities'].items() if (code >> MODALITIES.index(name)) & 1}
        data, props, _ = preprocess_available_case(selected, predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
        return data, props
    inputs = {c['case_id']: {m: {'path': p, 'sha256': file_sha256(p)} for m, p in c['modalities'].items()} for c in cases}
    return evaluate_cases(predictor, list(lookup), source, gt_folder, output, codes=codes,
        save_probabilities=save_probabilities, provenance=dict(provenance or {}, inputs=inputs), protocol='available_files')
