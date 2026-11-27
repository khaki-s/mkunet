"""Learn which actual nnInteractive changes to keep on the native voxel grid."""
from collections import defaultdict
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ..common import check_file, file_identity, read_json, read_labels, same_grid, save_mask, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import candidates, dense
from .cohorts import read_cohort, rows

THRESHOLDS = (.35, .45, .50, .55, .62, .70, .80, .9)
MODEL_SCHEMA = 'sam3_paired_interactive_residual_v1'
DICE_TARGET_PRIMARY = 'WT'


def _window(mask, padding=7):
    return tuple(slice(max(0, points.min() - padding), min(n, points.max() + padding + 1))
                 for points, n in zip(np.nonzero(mask), mask.shape))


def _truth(row, reference):
    check_file(row['gt'])
    image = nib.load(row['gt']['path'])
    same_grid(image, reference)
    return read_labels(image)


def _image_features(mri, original, window, coords, sampling):
    columns = []
    for image in mri:
        body = image[::2, ::2, ::2]
        body = body[body != 0]
        if not len(body):
            # Absent modalities have an explicit indicator. No zero-channel
            # input is routed into nnInteractive.
            columns.append(np.zeros((len(coords[0]), 6), dtype=np.float32))
            continue
        lower, center, upper = np.percentile(body, (10, 50, 90))
        scale = max(1e-6, float(upper - lower))
        native = np.clip((image[window] - center) / scale, -5, 5)
        foreground = original & (image[window] != 0)
        tumor = np.median(native[foreground]) if foreground.any() else 0.
        fine = ndimage.gaussian_filter(native, sigma=.9 / sampling)
        coarse = ndimage.gaussian_filter(native, sigma=2.8 / sampling)
        columns.append(np.column_stack((native[coords], fine[coords], coarse[coords],
            (native - fine)[coords], native[coords] - tumor, np.ones(len(coords[0])))).astype(np.float32))
    # Enhancement is only meaningful with both T1 and T1ce available.
    contrast = (columns[1][:, 0] - columns[0][:, 0]) * columns[0][:, -1] * columns[1][:, -1]
    return np.column_stack((*columns, contrast))


def observations(mri, baseline, sam_mask, responses, scores, affine, h):
    """Return index and native-neighborhood features where a real response differs."""
    base = baseline > h
    any_change = np.any(responses != base, axis=0)
    any_change &= np.any(mri != 0, axis=0)
    if not any_change.any():
        return np.empty(0, dtype=np.int32), np.empty((0, 0), dtype=np.float32)
    sampling = np.linalg.norm(affine[:3, :3], axis=0)
    window = _window(any_change)
    coords = np.nonzero(any_change[window])
    center = np.array([s.start for s in window])
    indices = tuple(c + offset for c, offset in zip(coords, center))
    original = base[window]
    sam = sam_mask[window]
    interactive = responses[(slice(None), *window)]
    norm = np.log1p(np.count_nonzero(base))
    dim = original.shape
    dist = (ndimage.distance_transform_edt(np.pad(original, 1), sampling=sampling) -
            ndimage.distance_transform_edt(np.pad(~original, 1), sampling=sampling))[1:-1, 1:-1, 1:-1]
    difference = ndimage.distance_transform_edt(np.pad((sam != original), 1), sampling=sampling)[1:-1, 1:-1, 1:-1]
    f = [original[coords], sam[coords], baseline[indices], np.clip(dist[coords], -30, 30),
         difference[coords], scores[indices[2]], np.full(len(coords[0]), norm),
         *(c / max(1, n-1) for c, n in zip(coords, dim)),
         np.full(len(coords[0]), np.log1p(np.count_nonzero(sam_mask))),
         *(np.full(len(coords[0]), np.log1p(np.count_nonzero(mask != base))) for mask in responses)]
    for mask in (original, sam, *interactive):
        f.extend((mask[coords], ndimage.uniform_filter(mask.astype(np.float32), size=5)[coords],
                  ndimage.uniform_filter(mask.astype(np.float32), size=11)[coords]))
    x = np.column_stack((*f, _image_features(mri, original, window, coords, sampling))).astype(np.float32)
    return np.ravel_multi_index(indices, baseline.shape).astype(np.int32), x


def _evidence(row, sam_dir, responses_dir):
    mri, baseline, ref = dense.load_native(row)
    sam, confidence, s = dense.read_masks(sam_dir, row)
    results, d = candidates.read_responses(responses_dir, row)
    if d['sam'] != s['arrays'] or sam.shape[-3:] != baseline.shape or results.shape[-3:] != baseline.shape:
        raise ValueError('Candidate evidence is not aligned')
    return mri, baseline, ref, sam, confidence, results, s, d


LEGACY_BASIC_FEATURE_CODE = '98ad6f21e42854a3a1d4805fe14609b298dafc7019022386623c5b2900e8c3c0'
LEGACY_CONTINUOUS_FEATURE_CODE = 'db44d1ad75422f70a801d3058965cd225527c961526a4cc6347d10b819940606'


def _precomputed_features(root, row, sam, native):
    if root is None:
        return None, None
    from .expert_features import read_features

    return read_features(root, row, sam, native)


def _case_observations(data, mri, baseline, masks, responses, scores, affine, v, h):
    if data is not None:
        return data[f'indices_{v}_{h}'], data[f'x_{v}_{h}']
    return observations(mri, baseline, masks[v, h], responses[v, h], scores[v, h], affine, h)


def _coverage(base, truth, possible):
    add_possible = int(np.count_nonzero(possible & ~base & truth))
    remove_possible = int(np.count_nonzero(possible & base & ~truth))
    old = 2 * np.count_nonzero(base & truth)
    denom = np.count_nonzero(base) + np.count_nonzero(truth)
    oracle = (old + 2 * add_possible) / max(1, denom + add_possible - remove_possible)
    initial_dice = old / denom if denom else 1.
    return {'baseline_pct': 100 * initial_dice, 'fn': int(np.count_nonzero(~base & truth)),
            'fp': int(np.count_nonzero(base & ~truth)), 'covered_fn': add_possible, 'covered_fp': remove_possible,
            'oracle_gain_pp': 100 * ((oracle if denom else 1.) - initial_dice)}


def sample(args):
    selected = rows(args.cohort, 'train', truth=True)[args.shard_index::args.num_shards]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        key = row['case_id']
        clean = {k:v for k,v in row.items() if k != 'gt'}
        mri, baseline, ref, masks, iou, responses, s, d = _evidence(clean, args.dense, args.responses)
        features, feature_identity = _precomputed_features(getattr(args, 'features', None), clean, s, d)
        folder = root / key
        receipt_path = folder / 'sample.json'
        binding = {'row': row, 'sam': s['arrays'], 'responses': d['arrays'], 'code_sha256': file_identity(__file__)['sha256']}
        if feature_identity is not None:
            binding['features'] = feature_identity
            binding['feature_schema'] = features['schema']
        if receipt_path.exists():
            existing = read_json(receipt_path)
            equivalent = (feature_identity is None
                          and existing['binding'] == {**binding, 'code_sha256': LEGACY_BASIC_FEATURE_CODE})
            previous = {k: v for k, v in binding.items() if k != 'feature_schema'}
            previous['code_sha256'] = LEGACY_CONTINUOUS_FEATURE_CODE
            same_continuous_features = (binding.get('feature_schema') == 'paired_expert_continuous_v1'
                                        and existing['binding'] == previous)
            if existing['binding'] != binding and not equivalent and not same_continuous_features:
                raise ValueError(f'{key}: sample already exists with another configuration')
            check_file(existing['arrays'])
            continue
        truth = _truth(row, ref)
        arrays, coverage = {}, {}
        for v, source in enumerate(dense.VARIANTS):
            for h, head in enumerate(dense.HEADS):
                coords, x = _case_observations(features, mri, baseline, masks, responses, iou, ref.affine, v, h)
                old, target = baseline > h, truth > h
                possible = np.any(responses[v,h] != old, axis=0) & np.any(mri != 0, axis=0)
                summary = _coverage(old, target, possible)
                summary['sam_oracle_gain_pp'] = _coverage(old, target, masks[v,h] != old)['oracle_gain_pp']
                coverage[f'{source}/{head}'] = summary
                rng = np.random.default_rng(5151 + h)
                keep = rng.choice(len(coords), min(7000, len(coords)), replace=False)
                arrays[f'x_{v}_{h}'] = x[keep]
                arrays[f'y_{v}_{h}'] = target.ravel()[coords[keep]]
                arrays[f'weight_{v}_{h}'] = np.array(2.00 - summary['baseline_pct'] / 100)
        folder.mkdir(exist_ok=True)
        arrays_path = folder / 'samples.npz'
        dense._atomic_npz(arrays_path, **arrays)
        write_json(receipt_path, {'binding': binding, 'coverage': coverage, 'arrays': file_identity(arrays_path)})
        print(f'Train samples {key}', flush=True)
    return root


def _sample_ids(root, selected):
    result = {}
    for row in selected:
        key = row['case_id']
        receipt = read_json(Path(root) / key / 'sample.json')
        if receipt['binding']['row'] != row:
            raise ValueError('Training samples differ from frozen split')
        check_file(receipt['arrays'])
        result[key] = receipt
    return result


def _fit(samples, keys, variant, head, nthreads):
    arrays, labels, weights = [], [], []
    for key in keys:
        receipt = samples[key]['arrays']
        with np.load(receipt['path'], allow_pickle=False) as f:
            x, y, care = (f[f'{kind}_{variant}_{head}'] for kind in ('x','y','weight'))
        if not len(y):
            continue
        arrays.append(x)
        labels.append(y)
        weights.append(np.full(len(y), float(care) / np.sqrt(len(y))))
    if not labels:
        return .5
    x = np.concatenate(arrays)
    y = np.concatenate(labels)
    w = np.concatenate(weights)
    if np.unique(y).size == 1:
        return float(y[0])
    tree = HistGradientBoostingClassifier(max_iter=125, learning_rate=.09, max_leaf_nodes=23,
                                           min_samples_leaf=60, l2_regularization=3.5, random_state=773)
    with threadpool_limits(limits=nthreads):
        tree.fit(x, y, sample_weight=w/w.mean())
    return tree


def posterior(model, x, nthreads=3):
    if not len(x):
        return np.empty(0, dtype=np.float32)
    if isinstance(model, float):
        return np.full(len(x), model, dtype=np.float32)
    with threadpool_limits(limits=nthreads):
        return model.predict_proba(x)[:, 1].astype(np.float32)


NATIVE_POLICY = 'native_point_box'


def combine_native_point_box(initial, additions, removals):
    """Apply each response only on the baseline labels it can change."""
    return np.where(initial, removals, additions)


def _native_point_values(replies, initial, indices):
    reference = initial.ravel()[indices]
    added = replies[1].ravel()[indices]
    removed = replies[2].ravel()[indices]
    return combine_native_point_box(reference, added, removed)


def _changed(prob, selected, old, add, remove, native=None):
    b = old.ravel()[selected]
    if add == NATIVE_POLICY or remove == NATIVE_POLICY:
        if native is None or native.shape != b.shape:
            raise ValueError('The selected policy requires native point and box predictions')
    add_threshold = np.inf if add is None else add
    remove_threshold = -np.inf if remove is None else remove
    additions = native if add == NATIVE_POLICY else prob >= add_threshold
    removals = ~native if remove == NATIVE_POLICY else prob <= remove_threshold
    return ~b & additions, b & removals


def _gains(prob, indices, before, truth, native=None):
    """Exact train-calibration Dice for all paired add/remove thresholds."""
    y = truth.ravel()[indices]
    b = before.ravel()[indices]
    n = int(before.sum() + truth.sum())
    tp = np.count_nonzero(before & truth)
    base = 2.*tp/n if n else 1.
    a = []
    r = []
    options = (*THRESHOLDS, NATIVE_POLICY, None) if native is not None else (*THRESHOLDS, None)
    for threshold in options:
        added, removed = _changed(prob, indices, before, threshold, threshold, native)
        a.append((int(added.sum()), int(y[added].sum())))
        r.append((int(removed.sum()), int(y[removed].sum())))
    return np.array([[100 * ((2. * (tp + c - d) / (n + u - v) if n + u - v else 1.) - base)
                      for v, d in r] for u, c in a])


def fit(args):
    # A separate job for each region allows parallel training without changing
    # data placement, full-volume geometry, or the fold used for selection.
    selected = rows(args.cohort, 'train', truth=True)
    keys = [r['case_id'] for r in selected]
    sources = _sample_ids(args.samples, selected)
    if any('features' in r['binding'] for r in sources.values()):
        raise ValueError('Continuous expert feature samples require fit-gpu --features')
    heldout = [r for i,r in enumerate(selected) if i % 5 == 4]
    folded = [r for i,r in enumerate(selected) if i % 5 != 4]
    fit_keys = [row['case_id'] for row in folded]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    h = args.head
    path = root / f'{dense.HEADS[h]}.joblib'
    binding = {'schema': MODEL_SCHEMA, 'cohort': file_identity(args.cohort),
               'train_ids': keys, 'calibration_ids': [r['case_id'] for r in heldout], 'head': h,
               'samples': {key:r['arrays'] for key,r in sources.items()}, 'code_sha256': file_identity(__file__)['sha256']}
    if path.exists():
        if joblib.load(path)['binding'] != binding:
            raise ValueError('Existing trained model belongs to another experiment')
        return path
    temporary = [_fit(sources, fit_keys, v, h, args.threads) for v in range(2)]
    gains = [[], []]
    for row in heldout:
        clean = {k:v for k,v in row.items() if k != 'gt'}
        mri, before, ref, sam, iou, responses, _, _ = _evidence(clean, args.dense, args.responses)
        y = _truth(row, ref) > h
        for v in range(2):
            idx, x = observations(mri, before, sam[v,h], responses[v,h], iou[v,h], ref.affine, h)
            gains[v].append(_gains(posterior(temporary[v], x, args.threads), idx, before > h, y))
        print(f'Calibration {dense.HEADS[h]} {row["case_id"]}', flush=True)
    options = (*THRESHOLDS, None)
    final = []
    for v, records in enumerate(gains):
        g = np.stack(records)
        # Keep an explicit STOP arm and penalize losses across patients.
        objective = g.mean(axis=0) - .2 * np.maximum(0., -g).mean(axis=0)
        if objective.max() <= 0:
            i, j = len(options)-1, len(options)-1
        else:
            i, j = np.unravel_index(np.argmax(objective), objective.shape)
        policy = {'add': options[i], 'remove': options[j], 'calibration_gain_pp': float(g[:,i,j].mean()),
                  'calibration_per_case_pp': g[:,i,j].tolist(), 'train_objective': float(objective[i,j])}
        final.append({**policy, 'model': _fit(sources, keys, v, h, args.threads)})
        print(f'Fit {dense.HEADS[h]} {dense.VARIANTS[v]} {policy}', flush=True)
    dump_atomic({'binding': binding, 'variants': final}, path)
    write_json(path.with_suffix('.json'), {'binding': binding, 'variants': [{k: v for k, v in r.items() if k!='model'} for r in final]})
    return path


def load_models(path, cohort=None):
    paths = [Path(path) / f'{h}.joblib' for h in dense.HEADS]
    models = [joblib.load(p) for p in paths]
    cohort_id = file_identity(cohort) if cohort is not None else models[0]['binding']['cohort']
    code = file_identity(__file__)['sha256']
    schemas = {model['binding'].get('feature_schema') for model in models}
    if len(schemas) != 1:
        raise ValueError('Residual heads use different feature schemas')
    known = {None: LEGACY_BASIC_FEATURE_CODE,
             'paired_expert_continuous_v1': LEGACY_CONTINUOUS_FEATURE_CODE}
    for h, model in enumerate(models):
        binding = model['binding']
        legacy = known.get(binding.get('feature_schema'))
        uses_native_policy = any(NATIVE_POLICY in (v['add'], v['remove']) for v in model['variants'])
        same_code = (binding['code_sha256'] == code
                     or (legacy is not None and binding['code_sha256'] == legacy and not uses_native_policy))
        if (binding['schema'] != MODEL_SCHEMA or binding['head'] != h or binding['cohort'] != cohort_id
                or not same_code):
            raise ValueError('Residual model binding mismatch')
    selection = max(range(2), key=lambda i: (models[0]['variants'][i]['calibration_gain_pp'],
                                           np.mean([m['variants'][i]['calibration_gain_pp'] for m in models[1:]])))
    return models, selection, [file_identity(p) for p in paths]


def merge_heads(results):
    """Restore BraTS nesting, with the outer WT decision taking priority."""
    wt, tc, et = results
    tc = wt & tc
    et = tc & et
    output = np.zeros(wt.shape, np.uint8)
    output[wt] = 1
    output[tc] = 2
    output[et] = 3
    return output


def infer(selected, model_root, cohort, sam_dir, response_dir, output, *, compare=False, features_root=None):
    models, selected_variant, ids = load_models(model_root, cohort)
    feature_schema = models[0]['binding'].get('feature_schema')
    if bool(feature_schema) != bool(features_root):
        raise ValueError('Prediction must use the same feature inputs as the fitted model')
    variants = range(2) if compare else (selected_variant,)
    root = Path(output)
    for row in selected:
        key = row['case_id']
        if 'gt' in row:
            raise ValueError('Prediction requires a label-free patient record')
        folder = root / key
        path = folder / 'prediction.json'
        mri, before, reference, masks, iou, responses, s, d = _evidence(row, sam_dir, response_dir)
        features, feature_identity = _precomputed_features(features_root, row, s, d)
        if features is not None and features['schema'] != feature_schema:
            raise ValueError('Prediction features use a different schema than the trained readout')
        binding = {'row': row, 'models': ids, 'sam': s['arrays'], 'responses': d['arrays'],
                   'selected': dense.VARIANTS[selected_variant], 'variants': [dense.VARIANTS[v] for v in variants]}
        if feature_identity is not None:
            binding['features'] = feature_identity
        if path.exists():
            prior = read_json(path)
            if prior['binding'] != binding:
                raise ValueError(f'{key}: output belongs to another trained pipeline')
            for record in prior['result'].values():
                check_file(record['mask'])
            continue
        folder.mkdir(parents=True, exist_ok=True)
        final = {}
        for v in variants:
            results = []
            edits = []
            for h, model in enumerate(models):
                fitted = model['variants'][v]
                initial = before > h
                coords, x = _case_observations(features, mri, before, masks, responses, iou, reference.affine, v, h)
                p = posterior(fitted['model'], x)
                native = (_native_point_values(responses[v, h], initial, coords)
                          if NATIVE_POLICY in (fitted['add'], fitted['remove']) else None)
                add, remove = _changed(p, coords, initial, fitted['add'], fitted['remove'], native)
                updated = np.ascontiguousarray(initial).copy()
                updated.ravel()[coords[add]] = True
                updated.ravel()[coords[remove]] = False
                results.append(updated)
                edits.append({'head': dense.HEADS[h], 'available': len(coords), 'added': int(add.sum()),
                              'removed': int(remove.sum()), 'stopped': not (add.any() or remove.any())})
            output_mask = merge_heads(results)
            variant = dense.VARIANTS[v]
            final[variant] = {'mask': save_mask(folder / f'{variant}.nii.gz', output_mask, reference), 'edits': edits}
        write_json(path, {'binding': binding, 'result': final, 'gt_used': False})
        print(f'Prediction {key} mask={row["mask_code"]} variant={dense.VARIANTS[selected_variant]}', flush=True)
    return root


def predict(args):
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    return infer(selected, args.model, args.cohort, args.dense, args.responses, args.output, compare=True,
                 features_root=getattr(args, 'features', None))
