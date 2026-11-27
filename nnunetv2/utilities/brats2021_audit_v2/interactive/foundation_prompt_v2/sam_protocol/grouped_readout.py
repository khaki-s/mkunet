"""Learn the value of local signed edits from paired SAM3 and 192³ responses."""
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from ..common import check_file, file_identity, manifest, read_json, read_labels, save_mask, same_grid, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import dense, residual, scoring
from .cohorts import rows

SCHEMA = 'sam_wt_local_signed_groups_v1'
DEFAULT_GROUP_MM = 8.
_FIXED_WIDTH_SHA = '7fdb6dfdce6fb1f39f0221eecc9473cd527644d3c58193a0883395d0fb11fcc8'
LEVELS = (.50, .55, .60, .65, .70, .75, .80, .85, .90, np.inf)


def _matches_method(record, group_mm, code):
    return record.get('group_mm', DEFAULT_GROUP_MM) == group_mm and (
        record['code'] == code or (group_mm == DEFAULT_GROUP_MM and record['code']['sha256'] == _FIXED_WIDTH_SHA))


def _fields(mri, labels, raw, response, scores, affine, group_mm):
    coords, x = residual.observations(mri, labels, raw, response, scores, affine, 0)
    allowed = labels.ravel()[coords] <= 1
    coords, x = coords[allowed], x[allowed]
    if not len(coords):
        empty = np.empty(0, dtype=np.int64)
        return coords, empty, np.empty((0, 0), np.float32), empty, np.empty(0, bool)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    width = np.floor((np.asarray(labels.shape) - 1) * spacing / group_mm).astype(int) + 1
    ijk = np.unravel_index(coords, labels.shape)
    cell = tuple(np.floor(a * s / group_mm).astype(int) for a, s in zip(ijk, spacing))
    prior = labels.ravel()[coords] > 0
    signature = (raw.ravel()[coords] != prior).astype(np.int64)
    for j, source in enumerate(response):
        signature |= (source.ravel()[coords] != prior).astype(np.int64) << (j + 1)
    address = (np.ravel_multi_index(cell, width) * 16 + signature) * 2 + prior
    keys, group = np.unique(address, return_inverse=True)
    size = np.bincount(group)
    features = np.empty((len(size), x.shape[1] + 1), np.float32)
    for col in range(x.shape[1]):
        features[:, col] = np.bincount(group, weights=x[:, col]) / size
    features[:, -1] = np.log1p(size)
    return coords, group, features, size, (keys % 2).astype(bool)


def group_sample(args):
    training = rows(args.cohort, 'train', truth=True)[args.shard_index::args.num_shards]
    site = Path(args.output)
    site.mkdir(parents=True, exist_ok=True)
    source_code = file_identity(__file__)
    for patient in training:
        key = patient['case_id']
        row = {k: v for k, v in patient.items() if k != 'gt'}
        mri, labels, ref, sam, scores, response, s, d = residual._evidence(row, args.dense, args.responses)
        json_path = site / f'{key}.json'
        binding = {'row': patient, 'sam': s['arrays'], 'response': d['arrays']}
        if json_path.is_file():
            receipt = read_json(json_path)
            saved = receipt['binding']
            if (any(saved[k] != v for k, v in binding.items())
                    or not _matches_method(saved, args.group_mm, source_code)):
                raise ValueError(f'{key}: mismatched grouped training sample')
            check_file(receipt['arrays'])
            continue
        gt = residual._truth(patient, ref) > 0
        original = labels > 0
        arrays = {'baseline': np.array([np.count_nonzero(original & gt), int(original.sum() + gt.sum())])}
        for v in range(len(dense.VARIANTS)):
            coords, index, x, count, old = _fields(mri, labels, sam[v, 0], response[v, 0], scores[v, 0],
                                                  ref.affine, args.group_mm)
            hits = np.bincount(index, weights=gt.ravel()[coords], minlength=len(count))
            arrays.update({f'x{v}': x, f'n{v}': count, f'tp{v}': hits, f'old{v}': old})
        data_path = site / f'{key}.npz'
        dense._atomic_npz(data_path, **arrays)
        write_json(json_path, {'binding': {**binding, 'group_mm': args.group_mm, 'code': source_code},
                               'arrays': file_identity(data_path)})
        print(f'Grouped WT training data {key}', flush=True)
    return site


def _fit(cases, variant, threads):
    output = []
    for old in (False, True):
        features, targets, weights = [], [], []
        for record in cases:
            take = record[f'old{variant}'] == old
            if not take.any():
                continue
            size = record[f'n{variant}'][take]
            denom = max(1, int(record['baseline'][1]))
            features.append(record[f'x{variant}'][take])
            targets.append(record[f'tp{variant}'][take] / size)
            weights.append(size / denom)
        if not features:
            output.append(.5)
            continue
        x, y, weight = map(np.concatenate, (features, targets, weights))
        model = HistGradientBoostingRegressor(max_iter=110, max_leaf_nodes=23, min_samples_leaf=40,
                                              learning_rate=.08, l2_regularization=3., random_state=781)
        with threadpool_limits(limits=threads):
            model.fit(x, y, sample_weight=weight / weight.mean())
        output.append(model)
    return output


def _predict_risk(models, features, prior, threads):
    risk = np.empty(len(prior), np.float32)
    with threadpool_limits(limits=threads):
        for i, model in enumerate(models):
            where = prior == bool(i)
            if not where.any():
                continue
            score = np.full(where.sum(), model) if isinstance(model, float) else model.predict(features[where])
            risk[where] = np.clip(1. - score if i else score, 0., 1.)
    return risk


def _calibrate(models, cases, variant, threads):
    scores = []
    for record in cases:
        prior, size, hits = (record[f'{p}{variant}'] for p in ('old', 'n', 'tp'))
        risk = _predict_risk(models, record[f'x{variant}'], prior, threads)
        tp0, denominator = record['baseline']
        if denominator == 0:
            continue
        old = 200. * tp0 / denominator
        growth = (risk[None, :] >= np.array(LEVELS)[:, None]) & ~prior
        shrink = (risk[None, :] >= np.array(LEVELS)[:, None]) & prior
        tp = tp0 + (growth @ hits)[:, None] - (shrink @ hits)[None, :]
        denom = denominator + (growth @ size)[:, None] - (shrink @ size)[None, :]
        scores.append(200. * tp / np.maximum(1, denom) - old)
    changes = np.stack(scores)
    benefit = changes.mean(0) - .25 * np.maximum(0, -changes).mean(0)
    i, j = np.unravel_index(np.argmax(benefit), benefit.shape) if benefit.max() > 0 else (-1, -1)
    return [LEVELS[i], LEVELS[j]], float(changes[:, i, j].mean())


def group_fit(args):
    source = Path(args.samples)
    cohort_id = file_identity(args.cohort)
    code = file_identity(__file__)
    cases = []
    training = rows(args.cohort, 'train', truth=True)
    for row in training:
        receipt = read_json(source / f'{row["case_id"]}.json')
        if receipt['binding']['row'] != row or not _matches_method(receipt['binding'], args.group_mm, code):
            raise ValueError('Grouped samples do not match the frozen training split')
        check_file(receipt['arrays'])
        with np.load(receipt['arrays']['path'], allow_pickle=False) as data:
            cases.append({k: data[k] for k in data.files})
    fit = [case for i, case in enumerate(cases) if i % 5 != 4]
    calibration = cases[4::5]
    plans = []
    for v, name in enumerate(dense.VARIANTS):
        models = _fit(fit, v, args.threads)
        levels, gain = _calibrate(models, calibration, v, args.threads)
        plans.append({'models': _fit(cases, v, args.threads), 'levels': levels, 'train_calibration_gain_pp': gain})
        print(name, levels, gain, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_atomic({'schema': SCHEMA, 'cohort': cohort_id, 'plans': plans, 'group_mm': args.group_mm, 'code': code}, output)
    return output


def _model(args):
    model = joblib.load(args.model)
    if (model['schema'] != SCHEMA or model['cohort'] != file_identity(args.cohort)
            or not _matches_method(model, model.get('group_mm', DEFAULT_GROUP_MM), file_identity(__file__))):
        raise ValueError('Wrong grouped model implementation or split')
    return model


def group_predict(args):
    model = _model(args)
    group_mm = model.get('group_mm', DEFAULT_GROUP_MM)
    for row in rows(args.cohort, 'validation')[args.shard_index::args.num_shards]:
        folder = Path(args.output) / row['case_id']
        mri, labels, ref, sam, scores, response, s, d = residual._evidence(row, args.dense, args.responses)
        results = {}
        for i, variant in enumerate(dense.VARIANTS):
            indices, group, x, _, prior = _fields(mri, labels, sam[i, 0], response[i, 0], scores[i, 0],
                                                  ref.affine, group_mm)
            action = model['plans'][i]
            confidence = _predict_risk(action['models'], x, prior, args.threads)
            accept = confidence >= np.where(prior, action['levels'][1], action['levels'][0])
            wt = (labels > 0).ravel()
            changed = indices[accept[group]]
            wt[changed] = ~wt[changed]
            # A WT-only pilot preserves the original TC and ET segmentation.
            updated = np.where(labels > 1, labels, wt.reshape(labels.shape)).astype(np.uint8)
            results[variant] = save_mask(folder / f'{variant}.nii.gz', updated, ref)
        write_json(folder/'prediction.json', {'row': row, 'model': file_identity(args.model), 'gt_used': False,
                   'sam': s['arrays'], 'response': d['arrays'], 'outputs': results})
        print(row['case_id'], flush=True)
    return args.output


def group_score(args):
    model = _model(args)
    expected = rows(args.cohort, 'validation')
    truth = {row['case_id']: row for row in manifest(args.truth, allowed=('validation',))['cases']}
    records = {name: [] for name in dense.VARIANTS}
    for row in expected:
        key = row['case_id']
        receipt = read_json(Path(args.predictions) / key / 'prediction.json')
        if receipt['row'] != row or receipt['model'] != file_identity(args.model) or receipt['gt_used']:
            raise ValueError('Wrong grouped prediction receipt')
        before, gt, ref, volume = scoring._image_pair(row, truth[key])
        for name, item in receipt['outputs'].items():
            check_file(item)
            prediction = nib.load(item['path'])
            same_grid(prediction, ref)
            records[name].extend({'case_id': key, **record} for record in scoring._measure(before, read_labels(prediction), gt, volume))
    index = max(range(len(dense.VARIANTS)), key=lambda i: model['plans'][i]['train_calibration_gain_pp'])
    report = {'schema': SCHEMA, 'n_cases': len(expected), 'chosen_on_train': dense.VARIANTS[index],
              'group_mm': model.get('group_mm', DEFAULT_GROUP_MM),
              'by_variant': {name: scoring.summary(values) for name, values in records.items()}, 'per_case': records}
    write_json(args.output, report)
    return args.output
