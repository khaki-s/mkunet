"""Fit and score signed native edits from matched SAM3 and expert probabilities."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ..common import check_file, file_identity, manifest, read_json, read_labels, same_grid, save_mask, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import dense, probability_features as features, residual, scoring
from .cohorts import rows

CUTS = np.r_[np.linspace(.35, .95, 25), 1.01]


def _sample(args, row, split):
    slot = Path(args.features) / split / row['case_id'] / 'features.json'
    receipt = read_json(slot)
    if (receipt['binding']['row'] != row or receipt['binding']['code'] != file_identity(features.__file__)
            or receipt['labeled'] != (split == 'train')):
        raise ValueError('Feature cache belongs to another patient or experiment')
    check_file(receipt['arrays'])
    return receipt


def load_batch(args, selected, split, variant, head):
    tag = f'{variant}_{head}'
    output = []
    for row in selected:
        rec = _sample(args, row, split)
        with np.load(rec['arrays']['path'], allow_pickle=False) as source:
            item = {name: source[f'{name}_{tag}'] for name in ('index', 'x', 'old')}
            if split == 'train':
                item['true'] = source[f'gt_{tag}']
                item['base'] = source['baseline'][head]
        output.append(item)
    return output


def _fit(cases, selected, max_samples, seed):
    rng = np.random.default_rng(seed)
    models = []
    for side in (False, True):
        vectors, labels, weights = [], [], []
        for i in selected:
            case = cases[i]
            indices = np.flatnonzero(case['old'] == side)
            if not len(indices):
                continue
            count = min(len(indices), max_samples)
            chosen = rng.choice(indices, count, replace=False)
            vectors.append(case['x'][chosen])
            labels.append(case['true'][chosen] != side)
            weights.append(np.full(count, len(indices) / (count * max(1, case['base'][1]))))
        if not vectors:
            models.append(0.)
            continue
        x, y, weight = (np.concatenate(value) for value in (vectors, labels, weights))
        finite = np.isfinite(x)
        varies = np.min(np.where(finite, x, np.inf), axis=0) < np.max(np.where(finite, x, -np.inf), axis=0)
        if not y.any() or y.all() or not varies.any():
            models.append(float(np.average(y, weights=weight)))
            continue
        clf = HistGradientBoostingClassifier(
            max_iter=72, max_leaf_nodes=18, min_samples_leaf=70, max_bins=127,
            learning_rate=.085, l2_regularization=2.5, early_stopping=False, random_state=seed)
        clf.fit(x[:, varies], y, sample_weight=weight / weight.mean())
        models.append((clf, varies))
    return models


def probability(models, case):
    result = np.empty(len(case['old']), np.float32)
    for side, model in enumerate(models):
        taken = case['old'] == bool(side)
        if taken.any():
            if isinstance(model, float):
                result[taken] = model
            else:
                estimator, varies = model
                result[taken] = estimator.predict_proba(case['x'][taken][:, varies])[:, 1]
    return result


def _outcomes(case, probabilities):
    tp, denom = case['base']
    old = case['old']
    original = tp / denom if denom else .5
    predictions, hits = [], []
    for side in (False, True):
        kept = old == side
        levels = np.searchsorted(CUTS, probabilities[kept], side='right')
        sizes = np.bincount(levels, minlength=len(CUTS) + 1)
        targets = np.bincount(levels, weights=case['true'][kept], minlength=len(CUTS) + 1)
        predictions.append(sizes[:0:-1].cumsum()[::-1])
        hits.append(targets[:0:-1].cumsum()[::-1])
    total = denom + predictions[0][:, None] - predictions[1][None, :]
    numerator = tp + hits[0][:, None] - hits[1][None, :]
    after = np.full_like(total, .5, dtype=np.float64)
    np.divide(numerator, total, out=after, where=total > 0)
    return 200. * (after - original)


def _case_folds(args, rows_):
    split = [[], []]
    for i, row in enumerate(rows_):
        current = _sample(args, row, 'train')
        split[int('extra_responses' in current['binding']['candidate_receipts'])].append(i)
    rng = np.random.default_rng(7153)
    result = np.empty(len(rows_), dtype=int)
    for items in split:
        indices = rng.permutation(items)
        result[indices] = np.arange(len(items)) % 5
    return result


def _fit_plan(pair, *, args, selected, folds):
    variant, head = pair
    cases = load_batch(args, selected, 'train', variant, head)
    gains = []
    with threadpool_limits(limits=args.threads), joblib.parallel_backend('threading', n_jobs=args.threads):
        for fold in range(5):
            models = _fit(cases, np.flatnonzero(folds != fold), args.samples_per_case, 129 + head)
            for j in np.flatnonzero(folds == fold):
                gains.append(_outcomes(cases[j], probability(models, cases[j])))
        full = _fit(cases, np.arange(len(cases)), args.samples_per_case, 129 + head)
    gain = np.stack(gains)
    expected = gain.mean(axis=0)
    utility = expected - .25 * np.maximum(-gain, 0.).mean(0)
    i, j = np.unravel_index(utility.argmax(), utility.shape)
    if utility[i, j] <= 0.:
        i = j = len(CUTS) - 1
    record = {'thresholds': [float(CUTS[i]), float(CUTS[j])], 'gain_pp': float(expected[i, j]),
              'models': full}
    print(dense.VARIANTS[variant], dense.HEADS[head], record['thresholds'], record['gain_pp'], flush=True)
    return record


def fit(args):
    selected = rows(args.cohort, 'train', truth=True)
    folds = _case_folds(args, selected)
    heads = len(dense.HEADS)
    keys = [(v, h) for v in range(len(dense.VARIANTS)) for h in range(heads)]
    task = partial(_fit_plan, args=args, selected=selected, folds=folds)
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        fitted = list(pool.map(task, keys))
    plans = [fitted[offset:offset + heads] for offset in range(0, len(fitted), heads)]
    choice = max(range(len(dense.VARIANTS)), key=lambda v: plans[v][0]['gain_pp'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = {'cohort': file_identity(args.cohort), 'source': file_identity(__file__),
                 'feature_code': file_identity(features.__file__)}
    dump_atomic({'signature': signature, 'plans': plans, 'selected_variant': choice}, path)
    write_json(path.with_suffix('.json'), {'signature': signature, 'fit_cases': len(selected),
        'variant_selected_on_training': dense.VARIANTS[choice],
        'calibration': {name: {head: {k: w for k, w in plan[h].items() if k != 'models'}
            for h, head in enumerate(dense.HEADS)} for name, plan in zip(dense.VARIANTS, plans)}})


def load_model(args):
    model = joblib.load(args.model)
    for source in model['signature'].values():
        check_file(source)
    if model['signature']['cohort'] != file_identity(args.cohort):
        raise ValueError('Voxel selector was trained on a different cohort')
    return model


def predict(args):
    model = load_model(args)
    checkpoint = file_identity(args.model)
    for row in rows(args.cohort, 'validation')[args.shard_index::args.num_shards]:
        source = _sample(args, row, 'validation')
        check_file(row['prediction'])
        image = nib.load(row['prediction']['path'])
        before = read_labels(image)
        folder = Path(args.output) / row['case_id']
        path = folder / 'prediction.json'
        binding = {'row': row, 'model': checkpoint, 'features': source['arrays']}
        if path.is_file():
            previous = read_json(path)
            if previous['binding'] != binding or previous['gt_used'] is not False:
                raise ValueError('Stored validation prediction differs from the frozen model')
            for p in previous['masks'].values():
                check_file(p)
            continue
        masks, changes = {}, {}
        with np.load(source['arrays']['path'], allow_pickle=False) as archive, threadpool_limits(limits=args.threads):
            for v, name in enumerate(dense.VARIANTS):
                heads = []
                for h, plan in enumerate(model['plans'][v]):
                    tag = f'{v}_{h}'
                    case = {key: archive[f'{key}_{tag}'] for key in ('index', 'x', 'old')}
                    correct = probability(plan['models'], case)
                    threshold = np.where(case['old'], plan['thresholds'][1], plan['thresholds'][0])
                    eligible = correct >= threshold
                    field = np.ascontiguousarray(before > h)
                    field.ravel()[case['index'][eligible]] = ~case['old'][eligible]
                    heads.append(field)
                final = residual.merge_heads(heads)
                changes[name] = int(np.count_nonzero(final != before))
                masks[name] = save_mask(folder / f'{name}.nii.gz', final, image)
        write_json(path, {'binding': binding, 'masks': masks, 'changes': changes, 'gt_used': False})
        print(row['case_id'], changes, flush=True)


def score(args):
    model = load_model(args)
    model_id = file_identity(args.model)
    chosen = rows(args.cohort, 'validation')
    truth = {row['case_id']: row for row in manifest(args.truth, allowed=('validation',))['cases']}
    measures = {name: [] for name in dense.VARIANTS}
    for row in chosen:
        key = row['case_id']
        identity = read_json(Path(args.predictions) / key / 'prediction.json')
        if identity['binding']['row'] != row or identity['binding']['model'] != model_id or identity['gt_used'] is not False:
            raise ValueError('Validation score must use the label-free prediction')
        old, target, reference, volume = scoring._image_pair(row, truth[key])
        for name in dense.VARIANTS:
            source = identity['masks'][name]
            check_file(source)
            image = nib.load(source['path'])
            same_grid(reference, image)
            after = read_labels(image)
            visible = (old > 0) | (target > 0) | (after > 0)
            area = residual._window(visible, padding=0) if visible.any() else np.s_[:, :, :]
            measures[name].extend({'case_id': key, **record} for record in
                                  scoring._measure(old[area], after[area], target[area], volume))
    results = {name: scoring.summary(value) for name, value in measures.items()}
    write_json(args.output, {'cohort': file_identity(args.cohort), 'n_cases': len(chosen), 'model': model_id,
        'variant_selected_on_training': dense.VARIANTS[model['selected_variant']],
        'comparison': results, 'per_case': measures})
    for name, summary in results.items():
        print(name, {head: summary[head]['gain_pp'] for head in dense.HEADS}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    p = commands.add_parser('collect')
    for key in ('cohort', 'probabilities', 'cache', 'previous', 'supervised', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--ready-only', action='store_true')
    for name, keys in (('fit', ('features', 'output')), ('predict', ('features', 'model', 'output')),
                       ('score', ('model', 'predictions', 'truth', 'output'))):
        stage = commands.add_parser(name)
        for key in ('cohort', *keys):
            stage.add_argument('--' + key, required=True)
        if name != 'score':
            stage.add_argument('--threads', default=3, type=int)
        if name == 'fit':
            stage.add_argument('--samples-per-case', default=1500, type=int)
    for stage in (p, commands.choices['predict']):
        stage.add_argument('--shard-index', default=0, type=int)
        stage.add_argument('--num-shards', default=1, type=int)
    args = parser.parse_args()
    action = features.collect if args.stage == 'collect' else globals()[args.stage]
    action(args)


if __name__ == '__main__':
    main()
