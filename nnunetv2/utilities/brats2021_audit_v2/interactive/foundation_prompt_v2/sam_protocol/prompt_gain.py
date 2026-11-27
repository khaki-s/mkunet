"""Calibrate signed SAM3 actions against observed 192³ responses on training patients."""
import argparse
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common import check_file, file_identity, manifest, read_json, read_labels, same_grid, save_mask, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import candidates, dense, residual, scoring
from .cohorts import calibration_rows, rows

OFFSETS = np.array([-.04, -.02, 0., .01, .02, .03, .04, .06, .08, .12, .18, .30, 1.])


def evidence(args, row, *, refined_actions=False):
    masks, _, sam = dense.read_masks(args.dense, row)
    replies, response = candidates.read_responses(args.responses, row)
    proposal = read_json(Path(args.proposals) / row['case_id'] / 'proposal.json')
    if (sam['arrays'] != response['sam'] or sam['proposal'] != proposal['arrays']
            or proposal['binding']['row'] != row or proposal['gt_used'] is not False
            or proposal['binding']['model'] != sam['contract']['proposal_model']):
        raise ValueError('Actual responses must be matched to the frozen label-free SAM cues')
    check_file(row['prediction'])
    check_file(proposal['arrays'])
    ref = nib.load(row['prediction']['path'])
    old = read_labels(ref)
    base = (old > 0).ravel()
    editable = old.ravel() <= 1
    replies = replies[:, 0].reshape(2, 3, -1)
    with np.load(sam['arrays']['path'], allow_pickle=False) as data:
        support = data['components'][0].ravel()
        core = data['core_components'][0].ravel()
    with np.load(proposal['arrays']['path'], allow_pickle=False) as data:
        cues = data['probabilities'][0].ravel()
    groups = [[], []]
    old_size = max(1, int(base.sum()))
    for action in sam['actions']:
        if action['head'] != 0:
            continue
        positive = action['positive']
        slot = 1 if positive else 2
        eligible = np.flatnonzero((support == action['id']) & editable & (base != positive))
        if not len(eligible):
            continue
        for i, branch in enumerate(action['variants']):
            if not branch['accept']:
                continue
            point = replies[i, slot, eligible] == positive
            options = [('point', eligible[point])]
            if refined_actions:
                mask = replies[i, 0, eligible] == positive
                options.extend((('mask', eligible[mask]), ('both', eligible[point & mask])))
            seen = []
            for kind, pool in options:
                confidence = cues[pool].astype(np.float32)
                if not positive:
                    confidence = 1. - confidence
                cuts = (0., .55, .62) if refined_actions else (0.,)
                for cut in cuts:
                    keep = confidence >= cut
                    changed = pool[keep]
                    if not len(changed) or any(np.array_equal(changed, prior) for prior in seen):
                        continue
                    seen.append(changed)
                    conf = confidence[keep]
                    others = replies[1-i, slot, changed] == positive
                    full_agreement = masks[i, 0].ravel()[changed] == positive
                    is_core = core[changed] == action['id']
                    features = [float(branch['agreement']), float(branch['quality']),
                                float(min(branch['plane_agreements'])), np.log1p(len(changed)),
                                np.log1p(1_000. * len(changed) / old_size), len(changed) / len(eligible),
                                float(others.mean()), float(full_agreement.mean()), float(is_core.mean()),
                                float(conf.mean()), float(np.quantile(conf, .25)), float(conf.max())]
                    if refined_actions:
                        features += [float(kind == 'mask'), float(kind == 'both'), cut]
                    groups[i].append({'features': features, 'positive': positive, 'indices': changed,
                                      'size': len(changed), 'id': action['id'], 'kind': kind, 'cutoff': cut})
    inputs = {'sam': sam['arrays'], 'responses': response['arrays'], 'cues': proposal['arrays'],
              'cue_model': proposal['binding']['model'], 'cue_plan': proposal['binding']['plan']}
    return old, ref, groups, inputs


def _fit(records, variant):
    models = []
    for positive in (False, True):
        values = [entry for case in records for entry in case['moves'][variant] if entry['positive'] == positive]
        if not values:
            models.append(0.)
            continue
        features = np.asarray([entry['features'] for entry in values])
        targets = np.asarray([entry['correct'] / entry['size'] for entry in values])
        weight = np.asarray([np.sqrt(entry['size'] / entry['denom']) for entry in values])
        if len(values) < 7:
            models.append(float(np.average(targets, weights=weight)))
        else:
            fit = make_pipeline(StandardScaler(), Ridge(alpha=12.))
            fit.fit(features, targets, ridge__sample_weight=weight / weight.mean())
            models.append(fit)
    return models


def _risks(models, actions):
    risks = np.zeros(len(actions))
    for side, fit in enumerate(models):
        selected = [i for i, action in enumerate(actions) if action['positive'] == bool(side)]
        if selected:
            risks[selected] = fit if isinstance(fit, float) else fit.predict(
                np.asarray([actions[i]['features'] for i in selected]))
    return risks.clip(0., 1.)


def _options(moves, risks, limits):
    selected = {}
    for move, score in zip(moves, risks):
        utility = move['size'] * (score - limits[int(move['positive'])])
        key = move['id']
        if utility > 0. and (key not in selected or utility > selected[key][0]):
            selected[key] = (utility, move, score)
    return [item[1:] for item in selected.values()]


def _training_gains(case, moves, risk, threshold):
    tp, denom = case['tp'], case['denom']
    baseline_half = tp / denom
    limits = np.stack((1. - threshold + OFFSETS, threshold + OFFSETS))
    new_tp = np.zeros_like(limits)
    size = np.zeros_like(limits)
    for j, limit in enumerate(limits.T):
        for move, _ in _options(moves, risk, limit):
            side = int(move['positive'])
            size[side, j] += move['size']
            new_tp[side, j] += move['correct'] if side else move['size'] - move['correct']
    numerator = tp + new_tp[1][None, :] - new_tp[0][:, None]
    denominator = denom + size[1][None, :] - size[0][:, None]
    return 200. * (numerator / np.maximum(1., denominator) - baseline_half)


def fit(args):
    ids = calibration_rows(args.cohort, 'train', args.case_list)
    gt = {row['case_id']: row for row in rows(args.cohort, 'train', truth=True)}
    cases = []
    for row in ids:
        key = row['case_id']
        old, ref, moves, inputs = evidence(args, row, refined_actions=args.refined_actions)
        target = residual._truth(gt[key], ref) > 0
        base = old > 0
        tp = int(np.count_nonzero(base & target))
        denom = max(1, int(base.sum() + target.sum()))
        case = {'case_id': key, 'tp': tp, 'denom': denom, 'moves': [[], []], 'inputs': inputs}
        flat = target.ravel()
        for i, actions in enumerate(moves):
            for action in actions:
                correct = int(np.count_nonzero(flat[action['indices']] == action['positive']))
                case['moves'][i].append({k: v for k, v in action.items() if k != 'indices'}
                                        | {'correct': correct, 'denom': denom})
        cases.append(case)
        print(f'{key}: measured {sum(map(len, case["moves"]))} realized training actions', flush=True)
    policies = []
    for i, label in enumerate(dense.VARIANTS):
        outcomes = []
        for fold in range(4):
            fit_cases = [case for j, case in enumerate(cases) if j % 4 != fold]
            models = _fit(fit_cases, i)
            threshold = np.median([case['tp'] / case['denom'] for case in fit_cases])
            for case in cases[fold::4]:
                moves = case['moves'][i]
                outcomes.append(_training_gains(case, moves, _risks(models, moves), threshold))
        gains = np.stack(outcomes)
        selection = gains.mean(0) - .3 * np.maximum(-gains, 0.).mean(0)
        best = np.unravel_index(selection.argmax(), selection.shape)
        if selection[best] <= 1.e-9:
            best = (-1, -1)
        policies.append({'offsets': OFFSETS[np.asarray(best)].tolist(), 'fold_gain_pp': float(gains.mean(0)[best]),
                         'models': _fit(cases, i)})
        print(label, policies[-1]['offsets'], policies[-1]['fold_gain_pp'], flush=True)
    chosen = max(range(len(dense.VARIANTS)), key=lambda i: policies[i]['fold_gain_pp'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    binding = {'cohort': file_identity(args.cohort), 'case_list': file_identity(args.case_list),
               'source': file_identity(__file__), 'refined_actions': args.refined_actions}
    for key in ('cue_model', 'cue_plan'):
        binding[key] = cases[0]['inputs'][key]
        if any(case['inputs'][key] != binding[key] for case in cases):
            raise ValueError('Calibration patients must share one frozen proposal model')
    base_half = float(np.median([case['tp'] / case['denom'] for case in cases]))
    dump_atomic({'binding': binding, 'variants': policies, 'selected': chosen, 'threshold_pivot': base_half}, path)
    write_json(path.with_suffix('.json'), {'binding': binding, 'variant_selected_on_training': dense.VARIANTS[chosen],
                'policies': [{k: v for k, v in a.items() if k != 'models'} for a in policies],
                'threshold_pivot': base_half, 'calibration': cases})


def read_model(args):
    model = joblib.load(args.model)
    for key in ('cohort', 'case_list', 'source', 'cue_model', 'cue_plan'):
        check_file(model['binding'][key])
    if model['binding']['cohort'] != file_identity(args.cohort):
        raise ValueError('The trained policy belongs to another cohort')
    return model


def predict(args):
    model = read_model(args)
    source = file_identity(args.model)
    for row in rows(args.cohort, 'validation')[args.shard_index::args.num_shards]:
        old, ref, moves, inputs = evidence(args, row, refined_actions=model['binding']['refined_actions'])
        if any(inputs[key] != model['binding'][key] for key in ('cue_model', 'cue_plan')):
            raise ValueError('Validation and training must share the same localizer and cue thresholds')
        folder = Path(args.output) / row['case_id']
        bindings = {'row': row, 'model': source, **inputs}
        path = folder / 'prediction.json'
        if path.is_file():
            saved = read_json(path)
            if saved['binding'] != bindings or saved['gt_used'] is not False:
                raise ValueError('Existing prediction was generated from different cues')
            for entry in saved['masks'].values():
                check_file(entry)
            continue
        masks, decisions = {}, {}
        # At deployment use the median frozen training Dice, which remains
        # close to the point where the expected benefit of an edit changes sign.
        base_half = model['threshold_pivot']
        for i, label in enumerate(dense.VARIANTS):
            policy = model['variants'][i]
            quality = _risks(policy['models'], moves[i])
            limits = np.asarray([1. - base_half, base_half]) + policy['offsets']
            wt = (old > 0).ravel()
            accepted = []
            for action, value in _options(moves[i], quality, limits):
                wt[action['indices']] = action['positive']
                accepted.append({'id': action['id'], 'positive': action['positive'],
                                 'voxels': action['size'], 'predicted_correct_fraction': float(value),
                                 'kind': action['kind'], 'cutoff': action['cutoff']})
            result = np.where(old > 1, old, wt.reshape(old.shape)).astype(np.uint8)
            masks[label] = save_mask(folder / f'{label}.nii.gz', result, ref)
            decisions[label] = accepted
        write_json(path, {'binding': bindings, 'masks': masks, 'decisions': decisions, 'gt_used': False})
        print(f'{row["case_id"]}: retained {sum(map(len, decisions.values()))} calibrated native actions', flush=True)


def score(args):
    model = read_model(args)
    expected = file_identity(args.model)
    ref = {q['case_id']: q for q in manifest(args.truth, allowed=('validation',))['cases']}
    measures = {label: [] for label in dense.VARIANTS}
    for row in rows(args.cohort, 'validation'):
        key = row['case_id']
        receipt = read_json(Path(args.predictions) / key / 'prediction.json')
        if receipt['binding']['row'] != row or receipt['binding']['model'] != expected or receipt['gt_used'] is not False:
            raise ValueError('Validation requires the frozen label-free policy output')
        old, truth, reference, volume = scoring._image_pair(row, ref[key])
        for name, identity in receipt['masks'].items():
            check_file(identity)
            image = nib.load(identity['path'])
            same_grid(image, reference)
            new = read_labels(image)
            visible = (old > 0) | (truth > 0) | (new > 0)
            window = residual._window(visible, padding=0) if visible.any() else np.s_[:, :, :]
            measures[name].extend({'case_id': key, **record} for record in
                                  scoring._measure(old[window], new[window], truth[window], volume))
    results = {label: scoring.summary(value) for label, value in measures.items()}
    chosen = dense.VARIANTS[model['selected']]
    write_json(args.output, {'n_cases': len(ref.keys() & {r['case_id'] for r in rows(args.cohort, 'validation')}),
                             'variant_selected_on_training': chosen, 'cohort': file_identity(args.cohort),
                             'model': expected, 'comparison': results, 'per_case': measures})
    for name, value in results.items():
        print(name, {head: value[head]['gain_pp'] for head in dense.HEADS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    for name, fields in (('fit', ('case-list', 'dense', 'responses', 'proposals', 'output')),
                         ('predict', ('model', 'dense', 'responses', 'proposals', 'output')),
                         ('score', ('model', 'predictions', 'truth', 'output'))):
        stage = stages.add_parser(name)
        for field in ('cohort', *fields):
            stage.add_argument('--' + field, required=True)
        if name == 'fit':
            stage.add_argument('--refined-actions', action='store_true')
        if name == 'predict':
            stage.add_argument('--shard-index', type=int, default=0)
            stage.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    globals()[args.stage](args)
