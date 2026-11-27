"""Train-only SAM prompt supervision, nnInteractive feedback and blind validation."""
from pathlib import Path
from time import monotonic, sleep

import joblib
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..common import (REGIONS, check_file, file_identity, load_case, manifest, read_json,
                     read_labels, region_metrics, same_grid, save_mask, write_json)
from ..executor import FrozenCorrector
from ...prompt_sampling import select_prompt
from .sam_component_cache import load_masks, rows_for
from .sam_prompt_components import (GROUPS, HEADS, component_features, dump_atomic, propose,
                                    train_direction)

THRESHOLDS = (.45, .55, .65, .75, .85)
PATCH_SIZE = (192, 192, 192)
FOLDS = 4


def partition(rows, args):
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('Invalid shard index')
    return rows[args.shard_index::args.num_shards]


def _artifact(root, key, row, receipt, inputs):
    folder = Path(root) / key
    path = folder / receipt
    binding = {'case': key, 'row': row, **inputs}
    if path.exists():
        prior = read_json(path)
        previous = dict(prior['binding'])
        code = previous.get('code')
        if isinstance(code, dict):
            previous['code'] = code['sha256']
        if previous != binding:
            raise ValueError(f'{key}: output binding changed')
        for item in prior['files']:
            check_file(item)
        return folder, path, binding, prior
    folder.mkdir(parents=True, exist_ok=True)
    return folder, path, binding, None


def _save_features(path, data):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **data)
    temporary.replace(path)
    return file_identity(path)


def features(args):
    is_train = args.split == 'train'
    rows = partition(rows_for(args.cohort, (args.split,), truth=is_train), args)
    code_sha256 = file_identity(Path(__file__).with_name('sam_prompt_components.py'))['sha256']
    pending = {r['case_id']: r for r in rows}
    deadline = monotonic() + args.wait
    while pending:
        for key, row in list(pending.items()):
            receipt = Path(args.dense) / key / 'candidate.json'
            if not receipt.is_file():
                continue
            dense_row = {k: v for k, v in row.items() if k != 'gt'}
            masks, _ = load_masks(args.dense, dense_row)
            folder, index, binding, prior = _artifact(args.output, key, row, 'features.json',
                {'dense': file_identity(receipt), 'code': code_sha256, 'profile': args.profile})
            if prior is None:
                mri, baseline, reference = load_case(dense_row)
                data = component_features(mri, baseline, masks, reference.affine, connected=args.profile == 'connected')
                if is_train:
                    check_file(row['gt'])
                    image = nib.load(row['gt']['path'])
                    same_grid(reference, image)
                    gt = read_labels(image)
                    counts = []
                    error_totals = []
                    for head in range(3):
                        pred, truth = baseline > head, gt > head
                        counts.append([np.count_nonzero(pred), np.count_nonzero(truth), np.count_nonzero(pred & truth)])
                        error_totals.extend([np.count_nonzero(truth & ~pred), np.count_nonzero(pred & ~truth)])
                    data['dice_counts'] = np.array(counts, np.int64)
                    data['error_totals'] = np.array(error_totals, np.int64)
                    for group, (head, positive) in enumerate(GROUPS):
                        if f'index_{group}' in data:
                            locations = data[f'index_{group}']
                            data[f'target_{group}'] = (gt.ravel()[locations] > head) == positive
                path = _save_features(folder / 'features.npz', data)
                write_json(index, {'binding': binding, 'files': [path], 'gt_used': is_train})
                sizes = [len(data.get(f'index_{g}', [])) for g in range(len(GROUPS))]
                print(f'features {args.split} {key}: eligible SAM voxel counts {sizes}', flush=True)
            del pending[key]
        if pending:
            if monotonic() >= deadline:
                raise ValueError(f'Missing SAM cache for {list(pending)}')
            sleep(12)
    return args.output


def direction(args):
    rows = rows_for(args.cohort, ('train',), truth=True)
    keys = [row['case_id'] for row in rows]
    files = {}
    for row in rows:
        key = row['case_id']
        record = read_json(Path(args.features) / key / 'features.json')
        if record['binding']['row'] != row or not record['gt_used']:
            raise ValueError('Direction learning requires bindings to the train manifest')
        files[key] = record['files'][0]
        check_file(files[key])
    sources = {key: receipt['path'] for key, receipt in files.items()}
    fold_count = min(FOLDS, len(keys))
    if fold_count < 2:
        raise ValueError('Training needs at least two patients for cross-fitting')
    folds = {key: i % fold_count for i, key in enumerate(keys)}
    crossfit = []
    for fold in range(fold_count):
        crossfit.append(train_direction(sources, [key for key in keys if folds[key] != fold]))
        print(f'direction fit fold {fold+1}/{fold_count}', flush=True)
    final = train_direction(sources, keys)
    dump_atomic({'schema': 'sam3_direction_v1', 'cohort': file_identity(args.cohort), 'features': files,
                 'folds': folds, 'crossfit': crossfit, 'final': final}, args.output)
    return args.output


def _interaction_protocol(path):
    cfg = read_json(path)['correction_protocol']
    if not cfg.get('correction', {}).get('full_image_context'):
        raise ValueError('Correction must retain full MRI context')
    if not cfg.get('conv3d_depth_tile'):
        raise ValueError('Configure bounded convolution workspace at native 192³')
    return cfg


def _dice_change(counts, correct, number, positive):
    existing, truth, overlap = (int(x) for x in counts)
    if not number:
        return 0.
    new_size = existing + (number if positive else -number)
    new_overlap = overlap + (correct if positive else -(number - correct))
    before = 2 * overlap / (existing + truth) if existing + truth else 1.
    after = 2 * new_overlap / (new_size + truth) if new_size + truth else 1.
    return 100 * (after - before)


def _estimate_gain(n, number, confidence, positive):
    # Approximate the denominator from initial volume and a fixed Dice prior.
    # Actual gain supervision below uses the exact Dice from the train GT.
    old_dice = .94
    overlap_change = number * (confidence if positive else -(1 - confidence))
    size_change = number if positive else -number
    return 100 * (2 * overlap_change - old_dice * size_change) / max(1., 2 * n + size_change)


def connected_edit(locations, probabilities, shape, threshold):
    """Progressive merge of a single confidence-supported 3D response."""
    selected = probabilities >= threshold
    coords = np.column_stack(np.unravel_index(locations[selected], shape))
    if len(coords) < 5:
        return np.empty(0, dtype=np.int64)
    native = tuple((coords - coords.min(0)).T)
    active = np.zeros(tuple(coords.max(0) - coords.min(0) + 1), dtype=bool)
    active[native] = True
    labels, n = ndimage.label(active)
    ids = labels[native]
    size = np.bincount(ids, minlength=n+1)
    score = np.bincount(ids, weights=probabilities[selected] - .5, minlength=n+1)
    score[size < 5] = -np.inf
    score[0] = -np.inf
    best = int(np.argmax(score))
    mask = selected.copy()
    mask[selected] = ids == best if score[best] > 0 else False
    return np.flatnonzero(mask)


def action_options(data, corrector, baseline, models):
    """Execute finite prompts and constrain accepted edits to their SAM component."""
    flat_base = baseline.ravel()
    connected = bool(data.get('connected', False))
    for prompt, locations, probability in propose(data, models):
        group = prompt['group']
        head, positive = GROUPS[group]
        output, _ = corrector(baseline, prompt, ())
        accepted = ((output.ravel()[locations] > head) == positive)
        changed = locations[accepted]
        probabilities = probability[accepted]
        number_raw = np.count_nonzero((output > head) != (baseline > head))
        proposed = []
        for threshold in ((.40, .50, .6, .7, .8, .9) if connected else THRESHOLDS):
            use = connected_edit(changed, probabilities, baseline.shape, threshold) if connected else probabilities >= threshold
            picked = changed[use]
            if len(picked) < 3:
                continue
            p = probabilities[use]
            count = len(p)
            volume = float(data['sizes'][head])
            expected = _estimate_gain(volume, count, float(p.mean()), positive)
            vector = [expected, count / max(1., volume), float(p.mean()), float(p.min()), float(p.std()),
                      prompt['anchor_probability'], np.log1p(count), np.log1p(float(number_raw)),
                      np.log1p(prompt['component_voxels']), np.log1p(prompt['component_slices']),
                      np.log1p(volume), float(positive), threshold]
            if np.any((flat_base[picked] > head) == positive):
                raise ValueError('SAM edit contradicts the original direction')
            proposed.append((threshold, picked, vector))
        yield prompt, int(number_raw), proposed


def _cuda_setup(name):
    device = torch.device(name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, _ = torch.cuda.mem_get_info(device)
        if free < 8 * 2**30:
            raise RuntimeError(f'{device} needs 8 GiB initial free memory for 192³')
        torch.cuda.reset_peak_memory_stats(device)
    return device


def _check_patch(engine):
    if tuple(engine.configuration_manager.patch_size) != PATCH_SIZE:
        raise ValueError('nnInteractive must preserve its checkpoint-native 192³ patch')


def _stats(output, args, device, count, executed, start):
    path = Path(output) / f'{args.stage}_shard{args.shard_index}.json'
    if not executed and path.exists():
        return
    value = {'completed': count, 'new_cases': executed, 'seconds': monotonic() - start,
             'patch_size': list(PATCH_SIZE)}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        value.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    write_json(path, value)


def feedback(args):
    rows = partition(rows_for(args.cohort, ('train',), truth=True), args)
    weights = joblib.load(args.direction)
    if weights['cohort'] != file_identity(args.cohort):
        raise ValueError('Direction model was fitted to a different patient cohort')
    cfg = _interaction_protocol(args.config)
    device = _cuda_setup(args.device)
    started = monotonic()
    engine = None
    executed = 0
    for row in rows:
        key = row['case_id']
        features_path = weights['features'][key]
        check_file(features_path)
        folder, path, binding, prior = _artifact(args.output, key, row, 'feedback.json',
            {'direction': file_identity(args.direction), 'features': features_path, 'config': file_identity(args.config)})
        if prior is not None:
            continue
        clean_row = {k: v for k, v in row.items() if k != 'gt'}
        mri, baseline, reference = load_case(clean_row)
        corrector = FrozenCorrector(mri, row['availability'], reference.affine, cfg,
                                    device=str(device), engine=engine)
        engine = corrector.backend.engine
        _check_patch(engine)
        with np.load(features_path['path'], allow_pickle=False) as data:
            models = weights['crossfit'][weights['folds'][key]]
            edit_options = list(action_options(data, corrector, baseline, models))
            counts = data['dice_counts']
            errors = data['error_totals'].tolist()
            coverage = [int(data[f'target_{g}'].sum()) if f'target_{g}' in data else 0 for g in range(len(GROUPS))]
        # Only the post-execution labels access GT.
        check_file(row['gt'])
        truth = nib.load(row['gt']['path'])
        same_grid(reference, truth)
        gt = read_labels(truth)
        truth_flat = gt.ravel()
        teacher = select_prompt(baseline, gt, reference.affine, target='joint', kind='point_box',
                                seed=20261118, box_max_extent=24)
        teacher_gain = None
        if teacher is not None:
            teacher_mask, _ = corrector(baseline, teacher, ())
            h = HEADS.index(teacher['head'])
            before, after = baseline > h, teacher_mask > h
            truth_binary = gt > h
            changed = (after != before)
            positive = teacher['positive']
            if (after[changed] == positive).all():
                teacher_gain = _dice_change(counts[h], np.count_nonzero(truth_binary[changed] == positive),
                                            np.count_nonzero(changed), positive)
        candidates = []
        for prompt, raw, options in edit_options:
            head, positive = GROUPS[prompt['group']]
            record = {'prompt': prompt, 'raw_edited_voxels': raw,
                      'anchor_direction_correct': bool((truth_flat[np.ravel_multi_index(prompt['voxel'], baseline.shape)] > head) == positive),
                      'options': []}
            for threshold, picked, vector in options:
                correct = np.count_nonzero((truth_flat[picked] > head) == positive)
                gain = _dice_change(counts[head], correct, len(picked), positive)
                record['options'].append({'threshold': threshold, 'vector': vector, 'gain_pp': gain,
                                          'correct_voxels': int(correct), 'voxels': len(picked)})
            candidates.append(record)
        payload = folder / 'actions.json'
        write_json(payload, {'case_id': key, 'fold': weights['folds'][key], 'candidates': candidates,
                             'train_only_teacher': teacher, 'teacher_gain_pp': teacher_gain,
                             'true_error_counts': errors, 'sam_coverage_counts': coverage})
        write_json(path, {'binding': binding, 'files': [file_identity(payload)]})
        executed += 1
        print(f'feedback {key}: {len(candidates)} model calls / '
              f'{sum(len(r["options"]) for r in candidates)} eligible edits', flush=True)
    _stats(args.output, args, device, len(rows), executed, started)
    return args.output


def _fit_value(options):
    if not options:
        return None
    x = np.array([row['vector'] for row in options])
    actual = np.array([row['gain_pp'] for row in options])
    # Learn the residual of estimated Dice using independently executed actions.
    return make_pipeline(StandardScaler(), Ridge(alpha=70.)).fit(x, actual - x[:, 0])


def _values(options, model, weight):
    x = np.array([row['vector'] for row in options])
    if model is None:
        return x[:, 0]
    return x[:, 0] + weight * model.predict(x)


def _pick(options, model, weight, margin, scores=None):
    if not options or margin is None:
        return None
    estimates = _values(options, model, weight) if scores is None else scores
    eligible = np.array([row['vector'][0] > 0 for row in options])
    estimates = np.where(eligible, estimates, -np.inf)
    best = int(np.argmax(estimates))
    return best if estimates[best] > margin else None


def _direction_choice(options, directions):
    selected, best_score = None, -np.inf
    for positive in (True, False):
        spec = directions[int(not positive)]
        eligible = [o for o in options if bool(o['vector'][11]) == positive]
        if not eligible or spec['margin'] is None:
            continue
        scores = _values(eligible, spec['model'], spec['weight'])
        i = int(np.argmax(scores))
        if scores[i] > spec['margin'] and scores[i] > best_score:
            selected, best_score = eligible[i], float(scores[i])
    return selected, best_score


def fit_direction_routes(cases, folds, head):
    """Add/remove have different Dice risks and distinct calibration models."""
    candidates = []
    for positive in (True, False):
        by_patient = {key: [o for o in parts[head] if bool(o['vector'][11]) == positive]
                      for key, parts in cases.items()}
        models = {fold: _fit_value([o for key, options in by_patient.items() if folds[key] != fold for o in options])
                  for fold in set(folds.values())}
        model = _fit_value([o for options in by_patient.values() for o in options])
        choices = []
        for weight in (.35, .7, 1.):
            predictions = {}
            for key, options in by_patient.items():
                if options:
                    scores = _values(options, models[folds[key]], weight)
                    i = int(np.argmax(scores))
                    predictions[key] = (float(scores[i]), options[i]['gain_pp'])
            for margin in (.001, .008, .025, .06, .15):
                outcomes = {key: (s, y) for key, (s, y) in predictions.items() if s > margin}
                choices.append((outcomes, {'model': model, 'margin': margin, 'weight': weight}))
        choices.append(({}, {'model': model, 'margin': None, 'weight': 0.}))
        candidates.append(choices)
    strategies = []
    for additions, add_model in candidates[0]:
        for removals, remove_model in candidates[1]:
            gains = [max(additions.get(key, (-np.inf, 0.)), removals.get(key, (-np.inf, 0.)),
                         key=lambda result: result[0])[1] for key in cases]
            objective = float(np.mean(gains) - .5 * np.mean(np.maximum(0., -np.asarray(gains))))
            enabled = int(add_model['margin'] is not None) + int(remove_model['margin'] is not None)
            strategies.append((objective, -enabled, gains, [add_model, remove_model]))
    objective, _, gains, routes = max(strategies, key=lambda trial: trial[:2])
    if objective <= 0.:
        routes = [{**spec, 'margin': None} for spec in routes]
        gains = [0.] * len(cases)
    return {'directions': routes, 'oof_gains': gains}


def policy(args):
    rows = rows_for(args.cohort, ('train',), truth=True)
    crossfit = joblib.load(args.direction)
    if crossfit['cohort'] != file_identity(args.cohort):
        raise ValueError('Direction model uses a different cohort')
    cases = {row['case_id']: [[], [], []] for row in rows}
    audited, coverage_rows = [], {}
    feedback_ids = []
    for row in rows:
        key = row['case_id']
        path = Path(args.feedback) / key / 'feedback.json'
        index = read_json(path)
        feedback_ids.append(file_identity(path))
        if (index['binding']['row'] != row or index['binding']['direction'] != file_identity(args.direction)
                or index['binding']['config'] != file_identity(args.config)):
            raise ValueError(f'{key}: feedback identity changed')
        receipt, = index['files']
        check_file(receipt)
        candidate = read_json(receipt['path'])
        if candidate['fold'] != crossfit['folds'][key]:
            raise ValueError('Feedback must use the direction model that did not see the patient')
        coverage_rows[key] = {k: candidate[k] for k in ('true_error_counts', 'sam_coverage_counts', 'teacher_gain_pp')}
        for record in candidate['candidates']:
            h = HEADS.index(record['prompt']['head'])
            cases[key][h].extend(record['options'])
            audited.append((key, record))
    selection = []
    for h, head in enumerate(HEADS):
        if args.routing == 'independent-directions':
            spec = fit_direction_routes(cases, crossfit['folds'], h)
            selection.append(spec)
            print(f'policy {head} directional margins={[v["margin"] for v in spec["directions"]]} '
                  f'OOF gain={np.mean(spec["oof_gains"]):.5f} pp', flush=True)
            continue
        fold_models = {
            fold: _fit_value([option for key, parts in cases.items() if crossfit['folds'][key] != fold
                              for option in parts[h]])
            for fold in set(crossfit['folds'].values())
        }
        values = []
        for residual_weight in (0., .35, .7, 1.):
            scores = {key: _values(parts[h], fold_models[crossfit['folds'][key]], residual_weight)
                      for key, parts in cases.items() if parts[h]}
            for margin in (.001, .01, .025, .06, .12, .25):
                gains = []
                for key, parts in cases.items():
                    choice = _pick(parts[h], None, residual_weight, margin, scores.get(key))
                    gains.append(0. if choice is None else parts[h][choice]['gain_pp'])
                objective = float(np.mean(gains) - .5 * np.mean(np.maximum(0., -np.array(gains))))
                values.append((objective, margin, residual_weight, gains))
        best = max(values, key=lambda v: (v[0], v[1], -v[2]))
        final = _fit_value([o for parts in cases.values() for o in parts[h]])
        selection.append({'model': final, 'margin': best[1] if best[0] > 0 else None,
                          'weight': best[2], 'oof_gains': best[3] if best[0] > 0 else [0.] * len(cases)})
        print(f'policy {head} margin={selection[-1]["margin"]} '
              f'OOF gain={np.mean(selection[-1]["oof_gains"]):.5f} pp', flush=True)
    bundle = {'schema': 'sam3_execution_selector_v1', 'heads': selection, 'train_ids': list(cases),
              'routing': args.routing,
              'cohort': file_identity(args.cohort), 'direction': file_identity(args.direction),
              'correction': file_identity(args.config), 'feedback_files': feedback_ids}
    dump_atomic(bundle, args.output)
    def describe(spec):
        result = {k: v for k, v in spec.items() if k not in ('model', 'directions')}
        if 'directions' in spec:
            result['directions'] = [{k: v for k, v in route.items() if k != 'model'} for route in spec['directions']]
        return result

    diagnostics = {'patient_ids': list(cases), 'prompts': len(audited), 'routing': args.routing,
                   'correct_directions': sum(r['anchor_direction_correct'] for _, r in audited),
                   'coverage_and_gt_teacher': coverage_rows,
                   'heads': {head: {**describe(spec),
                                     'mean_oof_gain_pp': float(np.mean(spec['oof_gains'])),
                                     'mean_oracle_gain_pp': float(np.mean([max([0., *(x['gain_pp'] for x in p[h])])
                                                                          for p in cases.values()]))}
                             for h, (head, spec) in enumerate(zip(HEADS, selection))}}
    write_json(Path(args.output).with_suffix('.json'), diagnostics)
    return args.output


def predict(args):
    rows = partition(rows_for(args.cohort, ('validation',)), args)
    cohort_id = file_identity(args.cohort)
    direction_model = joblib.load(args.direction)
    chooser = joblib.load(args.policy)
    if (direction_model['cohort'] != cohort_id or chooser['cohort'] != cohort_id
            or chooser['direction'] != file_identity(args.direction)
            or chooser['correction'] != file_identity(args.config)):
        raise ValueError('Prediction policy and inputs are not bound to this frozen configuration')
    cfg = _interaction_protocol(args.config)
    device = _cuda_setup(args.device)
    started = monotonic()
    engine = None
    executed = 0
    for row in rows:
        key = row['case_id']
        feature_index = read_json(Path(args.features) / key / 'features.json')
        if feature_index['binding']['row'] != row or feature_index['gt_used'] or 'gt' in row:
            raise ValueError('Prediction inputs must be GT-free')
        feature_id, = feature_index['files']
        check_file(feature_id)
        folder, path, binding, prior = _artifact(args.output, key, row, 'prediction.json',
                                                {'policy': file_identity(args.policy), 'features': feature_id})
        if prior is not None:
            continue
        mri, baseline, reference = load_case(row)
        corrector = FrozenCorrector(mri, row['availability'], reference.affine, cfg,
                                    device=str(device), engine=engine)
        engine = corrector.backend.engine
        _check_patch(engine)
        choices = [[], [], []]
        evaluated = []
        with np.load(feature_id['path'], allow_pickle=False) as data:
            for prompt, raw, options in action_options(data, corrector, baseline, direction_model['final']):
                h = HEADS.index(prompt['head'])
                evaluated.append({'prompt': prompt, 'raw_edited_voxels': raw, 'options': len(options)})
                for threshold, picked, vector in options:
                    choices[h].append({'prompt': prompt, 'threshold': threshold, 'picked': picked, 'vector': vector})
        result = baseline.copy()
        selected = []
        # Each action changes exactly one head. Adjacent edits must agree at
        # shared voxels. Allocate WT first; let TC/ET use the remaining space.
        for head, options in enumerate(choices):
            spec = chooser['heads'][head]
            compatible = [row for row in options if np.all(result.ravel()[row['picked']] == baseline.ravel()[row['picked']])]
            if 'directions' in spec:
                edit, value = _direction_choice(compatible, spec['directions'])
            else:
                best = _pick(compatible, spec['model'], spec['weight'], spec['margin'])
                edit = None if best is None else compatible[best]
                value = None if edit is None else float(_values([edit], spec['model'], spec['weight'])[0])
            if edit is None:
                selected.append({'head': HEADS[head], 'stop': True})
                continue
            positive = edit['prompt']['positive']
            result.ravel()[edit['picked']] = head + int(positive)
            selected.append({'head': HEADS[head], 'prompt': edit['prompt'], 'threshold': edit['threshold'],
                             'voxels': len(edit['picked']),
                             'predicted_gain_pp': value})
        after = save_mask(folder / 'final.nii.gz', result, reference)
        write_json(path, {'binding': binding, 'files': [after], 'selected': selected,
                          'evaluated': evaluated, 'gt_used': False})
        executed += 1
        print(f'predict {key}: {[r.get("voxels",0) for r in selected]} voxels by head', flush=True)
    _stats(args.output, args, device, len(rows), executed, started)
    return args.output


def score(args):
    source = manifest(args.manifest, allowed=('validation',))
    selected = rows_for(args.cohort, ('validation',))
    truth_rows = {row['case_id']: row for row in source['cases']}
    records = []
    receipts = []
    policy_id = file_identity(args.policy)
    for clean in selected:
        key = clean['case_id']
        row = truth_rows[key]
        if clean != {k: v for k, v in row.items() if k != 'gt'}:
            raise ValueError('GT manifest is bound to different images')
        path = Path(args.predictions) / key / 'prediction.json'
        identity = file_identity(path)
        prediction = read_json(path)
        receipts.append(identity)
        if prediction['gt_used'] or prediction['binding']['row'] != clean or prediction['binding']['policy'] != policy_id:
            raise ValueError('The completed prediction is not bound to the intended inputs')
        final, = prediction['files']
        for item in (row['prediction'], row['gt'], final):
            check_file(item)
        before, after, truth = (nib.load(t['path']) for t in (row['prediction'], final, row['gt']))
        same_grid(before, truth)
        same_grid(after, truth)
        gt = read_labels(truth)
        metrics = [region_metrics(read_labels(image), gt) for image in (before, after)]
        regions = {}
        for h in REGIONS:
            b, a = [m[h]['dice'] for m in metrics]
            regions[h] = {'baseline_pct': None if b is None else b * 100,
                          'final_pct': None if a is None else a * 100,
                          'gain_pp': None if b is None or a is None else (a - b) * 100,
                          'new_fp_in_empty_gt': max(0, int(metrics[1][h]['fp']-metrics[0][h]['fp']))
                          if metrics[1][h]['n_ref'] == 0 else 0}
        records.append({'case_id': key, 'regions': regions, 'actions': prediction['selected']})
    means = {}
    for h in REGIONS:
        valid = [r['regions'][h] for r in records if r['regions'][h]['gain_pp'] is not None]
        means[h] = {'n': len(valid), **{name: float(np.mean([v[name] for v in valid])) if valid else None
                                       for name in ('baseline_pct', 'final_pct', 'gain_pp')}}
    write_json(args.output, {'schema': 'sam3_component_validation_v1', 'policy': policy_id,
                            'cohort': file_identity(args.cohort), 'case_count': len(records),
                            'predictions': receipts, 'regions': means, 'per_case': records})
    return args.output
