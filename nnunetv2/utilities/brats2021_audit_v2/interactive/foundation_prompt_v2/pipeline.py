"""Offline feedback learning and GT-free inference of a foundation prompt router."""
from pathlib import Path
import csv
import time
import random
import numpy as np
import nibabel as nib
import torch

from .common import (manifest, load_case, reward, save_mask, runtime_contract, signature,
                     read_json, write_json, file_identity, check_file, region_metrics, REGIONS)
from .executor import FrozenCorrector, execute_plan
from .router import upstream_router, possible_pairs, choose, regression_loss
from .parallel import partition


REWARD_WEIGHTS = {
    'macro': (0.2, 0.2, 0.2, 0.2, 0.2),
    'wt_priority': (0.6, 0.2, 0.2, 0., 0.),
}


def reward_profile(config):
    name = config.get('reward_profile', 'macro')
    if name not in REWARD_WEIGHTS:
        raise ValueError(f'Unknown Dice reward profile: {name}')
    return name


def candidate_reward(before, after, gt, profile):
    macro, deltas = reward(before, after, gt)
    return (macro if profile == 'macro' else float(deltas @ REWARD_WEIGHTS[profile])), deltas


def generator(config, device):
    from .sam3_candidates import Sam3PromptCandidates
    return Sam3PromptCandidates(**config['sam3'], device=device)


def load_router(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    check_file(state['router_source'])
    model = upstream_router(state['router_source']['path'], state['cfg']).to(device)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    return model, state


@torch.inference_mode()
def router_scores(model, state, context, device):
    g = torch.as_tensor((context - state['mean']) / state['std'], dtype=torch.float32, device=device)[None]
    u, _, s = model.forward_scores(g)
    # The upstream PairRouter can return bfloat16 logits on CUDA. Convert them
    # exactly to float32 before crossing the NumPy action-selection boundary.
    return u[0].float().cpu().numpy(), s[0].float().cpu().numpy()


def collect(args, *, proposal_generator=None, corrector_factory=FrozenCorrector):
    source = manifest(args.manifest, allowed=('train',))
    config = read_json(args.config)
    profile = reward_profile(config)
    contract = runtime_contract(config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'foundation_feedback_v2', 'source_manifest': file_identity(args.manifest),
               'contract': contract, 'rounds': args.rounds, 'pairs': args.pairs,
               'save_adapter_pairs': args.save_adapter_pairs,
               'rollout_policy': file_identity(args.rollout_policy) if args.rollout_policy else None,
               'split': 'train', 'limit_cases': args.limit_cases}
    rows = source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    rows = partition(rows, args, binding)
    index_path = root / 'dataset.json'
    if index_path.exists():
        previous = read_json(index_path)
        if previous['binding'] != binding:
            raise ValueError('Collection output exists with a different contract; use a new folder')
        completed = dict(previous['completed'])
    else:
        completed = {}
    proposal_generator = proposal_generator or generator(config, args.device)
    rollout = load_router(args.rollout_policy, args.device) if args.rollout_policy else None
    if rollout and rollout[1]['contract'] != contract:
        raise ValueError('On-policy collection must use the same foundation/correction contract')
    engine = None
    for row in rows:
        key = row['case_id']
        if key in completed:
            for identity in completed[key]['files']:
                check_file(identity)
            continue
        mri, current, reference, gt = load_case(row, with_gt=True)
        corrector = corrector_factory(mri, row.get('availability', [1]*4), reference.affine,
                                      config['correction_protocol'],
                                      device=getattr(args, 'nninteractive_device', None) or args.device,
                                      engine=engine)
        engine = getattr(getattr(corrector, 'backend', None), 'engine', None)
        history, records, files, adapter_rows = [], [], [], []
        for turn in range(args.rounds):
            # Only these arguments reach the foundation model. GT is not an
            # input to candidate generation or execution, even during training.
            generated = proposal_generator.generate(mri, current, row.get('availability', [1]*4), history,
                                                     affine=reference.affine)
            actions, context = generated['actions'], generated['context']
            if len(actions) != 6:
                raise ValueError('Six fixed head/polarity slots required')
            valid = np.array([a is not None for a in actions], bool)
            utility, delta = np.zeros(6, np.float32), np.zeros((6, 5), np.float32)
            pair_values, pair_valid = np.zeros((6, 6), np.float32), np.zeros((6, 6), bool)
            proposals, option_values = {}, []
            for i in np.flatnonzero(valid):
                i = int(i)
                output, _, _ = execute_plan(corrector, current, actions, (i,), history)
                utility[i], delta[i] = candidate_reward(current, output, gt, profile)
                proposals[(i,)] = output
                option_values.append((float(utility[i]), (i,)))
                if args.save_adapter_pairs:
                    pair_path = root / key / f'turn{turn:02d}_candidate{i}.npz'
                    pair_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(pair_path, mri=mri, baseline=current, proposal=output,
                                        gt=gt, head=('WT', 'TC', 'ET').index(actions[i]['head']))
                    identity = file_identity(pair_path)
                    files.append(identity)
                    adapter_rows.append({'case_id': key, 'file': str(pair_path.resolve())})
            if args.pairs:
                for i, j in possible_pairs(actions):
                    output, _, _ = execute_plan(corrector, current, actions, (i, j), history)
                    value, _ = candidate_reward(current, output, gt, profile)
                    pair_values[i, j] = pair_values[j, i] = value
                    pair_valid[i, j] = True  # count one unordered pair only
                    proposals[(i, j)] = output
                    option_values.append((value, (i, j)))
            state_path = root / key / f'turn{turn:02d}.npz'
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(state_path, context=context, utility=utility, valid=valid,
                                pairs=pair_values, pair_valid=pair_valid, region_delta=delta)
            files.append(file_identity(state_path))
            if rollout:
                U, S = router_scores(*rollout, context, args.device)
                indices, value = choose(U, S, actions, remaining=2, margin_pp=args.rollout_margin_pp,
                                        pairs=args.pairs)
            else:
                value, indices = max(option_values, default=(0.0, ()), key=lambda x: x[0])
                if value <= 0:
                    indices = ()
            records.append({'case_id': key, 'turn': turn, 'state': file_identity(state_path),
                            'actions': actions, 'metadata': generated['metadata'],
                            'chosen': list(indices), 'selection': 'learned' if rollout else 'train_GT_reward',
                            'selected_score_pp': float(value), 'history_length': len(history)})
            if not indices:
                break
            current = proposals[tuple(indices)]
            history.extend(actions[i] for i in indices)
        completed[key] = {'files': files, 'records': records, 'adapter_rows': adapter_rows}
        write_json(index_path, {'binding': binding, 'completed': completed,
                               'full_cohort': len(completed) == len(source['cases'])})
        print(f'collect {len(completed)}/{len(rows)} {key}: {len(records)} states', flush=True)
    # Empty shards also need an index for a checked merge.
    if not index_path.exists():
        write_json(index_path, {'binding': binding, 'completed': completed,
                               'full_cohort': len(completed) == len(source['cases'])})
    if args.save_adapter_pairs:
        write_json(root / 'adapter_pairs.json', {'schema': 'foundation_adapter_pairs_v1', 'split': 'train',
                   'source_manifest': str(Path(args.manifest).resolve()),
                   'cases': [r for case in completed.values() for r in case['adapter_rows']]})
    return index_path


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    records, datasets, contract = [], [], None
    for path in args.datasets:
        data = read_json(path)
        binding = data['binding']
        if binding.get('schema') != 'foundation_feedback_v2' or binding['split'] != 'train':
            raise ValueError('Only train320 feedback may update router parameters')
        check_file(binding['source_manifest'])
        source = manifest(binding['source_manifest']['path'], allowed=('train',))
        allowed = {r['case_id'] for r in source['cases']}
        if contract is not None and binding['contract'] != contract:
            raise ValueError('Cannot mix different foundation/correction protocols')
        contract = binding['contract']
        datasets.append(file_identity(path))
        for key, case in data['completed'].items():
            if key not in allowed:
                raise ValueError('Foreign patient in feedback dataset')
            for record in case['records']:
                if record['case_id'] != key:
                    raise ValueError('Patient record mismatch')
                check_file(record['state'])
                with np.load(record['state']['path'], allow_pickle=False) as archive:
                    item = {name: archive[name].copy() for name in
                            ('context', 'utility', 'valid', 'pairs', 'pair_valid')}
                if item['valid'].any():
                    records.append(item)
    if not records:
        raise ValueError('No valid measured candidates; inspect SAM3 proposal coverage first')
    check_file(contract['router_source'])
    profile = reward_profile(contract['config'])
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new training output directory')
    root.mkdir(parents=True, exist_ok=True)
    x = np.stack([r['context'] for r in records]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite foundation features')
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    tensors = [torch.from_numpy((x-mean)/std)] + [torch.from_numpy(np.stack([r[k] for r in records]))
                   for k in ('utility', 'valid', 'pairs', 'pair_valid')]
    cfg = dict(E=6, d_ctx=x.shape[1], d_hidden=128, n_layers=3, dropout=0.1)
    model = upstream_router(contract['router_source']['path'], cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),
                                        batch_size=args.batch_size, shuffle=True, num_workers=0)
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            batch = [x.to(args.device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss = regression_loss(model, *batch)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite utility regression')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses))})
        state = {'schema': 'foundation_pair_router_v2', 'state_dict': model.cpu().state_dict(),
                 'cfg': cfg, 'mean': mean, 'std': std, 'contract': contract,
                 'router_source': contract['router_source'], 'datasets': datasets,
                 'pairs_trained': bool(tensors[-1].any()), 'epochs': epoch+1,
                 'reward': {'profile': profile, 'weights': dict(zip(REGIONS, REWARD_WEIGHTS[profile])),
                            'unit': 'Dice percentage points', 'both_empty_dice': 1.},
                 'training_history': history}
        torch.save(state, root / 'checkpoint_last.pth')
        model.to(args.device)
        print(history[-1], flush=True)
    write_json(root / 'training.json', {'history': history, 'records': len(records),
               'train_only': True, 'datasets': datasets, 'config': vars(args)})
    return root / 'checkpoint_last.pth'


def feedback_report(args):
    data = read_json(args.dataset)
    binding = data['binding']
    profile = reward_profile(binding['contract']['config'])
    check_file(binding['source_manifest'])
    manifest(binding['source_manifest']['path'], allowed=('train',))
    rows, slots, first_turn_head_best = [], [[] for _ in range(6)], []
    region_slots = [[] for _ in range(6)]
    margin = getattr(args, 'margin_pp', 0.1)
    for case in data['completed'].values():
        for record in case['records']:
            check_file(record['state'])
            with np.load(record['state']['path'], allow_pickle=False) as value:
                u, valid = value['utility'], value['valid']
                scores = u[valid].tolist() + value['pairs'][value['pair_valid']].tolist()
                for i in np.flatnonzero(valid):
                    slots[i].append(float(u[i]))
                    if 'region_delta' in value.files:
                        region_slots[i].append(value['region_delta'][i].astype(float))
                if record['turn'] == 0 and 'region_delta' in value.files:
                    first_turn_head_best.append({
                        head: max([0., *value['region_delta'][valid, i].tolist()])
                        for i, head in enumerate(REGIONS[:3])
                    })
                rows.append({'case_id': record['case_id'], 'turn': record['turn'],
                             'candidate_count': int(valid.sum()),
                             'positive_candidates': int((u[valid] > 0).sum()),
                             'best_measured_gain_pp': max([0.0, *scores]),
                             'candidate_exceeds_margin': any(x > margin for x in scores)})
    names = ('WT+', 'WT-', 'TC+', 'TC-', 'ET+', 'ET-')
    result = {'dataset': file_identity(args.dataset), 'cases': len(data['completed']),
              'states': len(rows), 'full_train320': data['full_cohort'],
              'states_with_beneficial_candidate': sum(r['best_measured_gain_pp'] > 0 for r in rows),
              'mean_best_training_gain_pp': float(np.mean([r['best_measured_gain_pp'] for r in rows])) if rows else None,
              'reward': {'profile': profile, 'weights': dict(zip(REGIONS, REWARD_WEIGHTS[profile]))},
              'train_GT_best_single_action_per_region_pp': {
                  head: float(np.mean([case[head] for case in first_turn_head_best]))
                  for head in REGIONS[:3]} if first_turn_head_best else None,
              'margin_pp': margin,
              'states_with_candidate_exceeding_margin': sum(r['candidate_exceeds_margin'] for r in rows),
              'slots': {n: {'count': len(v), 'positive_count': sum(x > 0 for x in v),
                       'mean_gain_pp': float(np.mean(v)) if v else None,
                       'mean_region_delta_pp': dict(zip(REGIONS, np.mean(d, axis=0).tolist())) if d else None}
                        for n, v, d in zip(names, slots, region_slots)},
              'interpretation': 'GT-scored train candidate coverage, NOT deployable performance or a test bound'}
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'feedback_summary.json', result)
    if rows:
        with (root / 'feedback_states.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return root / 'feedback_summary.json'


def predict(args, *, proposal_generator=None, corrector_factory=FrozenCorrector):
    source = manifest(args.manifest, deployment=True, allowed=('validation', 'test'))
    model, state = load_router(args.checkpoint, args.device)
    config = read_json(args.config)
    contract = runtime_contract(config)
    if contract != state['contract']:
        raise ValueError('Model/source/config changed since feedback collection; use the bound version')
    max_actions = config.get('max_actions', 5)
    if type(max_actions) is not int or not 1 <= max_actions <= 5:
        raise ValueError('max_actions must be an integer from 1 to 5')
    deployment = {'checkpoint': file_identity(args.checkpoint), 'contract': contract,
                  'margin_pp': args.margin_pp, 'pairs': bool(args.pairs), 'budgets': [0, 1, 3, 5]}
    if max_actions != 5:
        deployment['max_actions'] = max_actions
    if args.pairs and not state['pairs_trained']:
        raise ValueError('Pair routing was not supervised in this checkpoint')
    if source['split'] == 'test':
        if not args.validation_lock:
            raise ValueError('Score full val80 and create a passing validation-lock before test100')
        locked = read_json(args.validation_lock)
        if not locked.get('passed') or locked['deployment'] != deployment:
            raise ValueError('Validation lock failed or belongs to a different method')
        check_file(locked['validation_summary'])
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'source_manifest': file_identity(args.manifest), 'deployment': deployment,
               'split': source['split'], 'limit_cases': args.limit_cases}
    index_path = root / 'predictions.json'
    if index_path.exists():
        previous = read_json(index_path)
        if previous['binding'] != binding:
            raise ValueError('Existing output belongs to a different inference run')
        completed = dict(previous['completed'])
    else:
        completed = {}
    proposal_generator = proposal_generator or generator(config, args.device)
    engine = None
    rows = source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    for row in rows:
        key = row['case_id']
        if key in completed:
            for value in completed[key]['snapshots'].values():
                check_file(value)
            continue
        started = time.perf_counter()
        mri, current, reference = load_case(row)  # GT is not loaded or searched for.
        corrector = corrector_factory(mri, row.get('availability', [1]*4), reference.affine,
                                      config['correction_protocol'],
                                      device=getattr(args, 'nninteractive_device', None) or args.device,
                                      engine=engine)
        engine = getattr(getattr(corrector, 'backend', None), 'engine', None)
        snapshots = {'0': save_mask(root / key / 'budget_0.nii.gz', current, reference)}
        history, events, count, stopped = [], [], 0, False
        budget_stats = {'0': {'rounds': 0, 'points': 0, 'boxes': 0, 'seconds': 0.0}}
        for budget in (1, 3, 5):
            while count < min(budget, max_actions) and not stopped:
                generated = proposal_generator.generate(mri, current, row.get('availability', [1]*4), history,
                                                         affine=reference.affine)
                U, S = router_scores(model, state, generated['context'], args.device)
                indices, value = choose(U, S, generated['actions'], remaining=min(budget, max_actions)-count,
                                        margin_pp=args.margin_pp, pairs=args.pairs)
                event = {'at_round': count, 'chosen': list(indices), 'predicted_gain_pp': value,
                         'metadata': generated['metadata'], 'U': U.tolist(), 'S': S.tolist()}
                if not indices:
                    future, _ = choose(U, S, generated['actions'], remaining=min(2, max_actions-count),
                                       margin_pp=args.margin_pp, pairs=args.pairs)
                    if future and len(future) > budget-count:
                        # A useful pair cannot fit budget1. Preserve this state
                        # and defer to budget3 instead of declaring global STOP.
                        event['deferred_for_budget'] = True
                        events.append(event)
                        break
                    stopped = True
                    event['stop'] = True
                else:
                    current, history, details = execute_plan(corrector, current, generated['actions'], indices, history)
                    count += len(indices)
                    event['actions'] = [generated['actions'][i] for i in indices]
                    # Keep finite action audit; backend records can contain tensors.
                    event['tool_calls'] = len(details)
                events.append(event)
            snapshots[str(budget)] = save_mask(root / key / f'budget_{budget}.nii.gz', current, reference)
            budget_stats[str(budget)] = {'rounds': count,
                'points': sum(x['kind'] in ('point', 'point_box') for x in history),
                'boxes': sum(x['kind'] in ('box', 'point_box') for x in history),
                'seconds': time.perf_counter()-started}
        completed[key] = {'snapshots': snapshots, 'events': events, 'budgets': budget_stats,
                          'stopped': stopped, 'seconds': time.perf_counter()-started}
        write_json(index_path, {'schema': 'foundation_predictions_v2', 'binding': binding,
                   'completed': completed, 'full_cohort': len(completed) == len(source['cases']),
                   'contains_gt': False})
        print(f'predict {len(completed)}/{len(rows)} {key} calls={count} stop={stopped}', flush=True)
    return index_path


def score(args):
    source = manifest(args.manifest, allowed=('validation', 'test'))
    predictions = read_json(args.predictions)
    check_file(predictions['binding']['source_manifest'])
    deploy_source = manifest(predictions['binding']['source_manifest']['path'], deployment=True)
    if source['split'] != deploy_source['split']:
        raise ValueError('Score split differs from predictions')
    by_id = {r['case_id']: r for r in deploy_source['cases']}
    for row in source['cases']:
        if {k: v for k, v in row.items() if k != 'gt'} != by_id.get(row['case_id']):
            raise ValueError('Scoring inputs differ from deployment inputs')
    records, deltas = [], {str(b): {r: [] for r in REGIONS} for b in (1, 3, 5)}
    paired_baselines = {b: {r: [] for r in REGIONS} for b in deltas}
    empty_fp = {str(b): {r: {'new_cases': 0, 'new_voxels': 0} for r in REGIONS} for b in (1, 3, 5)}
    truth_ids = {row['case_id'] for row in source['cases']}
    if not set(predictions['completed']).issubset(truth_ids):
        raise ValueError('Foreign predictions')
    for row in source['cases']:
        key = row['case_id']
        if key not in predictions['completed']:
            continue
        check_file(row['gt'])
        gt_image = nib.load(row['gt']['path'])
        from ..evaluation import read_labels, same_grid
        gt = read_labels(gt_image)
        baseline = None
        for budget in ('0', '1', '3', '5'):
            identity = predictions['completed'][key]['snapshots'][budget]
            check_file(identity)
            image = nib.load(identity['path'])
            same_grid(image, gt_image)
            labels = read_labels(image)
            if budget == '0':
                check_file(row['prediction'])
                initial = nib.load(row['prediction']['path'])
                same_grid(initial, image)
                if not np.array_equal(labels, read_labels(initial)):
                    raise ValueError('Budget0 was changed')
            metrics = region_metrics(labels, gt)
            if baseline is None:
                baseline = metrics
            for region in REGIONS:
                value, original = metrics[region]['dice'], baseline[region]['dice']
                delta = None if value is None or original is None else 100*(value-original)
                records.append({'case_id': key, 'budget': budget, 'region': region,
                                'dice_percent': None if value is None else 100*value,
                                'delta_pp': delta,
                                **predictions['completed'][key]['budgets'][budget]})
                if budget != '0' and delta is not None:
                    deltas[budget][region].append(delta)
                    paired_baselines[budget][region].append(100 * original)
                if budget != '0' and metrics[region]['n_ref'] == 0:
                    extra = max(0, metrics[region]['fp'] - baseline[region]['fp'])
                    empty_fp[budget][region]['new_cases'] += int(extra > 0)
                    empty_fp[budget][region]['new_voxels'] += extra
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError('No completed predictions to score')
    with (root / 'per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: f'{v:.3f}' if isinstance(v, float) else v for k, v in r.items()} for r in records)
    summary = {'split': source['split'], 'cases': len(predictions['completed']),
               'full_cohort': set(predictions['completed']) == truth_ids,
               'deployment': predictions['binding']['deployment'],
               'predictions': file_identity(args.predictions), 'scores': {},
               'empty_reference_added_fp': empty_fp}
    for budget, regions in deltas.items():
        summary['scores'][budget] = {r: {'mean_delta_pp': float(np.mean(v)) if v else None,
           'paired_baseline_dice_percent': float(np.mean(paired_baselines[budget][r])) if v else None,
           'paired_final_dice_percent': float(np.mean(np.asarray(paired_baselines[budget][r]) + v)) if v else None,
           'n': len(v), 'improved': sum(x > 1e-4 for x in v), 'worsened': sum(x < -1e-4 for x in v),
           'unchanged': sum(abs(x) <= 1e-4 for x in v)} for r, v in regions.items()}
    write_json(root / 'summary.json', summary)
    return root / 'summary.json'


def validation_lock(args):
    summary = read_json(args.summary)
    values = summary['scores']['5']
    # A conservative development criterion, not proof of per-patient benefit.
    passed = (summary['split'] == 'validation' and summary['full_cohort'] and summary['cases'] == 80
              and all(values[r]['mean_delta_pp'] is not None and values[r]['mean_delta_pp'] >= 0 for r in REGIONS)
              and all(summary['empty_reference_added_fp']['5'][r]['new_voxels'] == 0 for r in REGIONS)
              and values['WT']['mean_delta_pp'] > 0)
    result = {'passed': bool(passed), 'deployment': summary['deployment'],
              'validation_summary': file_identity(args.summary),
              'criterion': 'Complete val80, budget5 WT >0, other region mean deltas >=0, no added FP in GT-empty regions',
              'warning': 'Validation selection is not a guarantee of test or individual improvement'}
    write_json(args.output, result)
    if not passed:
        raise ValueError('Validation criterion failed. Do not promote this method to test100.')
    return args.output
