"""WT pilot for a supervised readout of frozen 192³ nnInteractive features."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ...inference_runtime import NATIVE_PATCH_SIZE, prediction_context
from ..common import check_file, file_identity, manifest, read_json, save_mask, write_json
from . import candidates, dense, residual, scoring, spatial
from .cohorts import rows

SCHEMA = 'nninteractive_native_wt_probe_v1'
VARIANT = 'native_wt_probe'


def _source():
    return {**spatial._source_files(), 'probe': file_identity(__file__)['sha256']}


def _decoder_features(executor, volume, base, prompts, xyz):
    engine = executor.engine
    destination = np.ascontiguousarray(base, dtype=np.uint8).copy()
    engine.set_image(np.ascontiguousarray(volume[None]))
    engine.set_target_buffer(destination)
    center = prompts[-1]['point']
    layer = engine.network.decoder.seg_layers[-1]
    features = []
    with prediction_context(engine):
        engine.add_initial_seg_interaction(destination.copy(), run_prediction=False)
        for prompt in prompts:
            engine.add_bbox_interaction(prompt['box'], include_interaction=prompt['positive'], run_prediction=False)
            engine.add_point_interaction(prompt['point'], include_interaction=prompt['positive'], run_prediction=False)
        x, patch, window, previous = engine._build_network_input(center, 1.)
        del previous
        engine.new_interaction_centers.clear()
        engine.new_interaction_zoom_out_factors.clear()
        if tuple(patch) != NATIVE_PATCH_SIZE or tuple(x.shape[1:]) != NATIVE_PATCH_SIZE:
            raise ValueError('nnInteractive feature extraction must use 192³')
        start, stop = np.asarray(window).T
        keep = np.all((xyz >= start) & (xyz < stop), axis=1)
        local = torch.as_tensor(xyz[keep] - start, device=executor.device)
        index = tuple(local[:, axis] for axis in range(3))

        def capture(_layer, inputs):
            voxels = inputs[0][0].permute(1, 2, 3, 0)
            features.append(voxels[index].contiguous())

        hook = layer.register_forward_pre_hook(capture)
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                logits = engine.network(x[None].contiguous())[0]
            margin = (logits[1] - logits[0])[index]
        finally:
            hook.remove()
    if len(features) != 1:
        raise ValueError('Expected a single forward through the original segmentation head')
    return features[0].float().cpu().numpy(), margin.float().cpu().numpy(), keep


def _cached_sources(native, distances, output, row, labeled):
    record = read_json(Path(native) / row['case_id'] / 'volume.json')
    distance = read_json(Path(distances) / row['case_id'] / 'WT' / 'distance.json')
    if record['binding']['row'] != row or record['labeled'] != labeled:
        raise ValueError('Native readout cache mismatch')
    if distance['binding']['inputs']['baseline'] != record['volumes']['baseline']:
        raise ValueError('Native WT boundary cache mismatch')
    signature = {'source': _source(), 'native': record, 'distance': distance['arrays']['baseline']}
    receipt = Path(output) / row['case_id'] / 'features.json'
    if receipt.exists():
        existing = read_json(receipt)
        if existing['binding'] != signature:
            raise ValueError('Existing interactive feature cache differs')
        check_file(existing['samples'])
        return None
    for product in record['volumes'].values():
        check_file(product)
    for product in ('sam', 'responses'):
        check_file(record['binding'][product])
    check_file(signature['distance'])
    return record, distance, signature, receipt


def _collect_case(executor, record, distance, row, labeled):
    inputs = record['volumes']
    mask = np.load(inputs['baseline']['path'], mmap_mode='r', allow_pickle=False)
    base = mask > 0
    brain = np.load(inputs['brain']['path'], mmap_mode='r', allow_pickle=False)
    brain = np.asarray(brain, dtype=bool)
    mri = np.load(inputs['mri']['path'], mmap_mode='r', allow_pickle=False)
    spacing = np.linalg.norm(np.asarray(record['affine'])[:3, :3], axis=0)
    with np.load(record['binding']['sam']['path'], allow_pickle=False) as source:
        sam = source['masks'][:, 0]
    responses = spatial._unpack(record['binding']['responses']['path'], base.shape, 0)
    support = brain & np.any(responses != base, axis=0)
    evidence = np.where(base, sam.any(axis=0), sam.all(axis=0))
    frame = row['modalities']['FLAIR']
    check_file(frame)
    source = nib.load(frame['path']).get_fdata(dtype=np.float32)
    prompts = [candidates.locate_component(evidence, base, source, spacing, direction)
               for direction in (False, True)]
    prompts = [prompt for prompt in prompts if prompt is not None]
    possible = np.flatnonzero(support)
    if not len(possible) or not prompts:
        return {'coords': np.empty(0, np.int32), 'features': np.empty((0, 0), np.float16),
                'old': np.empty(0, dtype=bool)}, prompts
    xyz = np.stack(np.unravel_index(possible, base.shape), axis=1)
    latent, margin, keep = _decoder_features(executor, source, base, prompts, xyz)
    coords = possible[keep]
    index = tuple(xyz[keep].T)
    priors = np.load(distance['arrays']['baseline']['path'], mmap_mode='r', allow_pickle=False)
    features = np.column_stack((latent, margin, priors[index], base[index],
                                mri[(slice(None), *index)].T, sam[(slice(None), *index)].T,
                                responses[(slice(None), *index)].T))
    payload = {'coords': coords.astype(np.int32), 'features': features.astype(np.float16), 'old': base[index]}
    if labeled:
        gt = np.load(inputs['truth']['path'], mmap_mode='r', allow_pickle=False) > 0
        payload['target'] = gt[index]
        payload['total'] = int(gt.sum() + base.sum())
        payload['tp'] = int((gt & base).sum())
    return payload, prompts


def collect_probe(args):
    labeled = args.split == 'train'
    chosen = rows(args.cohort, args.split, truth=labeled)[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    executor = None
    for row in chosen:
        result = _cached_sources(args.volumes, args.distances, args.output, row, labeled)
        if result is None:
            continue
        record, distance, signature, receipt = result
        if executor is None:
            if torch.cuda.mem_get_info(device)[0] < 8 * 2**30:
                raise RuntimeError('Need 8 GiB free for original 192³ nnInteractive feature extraction')
            executor = candidates.NativeExecutor(str(device))
        payload, prompts = _collect_case(executor, record, distance, row, labeled)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        data = receipt.parent / 'features.npz'
        dense._atomic_npz(data, **payload)
        write_json(receipt, {'schema': SCHEMA, 'binding': signature, 'samples': file_identity(data),
                             'gt_used': labeled, 'prompts': prompts})
        print('Interactive features', row['case_id'], args.split, len(payload['coords']), flush=True)
    print('probe peak allocated/reserved GiB', torch.cuda.max_memory_allocated(device) / 2**30,
          torch.cuda.max_memory_reserved(device) / 2**30, flush=True)
    return args.output


def _records(selected, data, labeled):
    for row in selected:
        receipt = read_json(Path(data) / row['case_id'] / 'features.json')
        binding = receipt['binding']
        if (receipt['schema'] != SCHEMA or binding['source'] != _source()
                or binding['native']['binding']['row'] != row or receipt['gt_used'] != labeled):
            raise ValueError('Probe cache split or case mismatch')
        check_file(receipt['samples'])
        with np.load(receipt['samples']['path'], allow_pickle=False) as values:
            yield row, dict(values), receipt


def _network(n_features, device):
    torch.manual_seed(20261118)
    return nn.Sequential(nn.Linear(n_features, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 1)).to(device)


def _predict(model, data, device, mean, scale):
    with torch.inference_mode():
        inputs = torch.as_tensor(data, dtype=torch.float32, device=device)
        values = model((inputs - mean) / scale).sigmoid().reshape(-1)
        return values.cpu().numpy()


def _thresholds(probabilities, case):
    def counts(mask):
        return int(mask.sum()), int(np.count_nonzero(mask & case['target']))
    n = int(case['total'])
    tp = int(case['tp'])
    original = 2. * tp / n if n else 1.
    b = case['old']
    options = (*residual.THRESHOLDS, None)
    add = [counts((~b) & (probabilities >= p)) if p is not None else (0, 0) for p in options]
    remove = [counts(b & (probabilities <= p)) if p is not None else (0, 0) for p in options]
    return np.asarray([[100. * ((2. * (tp + u - v) / (n + a - d) if n + a - d else 1.) - original)
                        for d, v in remove] for a, u in add])


def fit_probe(args):
    if args.epochs < 1:
        raise ValueError('Expected at least one epoch')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    train, calibrate = [], []
    rng = np.random.default_rng(61201)
    for i, (_, cache, _) in enumerate(_records(rows(args.cohort, 'train', truth=True), args.features, True)):
        if not len(cache['coords']):
            continue
        if i % 5 == 4:
            calibrate.append(cache)
        else:
            sample = rng.choice(len(cache['coords']), min(30000, len(cache['coords'])), replace=False)
            train.append((cache['features'][sample], cache['old'][sample], cache['target'][sample]))
    x, old, labels = [np.concatenate(col) for col in zip(*train)]
    data = torch.as_tensor(x, device=device, dtype=torch.float32)
    old = torch.as_tensor(old, device=device)
    target = torch.as_tensor(labels, device=device, dtype=torch.float32)
    mean, scale = data.mean(dim=0), data.std(dim=0).clamp_min(.025)
    data.sub_(mean).div_(scale)
    network = _network(data.shape[1], device)
    optim = torch.optim.AdamW(network.parameters(), lr=.001, weight_decay=.01)
    for step in range(args.epochs):
        order = rng.permutation(len(x))
        losses = []
        for begin in range(0, len(x), 8192):
            idx = torch.as_tensor(order[begin:begin + 8192], device=device)
            fit = network(data[idx]).flatten()
            w = 1. + .5 * (target[idx] != old[idx]).float()
            loss = F.binary_cross_entropy_with_logits(fit, target[idx], weight=w)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
        print(f'Interactive readout epoch {step + 1}/{args.epochs} loss={np.mean(losses):.5f}', flush=True)
    network.eval()
    scores = np.stack([_thresholds(_predict(network, row['features'], device, mean, scale), row)
                       for row in calibrate])
    objective = scores.mean(0) - .18 * np.maximum(0., -scores).mean(0)
    options = (*residual.THRESHOLDS, None)
    choice = np.unravel_index(np.argmax(objective), objective.shape) if objective.max() > 0 else (len(options)-1,)*2
    calibration = {'add': options[choice[0]], 'remove': options[choice[1]],
                   'train_calibration_gain_pp': float(scores[:, choice[0], choice[1]].mean())}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'cohort': file_identity(args.cohort), 'method': _source(), 'weights': network.cpu().state_dict(),
                'mean': mean.cpu(), 'scale': scale.cpu(), 'calibration': calibration}, output)
    print(calibration, flush=True)
    return output


def predict_probe(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    state = torch.load(args.model, map_location='cpu', weights_only=True)
    if state['cohort'] != file_identity(args.cohort) or state['method'] != _source():
        raise ValueError('Probe weights belong to a different profile')
    mean, scale = state['mean'].to(device), state['scale'].to(device)
    model = _network(len(mean), device)
    model.load_state_dict(state['weights'])
    model.eval()
    chosen = rows(args.cohort, 'validation')
    for row, record, prior in _records(chosen, args.features, False):
        orig = nib.load(row['prediction']['path'])
        new = np.ascontiguousarray(dense.read_labels(orig)).copy()
        coords = record['coords']
        if len(coords):
            p = _predict(model, record['features'], device, mean, scale)
            limits = state['calibration']
            added, removed = residual._changed(p, coords, new > 0, limits['add'], limits['remove'])
            new.flat[coords[added]] = 1
            new.flat[coords[removed]] = 0
        folder = Path(args.output) / row['case_id']
        result = save_mask(folder / 'wt_probe.nii.gz', new, orig)
        write_json(folder / 'prediction.json', {'gt_used': False, 'row': row, 'mask': result,
                   'model': file_identity(args.model), 'prompts': prior['prompts']})
    return args.output


def score_probe(args):
    selected = rows(args.cohort, 'validation')
    truth = manifest(args.truth, allowed=('validation',))
    refs = {row['case_id']: row for row in truth['cases']}
    records = []
    for row in selected:
        path = Path(args.predictions) / row['case_id'] / 'prediction.json'
        receipt = read_json(path)
        if receipt['row'] != row or receipt['gt_used'] is not False or receipt['model'] != file_identity(args.model):
            raise ValueError('Prediction and learned readout differ')
        check_file(receipt['mask'])
        old, target, ref, volume = scoring._image_pair(row, refs[row['case_id']])
        prediction = nib.load(receipt['mask']['path'])
        scoring.same_grid(prediction, ref)
        for region in scoring._measure(old, dense.read_labels(prediction), target, volume):
            records.append({'case_id': row['case_id'], **region})
    report = {'schema': SCHEMA, 'n_cases': len(selected), 'model': file_identity(args.model),
              'cohort': file_identity(args.cohort), 'regions': scoring.summary(records)}
    root = Path(args.output)
    write_json(root / 'score.json', report)
    scoring._csv(root / 'per_case_metrics.csv', records)
    return root / 'score.json'
