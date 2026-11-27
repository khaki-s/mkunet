"""Learn where to prompt by supervising the signed errors of the original WT mask."""
import argparse
from itertools import product
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ..common import check_file, file_identity, read_json, write_json
from . import appearance, dense, residual, teacher_fields
from .cohorts import rows

SEED = 34573
RADIUS = teacher_fields.RADIUS_MM
LEVELS = np.r_[np.linspace(.025, .975, 39), 1.01]


def groups(cohort):
    patients = rows(cohort, 'train', truth=True)
    # Freeze case-level calibration before optimization and keep the same
    # selection for every variant of the downstream paired prompt.
    selected = np.random.default_rng(SEED).permutation(len(patients))
    return [patients[i] for i in selected[20:]], [patients[i] for i in selected[:20]]


def network(device):
    return appearance.AppearanceProposal((16, 32, 64), channels=9, heads=1, prior_bias=False).to(
        device, memory_format=torch.channels_last_3d)


def case_inputs(args, row, split, *, teacher=False):
    raw, old, brain, truth, lr = appearance.read_case(args.cache, row, split)
    site = Path(args.fields) / split / row['case_id']
    receipt = read_json(site / 'fields.json')
    if receipt['binding']['row'] != row or receipt['labeled'] != (split == 'train'):
        raise ValueError('Signed distance cache does not match its patient')
    check_file(receipt['binding']['code'])
    dist = np.load(site / 'distance.npy', mmap_mode='r', allow_pickle=False)
    depth = np.load(site / 'error.npy', mmap_mode='r', allow_pickle=False) if teacher else None
    return raw, old, brain, truth, lr, dist, depth


def image_patch(case, start, rng=None):
    raw, old, brain, truth, lr, dist, depth = case
    source, dest = appearance.placement(start, old.shape)
    inputs = np.zeros((9, *appearance.NATIVE_PATCH_SIZE), np.float16)
    inputs[(slice(0, 4), *dest)] = raw[(slice(None), *source)]
    for i in range(3):
        inputs[(4 + i, *dest)] = old[source] > i
    flair = raw[3][source]
    mirrored = np.flip(raw[3], axis=lr)[source]
    inputs[(7, *dest)] = np.where((flair != 0) & (mirrored != 0), flair - mirrored, 0.)
    inputs[8].fill(-1.)
    inputs[(8, *dest)] = dist[source] / RADIUS
    if rng is None:
        return inputs, source, dest
    inputs[:4] *= rng.uniform(.85, 1.15, 4).astype(np.float16)[:, None, None, None]
    if rng.random() < .18:
        missing = int(rng.integers(4))
        inputs[missing].fill(0)
        if missing == 3:
            inputs[7].fill(0)
    target = np.zeros((3, *appearance.NATIVE_PATCH_SIZE), np.float16)
    target[(0, *dest)] = depth[source]
    target[(1, *dest)] = truth[source] > 0
    target[(2, *dest)] = (brain[source] | (truth[source] > 0)) & (old[source] <= 1)
    if rng.random() < .5:
        inputs = np.ascontiguousarray(np.flip(inputs, axis=lr + 1))
        target = np.ascontiguousarray(np.flip(target, axis=lr + 1))
    return inputs, target


def objective(logits, source, target):
    depth, truth, allowed = target.unbind(1)
    base = source[:, 4].float()
    depth, truth, allowed = depth.float(), truth.float(), allowed.float()
    near = .05 + .95 * torch.exp(-source[:, 8].float().abs() * RADIUS / 7.)
    labels = (depth > 0).float()
    importance = allowed * torch.where(labels > 0, 3.5 + .5 * depth, near)
    bce = F.binary_cross_entropy_with_logits(logits, labels, weight=importance, reduction='sum')
    probability = logits.sigmoid() * allowed
    corrected = base + (1. - 2. * base) * probability
    dice = (2. * (corrected * truth).sum() + 1.) / (corrected.sum() + truth.sum() + 1.)
    return bce / importance.sum().clamp_min(1.) + 1.5 * (1. - dice)


def fit(args):
    patients, calibration = groups(args.cohort)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if len(patients) % world:
        raise ValueError('The number of workers must divide the fitting cohort')
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    if world > 1:
        torch.distributed.init_process_group('nccl')
    model = network(device)
    wrapped = DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model
    optim = torch.optim.AdamW(wrapped.parameters(), lr=.0015, weight_decay=.015)
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=.00012)
    amp = torch.amp.GradScaler('cuda', init_scale=256.)
    site = Path(args.output)
    site.mkdir(parents=True, exist_ok=True)
    path = site / 'checkpoint.pth'
    binding = {'cohort': file_identity(args.cohort), 'localizer': file_identity(__file__),
               'fields': file_identity(teacher_fields.__file__), 'network': file_identity(appearance.__file__),
               'patch': list(appearance.NATIVE_PATCH_SIZE), 'epochs': args.epochs, 'workers': world,
               'fit_ids': [q['case_id'] for q in patients], 'calibration_ids': [q['case_id'] for q in calibration]}
    start = 0
    if path.exists():
        state = torch.load(path, weights_only=True, map_location=device)
        if state['binding'] != binding:
            raise ValueError('Wrong checkpoint for the frozen teacher experiment')
        model.load_state_dict(state['model'])
        optim.load_state_dict(state['optimizer'])
        decay.load_state_dict(state['decay'])
        amp.load_state_dict(state['amp'])
        start = state['epoch']
    try:
        for epoch in range(start, args.epochs):
            tic = monotonic()
            indices = np.random.default_rng(SEED + epoch).permutation(len(patients))[rank::world]
            rng = np.random.default_rng(SEED + 31 * epoch + rank)
            outcomes = []
            for i in indices:
                case = case_inputs(args, patients[i], 'train', teacher=True)
                raw, old, brain, truth, lr, dist, depth = case
                # Alternate original tumor context with crops centered on a
                # real error. Validation covers the entire brain independently.
                start_xyz = appearance.training_start(old, brain, rng)
                errors = np.argwhere(depth[::4, ::4, ::4] > 0) if rng.random() < .33 else ()
                if len(errors):
                    start_xyz = 4 * errors[rng.integers(len(errors))] - appearance.SIDE // 2
                    start_xyz += rng.integers(-15, 16, 3)
                x, y = image_patch(case, start_xyz, rng)
                inputs = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
                target = torch.from_numpy(y[None]).to(device)
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = wrapped(inputs)[:, 0]
                loss = objective(logits, inputs, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite supervised prompt loss')
                amp.scale(loss).backward()
                amp.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 6.)
                amp.step(optim)
                amp.update()
                outcomes.append(float(loss.detach()))
            decay.step()
            results = torch.tensor([sum(outcomes), len(outcomes)], device=device)
            memory = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
            if world > 1:
                torch.distributed.all_reduce(results)
                torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                temp = path.with_suffix('.partial')
                torch.save({'binding': binding, 'epoch': epoch + 1, 'model': model.state_dict(),
                            'optimizer': optim.state_dict(), 'decay': decay.state_dict(), 'amp': amp.state_dict()}, temp)
                temp.replace(path)
                record = {'epoch': epoch + 1, 'n_fit': int(results[1]), 'n_calibration': len(calibration),
                          'loss': float(results[0] / results[1]), 'peak_reserved_gib': float(memory),
                          'seconds': monotonic() - tic}
                write_json(site / 'progress.json', record)
                print(record, flush=True)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


def load(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    state = torch.load(args.model, weights_only=True, map_location=device)
    for key in ('cohort', 'localizer', 'fields', 'network'):
        check_file(state['binding'][key])
    if state['binding']['cohort'] != file_identity(args.cohort) or state['epoch'] != state['binding']['epochs']:
        raise ValueError('The same fully fitted localizer must be used for calibration and prediction')
    model = network(device)
    model.load_state_dict(state['model'])
    return model.eval(), device


@torch.inference_mode()
def predict_risk(case, model, device):
    _, old, brain, _, _, _, _ = case
    size = appearance.SIDE
    starts = []
    for axis in residual._window(brain, padding=0):
        if axis.stop - axis.start <= size:
            starts.append(((axis.start + axis.stop - size) // 2,))
        else:
            starts.append((axis.start, axis.stop - size))
    taper = np.hanning(size).astype(np.float32)
    weight = np.maximum(.002, taper[:, None, None] * taper[None, :, None] * taper[None, None, :])
    value = np.zeros(old.shape, np.float32)
    norm = np.zeros_like(value)
    for initial in product(*starts):
        x, source, dest = image_patch(case, initial)
        inputs = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(inputs)[0, 0].sigmoid().cpu().numpy()
        value[source] += weight[dest] * p[dest]
        norm[source] += weight[dest]
    if np.any(brain & (norm == 0)):
        raise ValueError('Tiled localizer missed part of the original MRI')
    value /= np.maximum(norm, 1.e-6)
    value[(old > 1) | ~brain] = 0.
    return value


def counts(risk, direction, actual):
    bins = np.searchsorted(LEVELS, risk[direction], side='right')
    prediction = np.bincount(bins, minlength=len(LEVELS) + 1)
    correct = np.bincount(bins, weights=actual[direction], minlength=len(LEVELS) + 1)
    return prediction[:0:-1].cumsum()[::-1], correct[:0:-1].cumsum()[::-1]


def publish(args, row, risk, old, thresholds, plan):
    folder = Path(args.output) / row['case_id']
    folder.mkdir(parents=True, exist_ok=True)
    site = folder / 'proposal.json'
    binding = {'row': row, 'model': file_identity(args.model), 'plan': plan, 'patch': list(appearance.NATIVE_PATCH_SIZE)}
    if site.is_file():
        previous = read_json(site)
        if previous['binding'] != binding or previous['gt_used'] is not False:
            raise ValueError('Existing cue was produced using different calibration')
        check_file(previous['arrays'])
        return
    base = old > 0
    cutoff = np.where(base, thresholds[1], thresholds[0])
    # Map each fitted threshold to the neutral probability 0.5. Maintain
    # the ranking above and below it for the paired component selector.
    rank = np.where(risk >= cutoff, .5 + .5 * (risk - cutoff) / np.maximum(1. - cutoff, .01),
                    .5 * risk / cutoff)
    wt = np.where(base, 1. - rank, rank)
    probabilities = np.stack((np.where(old > 1, 1., wt), old > 1, old > 2)).astype(np.float16)
    arrays = folder / 'proposal.npz'
    dense._atomic_npz(arrays, probabilities=probabilities)
    write_json(site, {'binding': binding, 'arrays': file_identity(arrays), 'gt_used': False})


def calibrate(args):
    model, device = load(args)
    _, patients = groups(args.cohort)
    estimates, quality, paths = [], [], []
    save_risk = Path(args.output).parent / 'train_risk'
    save_risk.mkdir(parents=True, exist_ok=True)
    for row in patients:
        case = case_inputs(args, row, 'train')
        _, old, brain, truth, _, _, _ = case
        risk = predict_risk(case, model, device)
        p = save_risk / f'{row["case_id"]}.npy'
        teacher_fields._write(p, risk.astype(np.float16))
        paths.append(p)
        target, base = truth > 0, old > 0
        adds, atp = counts(risk, brain & ~base, target)
        deletes, dtp = counts(risk, brain & (old == 1), ~target)
        numerator = 2 * np.count_nonzero(base & target)
        denominator = np.count_nonzero(base) + np.count_nonzero(target)
        new = numerator + 2 * (atp[:, None] - (deletes - dtp)[None, :])
        total = denominator + adds[:, None] - deletes[None, :]
        estimates.append(100. * (new / np.maximum(1, total) - numerator / max(1, denominator)))
        detected = np.stack((atp, dtp))
        possible = (np.count_nonzero(target & ~base), np.count_nonzero(~target & (old == 1)))
        quality.append(1.49 * detected / np.maximum(1., np.stack((adds, deletes)) + .49 * np.asarray(possible)[:, None]))
        print(f'{row["case_id"]}: harvested calibration-only candidate statistics', flush=True)
    gains = np.stack(estimates)
    utility = gains.mean(0) - .25 * np.maximum(0, -gains).mean(0)
    pair = np.unravel_index(utility.argmax(), utility.shape)
    best_raw = LEVELS[np.asarray(pair)].tolist()
    fscore = np.stack(quality).mean(0)
    thresholds = LEVELS[fscore.argmax(axis=1)].tolist()
    config = {'model': file_identity(args.model), 'cohort': file_identity(args.cohort),
              'train_calibration': [p['case_id'] for p in patients], 'proposal_thresholds': thresholds,
              'best_train_raw_thresholds': best_raw, 'best_train_raw_gain_pp': float(gains.mean(0)[pair])}
    write_json(args.output, config)
    write_json(Path(args.output).parent / 'calibration_cases.json', config['train_calibration'])
    plan = file_identity(args.output)
    args.output = str(Path(args.output).parent / 'calibration_proposals')
    for row, path in zip(patients, paths):
        _, old, _, _, _, _, _ = case_inputs(args, row, 'train')
        clean = {k: v for k, v in row.items() if k != 'gt'}
        publish(args, clean, np.load(path, mmap_mode='r'), old, thresholds, plan)
    print(config, flush=True)


def predict(args):
    model, device = load(args)
    plan = read_json(args.plan)
    if plan['model'] != file_identity(args.model) or plan['cohort'] != file_identity(args.cohort):
        raise ValueError('Thresholds must have been determined from the model\'s training split')
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    for row in selected:
        case = case_inputs(args, row, 'validation')
        risk = predict_risk(case, model, device)
        publish(args, row, risk, case[1], plan['proposal_thresholds'], file_identity(args.plan))
        print(f'{row["case_id"]}: label-free native WT cue', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    stages = p.add_subparsers(dest='stage', required=True)
    fit_parser = stages.add_parser('fit')
    cal_parser = stages.add_parser('calibrate')
    pred_parser = stages.add_parser('predict')
    for stage in (fit_parser, cal_parser, pred_parser):
        for arg in ('cohort', 'cache', 'fields', 'output'):
            stage.add_argument('--' + arg, required=True)
    fit_parser.add_argument('--epochs', type=int, default=14)
    for stage in (cal_parser, pred_parser):
        stage.add_argument('--model', required=True)
        stage.add_argument('--device', default='cuda:0')
    pred_parser.add_argument('--plan', required=True)
    pred_parser.add_argument('--shard-index', type=int, default=0)
    pred_parser.add_argument('--num-shards', type=int, default=1)
    arguments = p.parse_args()
    globals()[arguments.stage](arguments)
