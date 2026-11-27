"""Train a 192³ residual correction of expert logits for paired SAM3 proposals."""
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
from . import appearance, dense, posterior_fields, residual
from .cohorts import rows

SEED = 12631


def network(device):
    model = appearance.AppearanceProposal(channels=14, heads=3, prior_bias=False)
    for layer in (model.output, model.native):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    return model.to(device, memory_format=torch.channels_last_3d)


def read_inputs(args, row, split):
    mri, old, brain, truth, lr = appearance.read_case(args.cache, row, split)
    site = Path(args.fields) / split / row['case_id']
    state = read_json(site / 'logits.json')
    clean = {k: v for k, v in row.items() if k != 'gt'}
    if state['binding']['row'] != clean or state['gt_used'] is not False:
        raise ValueError('Expert probability maps do not match the MRI inputs')
    check_file(state['binding']['code'])
    p = np.load(site / 'logits.npy', mmap_mode='r', allow_pickle=False)
    if p.shape != (9, *old.shape):
        raise ValueError('Expert maps must remain on the original MRI grid')
    return mri, old, brain, truth, lr, p


def patch(case, start, rng=None):
    mri, old, brain, truth, lr, p = case
    source, dest = appearance.placement(start, old.shape)
    inputs = np.zeros((14, *appearance.NATIVE_PATCH_SIZE), np.float16)
    inputs[4:13].fill(-1.)
    inputs[(slice(0, 4), *dest)] = mri[(slice(None), *source)]
    inputs[(slice(4, 13), *dest)] = p[(slice(None), *source)]
    flair = mri[3][source]
    mirror = np.flip(mri[3], axis=lr)[source]
    inputs[(13, *dest)] = np.where((flair != 0) & (mirror != 0), flair - mirror, 0.)
    if rng is None:
        return inputs, source, dest
    inputs[:4] *= rng.uniform(.88, 1.12, 4).astype(np.float16)[:, None, None, None]
    if rng.random() < .15:
        missing = int(rng.integers(4))
        inputs[missing].fill(0)
        if missing == 3:
            inputs[13].fill(0)
    target = np.zeros((5, *appearance.NATIVE_PATCH_SIZE), np.uint8)
    target[(slice(0, 3), *dest)] = truth[source][None] > np.arange(3)[:, None, None, None]
    target[(3, *dest)] = brain[source] | (truth[source] > 0)
    target[(4, *dest)] = old[source]
    if rng.random() < .5:
        inputs = np.ascontiguousarray(np.flip(inputs, axis=lr + 1))
        target = np.ascontiguousarray(np.flip(target, axis=lr + 1))
    return inputs, target


def logits(model, inputs):
    anchor = .50 * inputs[:, 4:7].float()
    anchor.add_(inputs[:, 7:10], alpha=.25)
    anchor.add_(inputs[:, 10:13], alpha=.25)
    anchor.mul_(posterior_fields.EXTENT)
    return anchor + 6. * torch.tanh(model(inputs).float() / 3.)


def loss(prediction, target):
    label = target[:, :3].float()
    valid = target[:, 3:4]
    prior = target[:, 4:5] > torch.arange(3, device=target.device)[None, :, None, None, None]
    weights = valid * (.5 + 2.5 * (prior != label) + .5 * (prior | (label > 0)))
    classification = F.binary_cross_entropy_with_logits(prediction, label, weight=weights, reduction='sum')
    probability = prediction.sigmoid() * valid
    dims = (2, 3, 4)
    overlap = (probability * label).sum(dims)
    dice = (2. * overlap + 1.) / ((probability + label).sum(dims) + 1.)
    return classification / weights.sum().clamp_min(1.) + .7 * (1. - dice.mean())


def fit(args):
    patients = rows(args.cohort, 'train', truth=True)
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if len(patients) % world:
        raise ValueError('Training examples must partition evenly across devices')
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    if world > 1:
        torch.distributed.init_process_group('nccl')
    model = network(device)
    wrapped = DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=.0012, weight_decay=.015)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00012)
    amp = torch.amp.GradScaler('cuda', init_scale=256.)
    site = Path(args.output)
    site.mkdir(parents=True, exist_ok=True)
    path = site / 'checkpoint.pth'
    binding = {'cohort': file_identity(args.cohort), 'source': file_identity(__file__),
               'network': file_identity(appearance.__file__), 'field_source': file_identity(posterior_fields.__file__),
               'patch': list(appearance.NATIVE_PATCH_SIZE), 'epochs': args.epochs, 'workers': world}
    start = 0
    if path.is_file():
        prior = torch.load(path, weights_only=True, map_location=device)
        if prior['binding'] != binding:
            raise ValueError('Checkpoint and training protocol differ')
        model.load_state_dict(prior['model'])
        optimizer.load_state_dict(prior['optimizer'])
        schedule.load_state_dict(prior['schedule'])
        amp.load_state_dict(prior['amp'])
        start = prior['epoch']
    try:
        for epoch in range(start, args.epochs):
            tick = monotonic()
            order = np.random.default_rng(SEED+epoch).permutation(len(patients))[rank::world]
            rng = np.random.default_rng(SEED+59*epoch+rank)
            losses = []
            for i in order:
                case = read_inputs(args, patients[i], 'train')
                point = appearance.training_start(case[1], case[2], rng)
                if rng.random() < .20:
                    disagreements = np.argwhere((case[1][::4, ::4, ::4] > 0) !=
                                                (case[3][::4, ::4, ::4] > 0))
                    if len(disagreements):
                        point = 4 * disagreements[rng.integers(len(disagreements))] - appearance.SIDE // 2
                x, y = patch(case, point, rng)
                images = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
                truth = torch.from_numpy(y[None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    estimate = logits(wrapped, images)
                value = loss(estimate, truth)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite regional probability loss')
                amp.scale(value).backward()
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 6.)
                amp.step(optimizer)
                amp.update()
                losses.append(float(value.detach()))
            schedule.step()
            totals = torch.tensor([sum(losses), len(losses)], device=device)
            peak = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
            if world > 1:
                torch.distributed.all_reduce(totals)
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                temp = path.with_suffix('.partial')
                torch.save({'binding': binding, 'epoch': epoch+1, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
                            'amp': amp.state_dict()}, temp)
                temp.replace(path)
                progress = {'epoch': epoch+1, 'n_train': int(totals[1]), 'loss': float(totals[0]/totals[1]),
                            'peak_reserved_gib': float(peak), 'seconds': monotonic()-tick}
                write_json(site / 'progress.json', progress)
                print(progress, flush=True)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    state = torch.load(args.model, map_location=device, weights_only=True)
    record = state['binding']
    if state['epoch'] != record['epochs'] or record['cohort'] != file_identity(args.cohort):
        raise ValueError('Only completed training may supply validation proposals')
    for name in ('source', 'network', 'field_source'):
        check_file(record[name])
    model = network(device)
    model.load_state_dict(state['model'])
    model.eval()
    size = appearance.SIDE
    edge = np.hanning(size).astype(np.float32)
    weights = np.maximum(.002, edge[:, None, None] * edge[None, :, None] * edge[None, None, :])
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    for row in selected:
        site = Path(args.output) / row['case_id']
        path = site / 'proposal.json'
        binding = {'row': row, 'model': file_identity(args.model), 'patch': list(appearance.NATIVE_PATCH_SIZE)}
        if path.is_file():
            previous = read_json(path)
            if previous['binding'] != binding or previous['gt_used'] is not False:
                raise ValueError('Stored cue came from another candidate network')
            check_file(previous['arrays'])
            continue
        case = read_inputs(args, row, 'validation')
        region = case[2]
        starts = [((sl.start + sl.stop - size)//2,) if sl.stop-sl.start <= size
                  else (sl.start, sl.stop-size) for sl in residual._window(region, padding=0)]
        field = np.zeros((3, *region.shape), np.float32)
        norm = np.zeros(region.shape, np.float32)
        for initial in product(*starts):
            x, source, dest = patch(case, initial)
            images = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                estimate = logits(model, images)[0].sigmoid().cpu().numpy()
            field[(slice(None), *source)] += estimate[(slice(None), *dest)] * weights[dest]
            norm[source] += weights[dest]
        if np.any(region & (norm == 0)):
            raise ValueError('192³ inference missed part of the original MRI')
        field /= np.maximum(norm, 1.e-6)
        field[:, ~region] = 0
        np.maximum(field[1], field[2], out=field[1])
        np.maximum(field[0], field[1], out=field[0])
        site.mkdir(parents=True, exist_ok=True)
        array = site / 'proposal.npz'
        dense._atomic_npz(array, probabilities=field.astype(np.float16))
        write_json(path, {'binding': binding, 'arrays': file_identity(array), 'gt_used': False})
        print(row['case_id'], 'native logits fusion', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    fit_args = commands.add_parser('fit')
    predict_args = commands.add_parser('predict')
    for stage in (fit_args, predict_args):
        for field in ('cohort', 'cache', 'fields', 'output'):
            stage.add_argument('--' + field, required=True)
    fit_args.add_argument('--epochs', type=int, default=12)
    predict_args.add_argument('--model', required=True)
    predict_args.add_argument('--device', default='cuda:0')
    predict_args.add_argument('--shard-index', type=int, default=0)
    predict_args.add_argument('--num-shards', type=int, default=1)
    args = parser.parse_args()
    globals()[args.stage](args)
