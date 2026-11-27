"""Train a native-resolution MRI proposal network for paired SAM3 prompts."""
import argparse
from itertools import product
import os
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ...inference_runtime import NATIVE_PATCH_SIZE
from ..common import check_file, file_identity, manifest, read_json, write_json
from . import dense, residual, scoring
from .cohorts import rows

SIDE = NATIVE_PATCH_SIZE[0]
SEED = 28591


def convolution(cin, cout, stride=1):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(4, cout), nn.SiLU())


class AppearanceProposal(nn.Module):
    def __init__(self, widths=(12, 20, 32), *, channels=8, heads=3, prior_bias=True):
        super().__init__()
        fine, mid, deep = widths
        self.prior_bias = prior_bias
        self.stem = convolution(channels, fine, 2)
        self.mid = nn.Sequential(convolution(fine, mid, 2), convolution(mid, mid))
        self.deep = nn.Sequential(convolution(mid, deep, 2), convolution(deep, deep))
        self.coarse = convolution(mid + deep, mid)
        self.merge = convolution(fine + mid, fine)
        self.output = nn.Conv3d(fine, heads, 1)
        self.native = nn.Conv3d(channels, heads, 1)

    def forward(self, x):
        a = self.stem(x)
        b = self.mid(a)
        depth = self.deep(b)
        q = self.coarse(torch.cat((b, F.interpolate(depth, size=b.shape[2:], mode='trilinear',
                                                    align_corners=False)), 1))
        half = self.merge(torch.cat((a, F.interpolate(q, size=a.shape[2:], mode='trilinear',
                                                     align_corners=False)), 1))
        logits = self.output(half)
        logits = self.native(x).float() + F.interpolate(logits.float(), size=x.shape[2:], mode='trilinear',
                                                       align_corners=False)
        return logits + (x[:, 4:7].float() - .5) if self.prior_bias else logits


def read_case(root, row, split):
    path = Path(root) / split / row['case_id']
    index = read_json(path / 'volume.json')
    if index['binding']['row'] != row or index['labeled'] != (split == 'train'):
        raise ValueError('Cached MRI does not match the frozen split')
    for k, value in index['volumes'].items():
        if not (path / f'{k}.npy').is_file():
            raise FileNotFoundError(value['path'])
    raw = np.load(path / 'mri.npy', mmap_mode='r')
    old = np.load(path / 'baseline.npy', mmap_mode='r')
    brain = np.load(path / 'brain.npy', mmap_mode='r')
    truth = np.load(path / 'truth.npy', mmap_mode='r') if split == 'train' else None
    codes = nib.aff2axcodes(np.asarray(index['affine']))
    left_right = next(i for i, code in enumerate(codes) if code in ('L', 'R'))
    return raw, old, brain, truth, left_right


def placement(start, shape):
    intervals = tuple((max(0, int(lo)), min(n, int(lo) + SIDE)) for lo, n in zip(start, shape))
    source = tuple(slice(a, b) for a, b in intervals)
    destination = tuple(slice(a - int(lo), b - int(lo)) for (a, b), lo in zip(intervals, start))
    return source, destination


def training_start(old, brain, rng):
    field = old[::4, ::4, ::4] > 0
    coordinates = np.argwhere(field if field.any() else brain[::4, ::4, ::4])
    center = (coordinates.min(0) + coordinates.max(0)) * 2
    return center.astype(int) - SIDE // 2 + rng.integers(-15, 16, 3)


def corrupt(prior, rng):
    choice = rng.random()
    if choice < .24 or not prior.any():
        return prior
    if choice < .35:
        return np.zeros_like(prior)
    if choice < .52:
        return ndimage.shift(prior, rng.integers(-4, 5, 3), order=0, mode='constant', prefilter=False)
    foreground = np.argwhere(prior[::3, ::3, ::3] > 0)
    if not len(foreground):
        return prior
    center = 3 * foreground[rng.integers(len(foreground))]
    radius = rng.integers(7, 37, 3)
    lower = np.maximum(0, center - radius)
    upper = np.minimum(prior.shape, center + radius + 1)
    patch = tuple(slice(a, b) for a, b in zip(lower, upper))
    grid = np.ogrid[patch]
    intersects = sum(((x - c) / r) ** 2 for x, c, r in zip(grid, center, radius)) <= 1.
    area = prior[patch]
    if choice < .79:
        area[intersects] = 0
    else:
        grown = ndimage.maximum_filter(area, size=int(rng.choice((5, 9, 13))))
        area[intersects] = grown[intersects]
    return prior


def patch_input(raw, old, brain, start, left_right, *, rng=None, truth=None, use_prior=True):
    source, dest = placement(start, old.shape)
    x = np.zeros((8, *NATIVE_PATCH_SIZE), dtype=np.float16)
    x[(slice(0, 4), *dest)] = raw[(slice(None), *source)]
    prior = np.zeros(NATIVE_PATCH_SIZE, dtype=np.uint8)
    if use_prior:
        prior[dest] = old[source]
    if rng is not None:
        prior = corrupt(prior, rng)
        x[:4] *= rng.uniform(.82, 1.18, 4).astype(np.float16)[:, None, None, None]
        if rng.random() < .25:
            x[int(rng.integers(4))] = 0
    for h in range(3):
        x[4+h] = prior > h
    mirror = np.flip(raw[3], axis=left_right)[source]
    real = raw[3][source]
    x[(7, *dest)] = np.where((real != 0) & (mirror != 0), real - mirror, 0.)
    x[7][x[3] == 0] = 0
    if truth is None:
        return x, source, dest
    y = np.zeros((3, *NATIVE_PATCH_SIZE), dtype=np.uint8)
    y[(slice(None), *dest)] = truth[source][None] > np.arange(3)[:, None, None, None]
    valid = np.zeros(NATIVE_PATCH_SIZE, dtype=bool)
    valid[dest] = brain[source] | (truth[source] > 0)
    return x, y, valid


def loss(logits, labels, original, valid):
    weights = valid * (1. + 1.5 * (labels != original))
    cross_entropy = (F.binary_cross_entropy_with_logits(logits, labels, reduction='none') * weights).sum() / weights.sum()
    probability = logits.sigmoid() * valid
    overlap = (probability * labels).sum(dim=(2, 3, 4))
    total = (probability + labels).sum(dim=(2, 3, 4))
    return cross_entropy + .7 * (1. - ((2 * overlap + 1.) / (total + 1.)).mean())


def fit(args):
    if not 0. <= args.prior_dropout <= 1.:
        raise ValueError('Prior dropout must lie in [0, 1]')
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    if world > 1:
        torch.distributed.init_process_group(backend='nccl')
    torch.manual_seed(SEED)
    widths = (16, 32, 64) if args.wide else (12, 20, 32)
    raw_model = AppearanceProposal(widths).to(device, memory_format=torch.channels_last_3d)
    model = DistributedDataParallel(raw_model, device_ids=[rank]) if world > 1 else raw_model
    train = rows(args.cohort, 'train', truth=True)
    if len(train) % world:
        raise ValueError('Use a worker count that partitions the frozen training cohort evenly')
    result = Path(args.output)
    result.mkdir(parents=True, exist_ok=True)
    path = result / 'checkpoint.pth'
    provenance = {'cohort': file_identity(args.cohort), 'source': file_identity(__file__)['sha256'],
                  'patch': list(NATIVE_PATCH_SIZE), 'epochs': args.epochs, 'workers': world,
                  'prior_dropout': args.prior_dropout, 'widths': list(widths)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.015)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00012)
    start = 0
    if path.exists():
        previous = torch.load(path, weights_only=True, map_location=device)
        if previous['binding'] != provenance:
            raise ValueError('The existing checkpoint uses another model or training schedule')
        raw_model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scaler.load_state_dict(previous['scaler'])
        schedule.load_state_dict(previous['schedule'])
        start = previous['epoch']
    try:
        for epoch in range(start, args.epochs):
            tic = monotonic()
            plan = np.random.default_rng(SEED + epoch).permutation(len(train))[rank::world]
            rng = np.random.default_rng(SEED + 31 * epoch + rank)
            costs = []
            for index in plan:
                raw, prior, brain, truth, left_right = read_case(args.cache, train[index], 'train')
                xyz = training_start(prior, brain, rng)
                use_prior = args.prior_dropout < 1. and (args.prior_dropout == 0. or rng.random() >= args.prior_dropout)
                x, target, valid = patch_input(raw, prior, brain, xyz, left_right, rng=rng, truth=truth,
                                               use_prior=use_prior)
                images = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
                labels = torch.from_numpy(target[None]).to(device=device, dtype=torch.float32)
                foreground = torch.from_numpy(valid[None, None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(images)
                value = loss(logits.float(), labels, images[:, 4:7], foreground)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite native-resolution training loss')
                scaler.scale(value).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 6.)
                scaler.step(optimizer)
                scaler.update()
                costs.append(float(value.detach()))
            schedule.step()
            totals = torch.tensor([sum(costs), len(costs), torch.cuda.max_memory_reserved(device)], device=device)
            if world > 1:
                torch.distributed.all_reduce(totals[:2])
                torch.distributed.all_reduce(totals[2:], op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                tmp = path.with_suffix('.partial')
                torch.save({'binding': provenance, 'epoch': epoch + 1, 'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                            'schedule': schedule.state_dict()}, tmp)
                tmp.replace(path)
                status = {'epoch': epoch + 1, 'n_train': int(totals[1]), 'loss': float(totals[0] / totals[1]),
                          'peak_reserved_gib': float(totals[2]) / 2**30, 'seconds': monotonic() - tic}
                write_json(result / 'progress.json', status)
                print(status, flush=True)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    source = file_identity(args.model)
    version = torch.load(args.model, map_location=device, weights_only=True)
    if version['binding']['cohort'] != file_identity(args.cohort) or version['epoch'] != version['binding']['epochs']:
        raise ValueError('Inference requires the completed model from its frozen train100')
    use_prior = not (args.no_prior or version['binding'].get('prior_dropout') == 1.)
    model = AppearanceProposal(version['binding'].get('widths', (12, 20, 32))).to(
        device, memory_format=torch.channels_last_3d)
    model.load_state_dict(version['model'])
    model.eval()
    t = np.maximum(1.e-3, np.hanning(SIDE)).astype(np.float32)
    weights = np.maximum(.002, t[:, None, None] * t[None, :, None] * t[None, None, :])
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    for row in selected:
        folder = Path(args.output) / row['case_id']
        index = folder / 'proposal.json'
        binding = {'row': row, 'model': source, 'patch': list(NATIVE_PATCH_SIZE)}
        if not use_prior:
            binding['prior'] = 'none'
        if index.is_file():
            result = read_json(index)
            if result['binding'] != binding or result['gt_used'] is not False:
                raise ValueError('Existing proposal came from different MRI or weights')
            check_file(result['arrays'])
            continue
        raw, old, brain, _, left_right = read_case(args.cache, row, 'validation')
        valid_box = residual._window(brain, padding=0)
        starts = []
        for bbox in valid_box:
            if bbox.stop - bbox.start <= SIDE:
                starts.append(((bbox.start + bbox.stop - SIDE) // 2,))
            else:
                starts.append((bbox.start, bbox.stop - SIDE))
        probabilities = np.zeros((3, *old.shape), dtype=np.float32)
        covered = np.zeros(old.shape, dtype=np.float32)
        for initial in product(*starts):
            image, source_slice, dest = patch_input(raw, old, brain, initial, left_right,
                                                    use_prior=use_prior)
            x = torch.from_numpy(image[None]).to(device, memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                p = model(x)[0].sigmoid().cpu().numpy()
            probabilities[(slice(None), *source_slice)] += p[(slice(None), *dest)] * weights[dest]
            covered[source_slice] += weights[dest]
        if np.any(brain & (covered == 0)):
            raise ValueError('Tiled native proposal missed part of the MRI foreground')
        probabilities /= np.maximum(covered, 1.e-6)
        probabilities[:, ~brain] = 0
        np.maximum(probabilities[1], probabilities[2], out=probabilities[1])
        np.maximum(probabilities[0], probabilities[1], out=probabilities[0])
        folder.mkdir(parents=True, exist_ok=True)
        array = folder / 'proposal.npz'
        dense._atomic_npz(array, probabilities=probabilities.astype(np.float16))
        write_json(index, {'binding': binding, 'arrays': file_identity(array), 'gt_used': False})
        print(f'{row["case_id"]}: stitched native proposals using {np.count_nonzero(probabilities[0] > .5)} WT voxels', flush=True)


def score(args):
    ref = {r['case_id']: r for r in manifest(args.truth, allowed=('validation',))['cases']}
    records, coverage = {}, []
    for row in rows(args.cohort, 'validation'):
        key = row['case_id']
        proposal = read_json(Path(args.proposals) / key / 'proposal.json')
        if proposal['binding']['row'] != row or proposal['gt_used'] is not False:
            raise ValueError('Scoring requires a frozen, label-free proposal')
        check_file(proposal['arrays'])
        old, truth, image, volume = scoring._image_pair(row, ref[key])
        base = np.stack([old > h for h in range(3)])
        with np.load(proposal['arrays']['path'], allow_pickle=False) as data:
            values = data['probabilities']
        if values.shape != base.shape:
            raise ValueError('Proposal has altered native MRI geometry')
        estimates = {'raw_p50': values > .5,
                     'confident_residual': np.where(base, values >= .2, values > .8),
                     'strong_additions': base | (values > .8)}
        for label, masks in estimates.items():
            region = residual._window((old > 0) | (truth > 0) | masks[0], padding=0)
            updated = residual.merge_heads(masks[(slice(None), *region)])
            records.setdefault(label, []).extend({'case_id': key, **q} for q in
                scoring._measure(old[region], updated, truth[region], volume))
        for h, head in enumerate(dense.HEADS):
            possible = np.where(base[h], values[h] < .5, values[h] > .5)
            coverage.append({'case_id': key, 'region': head, **residual._coverage(base[h], truth > h, possible)})
    summary = {method: scoring.summary(result) for method, result in records.items()}
    write_json(args.output, {'coverage': coverage, 'comparison': summary, 'per_case': records})
    for name, info in summary.items():
        print(name, {k: info[k]['gain_pp'] for k in dense.HEADS}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('fit')
    for field in ('cohort', 'cache', 'output'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--prior-dropout', type=float, default=0.)
    p.add_argument('--wide', action='store_true')
    p = sub.add_parser('predict')
    for field in ('cohort', 'cache', 'model', 'output'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--no-prior', action='store_true')
    p = sub.add_parser('score')
    for field in ('cohort', 'truth', 'proposals', 'output'):
        p.add_argument('--' + field, required=True)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
