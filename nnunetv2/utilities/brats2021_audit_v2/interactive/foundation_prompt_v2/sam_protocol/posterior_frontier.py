"""Fine-tune the native expert correction to expose high-confidence false negatives."""
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
from . import appearance, dense, posterior_fields, posterior_fusion, residual
from .cohorts import rows

BASE_CAP = 6.
EXTRA_ADDITION_CAP = 3.
HARD_ANCHOR = -4.5
SEED = 50707


def anchor_logits(inputs):
    anchor = .5 * inputs[:, 4:7].float()
    anchor.add_(inputs[:, 7:10], alpha=.25)
    anchor.add_(inputs[:, 10:13], alpha=.25)
    return anchor.mul_(posterior_fields.EXTENT)


def expanded_prediction(model, inputs):
    anchor = anchor_logits(inputs)
    change = torch.tanh(model(inputs).float() / 3.)
    extra = anchor.clone().sub_(HARD_ANCHOR).mul_(-2.).sigmoid_().mul_(EXTRA_ADDITION_CAP)
    scores = change.mul(BASE_CAP).addcmul_(extra, change.clamp_min(0.)).add_(anchor)
    return scores, anchor[:, 0]


def balanced_loss(scores, anchor, target):
    objective = posterior_fusion.loss(scores, target)
    hard_false_negative = target[:, 0].bool() & (target[:, 4] == 0) & (anchor < HARD_ANCHOR)
    missed = scores[:, 0][hard_false_negative]
    if missed.numel():
        objective = objective + .08 * F.softplus(-missed).mean()
    return objective


def simulate_expert_miss(images, target, rng):
    """Hide one peripheral WT portion in expert inputs while retaining its MRI."""
    edema = (target[0] != 0) & (target[1] == 0)
    points = np.argwhere(edema[::6, ::6, ::6] != 0)
    if not len(points):
        return
    center = 6 * points[rng.integers(len(points))]
    radius = rng.integers(12, 26, 3)
    lo = np.maximum(0, center - radius)
    hi = np.minimum(np.asarray(edema.shape), center + radius + 1)
    window = tuple(slice(a, b) for a, b in zip(lo, hi))
    grid = np.ogrid[window]
    local = sum(((coordinate - midpoint) / breadth) ** 2
                for coordinate, midpoint, breadth in zip(grid, center, radius))
    strength = np.clip((1. - local) * 4., 0., 1.).astype(np.float16)
    logits = images[(slice(4, 13), *window)]
    logits -= (logits + 1.) * strength
    old = target[(4, *window)]
    old[local < .75] = 0


def _training_binding(args, world):
    permission = torch.load(args.initial_model, map_location='cpu', weights_only=True)
    foundation = permission['binding']
    cohort = file_identity(args.cohort)
    if (permission['epoch'] != foundation['epochs'] or foundation['cohort'] != cohort
            or foundation['patch'] != list(appearance.NATIVE_PATCH_SIZE)
            or foundation['source'] != file_identity(posterior_fusion.__file__)):
        raise ValueError('Expanded residual requires the completed, matching bounded model')
    for key in ('source', 'network', 'field_source'):
        check_file(foundation[key])
    binding = {'cohort': cohort, 'source': file_identity(__file__),
               'starting_model': file_identity(args.initial_model), 'patch': foundation['patch'],
               'epochs': args.epochs, 'world': world}
    return binding, permission['model']


def fit(args):
    training = rows(args.cohort, 'train', truth=True)
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if len(training) % world:
        raise ValueError('All training patients must partition evenly across the requested GPUs')
    binding, weights = _training_binding(args, world)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    if world > 1:
        torch.distributed.init_process_group('nccl')
    model = posterior_fusion.network(device)
    model.load_state_dict(weights)
    wrapped = DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=.00055, weight_decay=.015)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.00008)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / 'checkpoint.pth'
    start = 0
    if checkpoint.is_file():
        state = torch.load(checkpoint, weights_only=True, map_location=device)
        if state['binding'] != binding:
            raise ValueError('Checkpoint belongs to another training protocol')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start = state['epoch']
    try:
        for epoch in range(start, args.epochs):
            tick = monotonic()
            order = np.random.default_rng(SEED + epoch).permutation(len(training))[rank::world]
            rng = np.random.default_rng(SEED + 59 * epoch + rank)
            values = []
            for i in order:
                case = posterior_fusion.read_inputs(args, training[i], 'train')
                _, old, brain, truth, _, fields = case
                point = appearance.training_start(old, brain, rng)
                if rng.random() < .35:
                    missed = (old[::4, ::4, ::4] == 0) & (truth[::4, ::4, ::4] > 0)
                    low_resolution = fields[:, ::4, ::4, ::4]
                    expert_anchor = .5 * low_resolution[0] + .25 * (low_resolution[3] + low_resolution[6])
                    weak = expert_anchor < HARD_ANCHOR / posterior_fields.EXTENT
                    coordinates = np.argwhere(missed & weak)
                    if len(coordinates):
                        point = coordinates[rng.integers(len(coordinates))] * 4 - appearance.SIDE // 2
                x, y = posterior_fusion.patch(case, point, rng)
                if rng.random() < .25:
                    simulate_expert_miss(x, y, rng)
                inputs = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
                target = torch.from_numpy(y[None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    scores, anchor = expanded_prediction(wrapped, inputs)
                loss = balanced_loss(scores, anchor, target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite expanded residual objective')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 6.)
                scaler.step(optimizer)
                scaler.update()
                values.append(float(loss.detach()))
            scheduler.step()
            pair = torch.tensor([sum(values), len(values)], device=device)
            peak = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
            if world > 1:
                torch.distributed.all_reduce(pair)
                torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
            if rank == 0:
                temp = checkpoint.with_suffix('.partial')
                torch.save({'binding': binding, 'epoch': epoch + 1, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict()}, temp)
                temp.replace(checkpoint)
                progress = {'epoch': epoch + 1, 'n_train': int(pair[1]), 'loss': float(pair[0] / pair[1]),
                            'peak_reserved_gib': float(peak), 'seconds': monotonic() - tick}
                write_json(root / 'progress.json', progress)
                print(progress, flush=True)
    finally:
        if world > 1:
            torch.distributed.destroy_process_group()


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(3)
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    identity = file_identity(args.model)
    binding = checkpoint['binding']
    if (checkpoint['epoch'] != binding['epochs'] or binding['cohort'] != file_identity(args.cohort)
            or binding['source'] != file_identity(__file__)):
        raise ValueError('Generate validation prompts with the finished, matching model')
    check_file(binding['starting_model'])
    model = posterior_fusion.network(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    size = appearance.SIDE
    edge = np.hanning(size).astype(np.float32)
    blend = np.maximum(.002, edge[:, None, None] * edge[None, :, None] * edge[None, None, :])
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    for row in selected:
        folder = Path(args.output) / row['case_id']
        receipt = folder / 'proposal.json'
        contract = {'row': row, 'model': identity, 'patch': list(appearance.NATIVE_PATCH_SIZE)}
        if receipt.is_file():
            previous = read_json(receipt)
            if previous['binding'] != contract or previous['gt_used'] is not False:
                raise ValueError('Existing inference has incompatible model or input')
            check_file(previous['arrays'])
            continue
        value = posterior_fusion.read_inputs(args, row, 'validation')
        brain = value[2]
        starts = [((sl.start + sl.stop - size) // 2,) if sl.stop - sl.start <= size
                  else (sl.start, sl.stop - size) for sl in residual._window(brain, padding=0)]
        field = np.zeros((3, *brain.shape), dtype=np.float32)
        weight = np.zeros(brain.shape, dtype=np.float32)
        for start in product(*starts):
            x, src, dst = posterior_fusion.patch(value, start)
            inputs = torch.from_numpy(x[None]).to(device, memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=torch.float16):
                probabilities = expanded_prediction(model, inputs)[0][0].sigmoid().cpu().numpy()
            field[(slice(None), *src)] += probabilities[(slice(None), *dst)] * blend[dst]
            weight[src] += blend[dst]
        if np.any(brain & (weight == 0)):
            raise ValueError('Native tiling did not cover the full MRI')
        field /= np.maximum(weight, 1.e-6)
        field[:, ~brain] = 0.
        np.maximum(field[1], field[2], out=field[1])
        np.maximum(field[0], field[1], out=field[0])
        folder.mkdir(parents=True, exist_ok=True)
        array = folder / 'proposal.npz'
        dense._atomic_npz(array, probabilities=field.astype(np.float16))
        write_json(receipt, {'binding': contract, 'arrays': file_identity(array), 'gt_used': False})
        print(row['case_id'], 'expanded residual on native MRI grid', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    f = sub.add_parser('fit')
    p = sub.add_parser('predict')
    for stage in (f, p):
        for field in ('cohort', 'cache', 'fields', 'output'):
            stage.add_argument('--' + field, required=True)
    f.add_argument('--initial-model', required=True)
    f.add_argument('--epochs', type=int, default=6)
    p.add_argument('--model', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    globals()[args.stage](args)
