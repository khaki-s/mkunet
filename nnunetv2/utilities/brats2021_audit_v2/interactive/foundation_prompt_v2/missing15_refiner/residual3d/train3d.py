"""DDP training of the residual 3D corrector on cached windows.

Weighted four-class cross-entropy (ROI errors up-weighted, ROI-correct protected,
background down-weighted) plus a SoftDice on the nested region probabilities. EMA
weights are exported; no validation set participates in model selection.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from ...common import file_identity, read_json, write_json
from ..dataset import catalog
from . import data3d as D
from .model3d import SCHEMA, atomic_save, build_model, implementation, regions_from_softmax_t


def region_targets(target):
    """Binary region targets (WT, TC, ET) from internal labels, shape (B, 3, ...)."""
    return torch.stack([target > 0, target >= 2, target >= 3], dim=1).float()


def voxel_weights(base, roi, target, error_weight, keep_weight, outside_weight):
    error = roi & (base != target)
    correct = roi & ~error
    weight = torch.full_like(base, outside_weight, dtype=torch.float32)
    weight[correct] = keep_weight
    weight[error] = keep_weight + error_weight
    return weight


def soft_dice(region_prob, region_target, eps=1e-5):
    dims = tuple(range(2, region_prob.ndim))
    intersection = (region_prob * region_target).sum(dims)
    denom = region_prob.sum(dims) + region_target.sum(dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_loss(logits, target, base, roi, weights):
    ce = F.cross_entropy(logits, target, reduction="none")
    weighted = (ce * weights).sum() / weights.sum().clamp_min(1.0)
    region_prob = regions_from_softmax_t(torch.softmax(logits, dim=1))
    dice = soft_dice(region_prob, region_targets(target))
    return weighted + 0.5 * dice, float(weighted.detach()), float(dice.detach())


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[key].copy_(value.detach())

    def state(self):
        return {k: v.cpu() for k, v in self.shadow.items()}


def lr_at(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def train(args):
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --cpu for diagnostics")
    device = torch.device("cpu" if args.cpu else f"cuda:{local}")
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("gloo" if args.cpu else "nccl",
                                **({"device_id": device} if not args.cpu else {}))
    try:
        run(args, device, local, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()


def run(args, device, local, rank, world):
    torch.manual_seed(args.seed + rank)
    torch.set_num_threads(args.cpu_threads)
    root = args.output.resolve()
    prepare = read_json(root / "prepare.json")
    binding = D.packet_binding("train", prepare["prepare_sha256"])
    cases = catalog(args.project, "train", per_case=args.per_case, limit_cases=args.limit_cases)
    # --steps counts optimizer steps per rank (plan: 30k per GPU); each rank draws its own patches.
    data = D.WindowPatchDataset(root, cases, binding, patch=args.patch,
                                steps=(args.steps + 1) * args.batch * args.accumulate,
                                seed=args.seed * 1000 + rank, augment=not args.no_augment)
    loader = DataLoader(data, batch_size=args.batch, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0, pin_memory=device.type == "cuda")
    model = build_model(device=device)
    wrapped = (DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None)
               if world > 1 else model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema_decay)
    output = root / "training3d"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint_last.pth"
    config = {"world_size": world, "seed": args.seed, "steps": args.steps, "batch": args.batch,
              "accumulate": args.accumulate, "lr": args.lr, "warmup": args.warmup,
              "patch": args.patch, "ema_decay": args.ema_decay, "per_case": args.per_case,
              "cases": list(cases), "binding": binding, "train_baseline": "fold_0 in-sample",
              "baseline_model": next(iter(cases.values()))[0].baseline_model}
    code = implementation()
    start_step, history = 0, []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["schema"], state["implementation"], state["training"]) != (SCHEMA, code, config):
            raise ValueError("Existing checkpoint came from a different configuration")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        ema.shadow = {k: v.clone() for k, v in state["ema"].items()}
        start_step, history = state["step"], state["history"]
    if rank == 0:
        print(f"Training {args.steps} steps, {len(data.records)} windows, world {world}", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    stats = np.zeros(3, np.float64)
    for step in range(start_step, args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args.warmup, args.steps, args.lr)
        for micro in range(args.accumulate):
            item = next(iterator)
            features = item["features"].to(device, non_blocking=True)
            prior = item["prior"].to(device, non_blocking=True)
            base = item["base"].to(device, non_blocking=True)
            roi = item["roi"].to(device, non_blocking=True)
            target = item["target"].to(device, non_blocking=True)
            weights = voxel_weights(base, roi, target, args.error_weight, args.keep_weight,
                                    args.outside_weight)
            with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                logits = wrapped(features, prior)
                loss, ce, dice = compute_loss(logits, target, base, roi, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / args.accumulate).backward()
            stats += [loss.item(), ce, dice]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        if rank == 0 and ((step + 1) % args.log_every == 0 or step == start_step):
            n = args.log_every * args.accumulate
            print(f"step {step + 1}/{args.steps} lr={optimizer.param_groups[0]['lr']:.2e} "
                  f"loss={stats[0] / n:.4f} ce={stats[1] / n:.4f} dice={stats[2] / n:.4f}", flush=True)
            stats[:] = 0
        if rank == 0 and (step + 1) % args.save_every == 0:
            history.append({"step": step + 1, "lr": optimizer.param_groups[0]["lr"]})
            atomic_save({"schema": SCHEMA, "implementation": code, "training": config,
                         "configuration": model.configuration(), "step": step + 1, "history": history,
                         "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                         "ema": ema.state(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict()}, checkpoint)
        if world > 1 and (step + 1) % args.save_every == 0:
            dist.barrier()
    finalize(args, model, ema, config, code, device, rank, world, output)


def finalize(args, model, ema, config, code, device, rank, world, output):
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 2**30
    else:
        peak = 0.0
    write_json(output / f"resources_rank{rank}.json", {"rank": rank, "world": world,
               "device": str(device), "peak_allocated_gib": peak})
    if rank == 0:
        ema_checkpoint = output / "checkpoint_ema.pth"
        atomic_save({"schema": SCHEMA, "implementation": code, "training": config,
                     "configuration": model.configuration(), "state_dict": ema.state(),
                     "weights": "ema"}, ema_checkpoint)
        write_json(output / "completed.json", {"checkpoint": file_identity(ema_checkpoint),
                   "training": config, "steps": args.steps,
                   "baseline_model": config["baseline_model"]})
        print(f"Finished; EMA checkpoint {ema_checkpoint}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=D.PROJECT)
    p.add_argument("--output", type=Path, default=D.DEFAULT_OUTPUT)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--error-weight", type=float, default=2.0)
    p.add_argument("--keep-weight", type=float, default=1.0)
    p.add_argument("--outside-weight", type=float, default=0.2)
    p.add_argument("--per-case", type=int, default=6)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
