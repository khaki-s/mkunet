"""Train the SAM3 decoder on all available training combinations with DDP."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from hashlib import sha256
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from ..common import file_identity, read_json, write_json
from .cache_features import packet_binding
from .dataset import PROJECT, CachedDataset, catalog, split_manifest
from .model import SCHEMA, autocast, implementation, loss, new_model, prepare_prompts, validate_encoder


class PatientSampler(Sampler):
    """Shuffle patient groups, keeping adjacent reads local and rank lengths identical."""

    def __init__(self, records, rank, world, seed):
        patients = defaultdict(list)
        for i, (patient, *_) in enumerate(records):
            patients[patient].append(i)
        self.groups = list(patients.values())
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch = 0
        self.per_rank = math.ceil(len(records) / world)

    def __len__(self):
        return self.per_rank

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        grouped = [rng.permutation(self.groups[i]) for i in rng.permutation(len(self.groups))]
        order = np.concatenate(grouped)
        padded = np.resize(order, self.per_rank * self.world)
        start = self.rank * self.per_rank
        return iter(padded[start:start + self.per_rank].tolist())


class ModelForward(torch.nn.Module):
    """Use normal module forward so DDP installs synchronization hooks."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, features, prompts, shape):
        return self.decoder.forward_cached(features, prompts, shape)


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def train(args):
    if args.epochs < 1 or args.accumulate < 1 or args.workers < 0:
        raise ValueError("Invalid training counts")
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --cpu is available for diagnostic runs")
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
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    root = args.output.resolve()
    encoder = read_json(root / "encoder.json")
    if rank == 0:
        validate_encoder(encoder)
    cases = catalog(args.project, "train", per_case=args.per_case, limit_cases=args.limit_cases)
    binding = packet_binding(encoder, "train", "all", args.max_slices)
    data = CachedDataset(root, cases, binding, split_manifest(args.project, "train", truth=True))
    sampler = PatientSampler(data.records, rank, world, args.seed)
    loader = DataLoader(data, batch_size=None, sampler=sampler, num_workers=args.workers,
                        persistent_workers=args.workers > 0)
    decoder = new_model(encoder, device, args.lora_rank)
    model = ModelForward(decoder)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None,
                                        find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=256.)
    output = root / "training"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint_last.pth"
    config = {
        "world_size": world, "seed": args.seed, "accumulate": args.accumulate, "lr": args.lr,
        "edit_weight": args.edit_weight, "keep_weight": args.keep_weight, "lora_rank": args.lora_rank,
        "per_case": args.per_case, "cases": list(cases), "packet_binding": binding,
        "training_pairs": sum(len(p) for p in cases.values()), "samples_per_mask": dict(data.counts),
        "sample_count": len(data), "train_baseline": "fold_0 in-sample",
        "baseline_model": next(iter(cases.values()))[0].baseline_model,
    }
    code = implementation()
    history = []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["schema"], state["encoder"], state["implementation"], state["training"]) != (SCHEMA, encoder, code, config):
            raise ValueError("Existing checkpoint came from a different training configuration")
        decoder.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        del state
    if rank == 0:
        print(f"Training {len(data)} samples across {len(cases)} patients and {world} ranks", flush=True)
    for epoch in range(len(history), args.epochs):
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stats = np.zeros(4, np.float64)
        for i, item in enumerate(loader):
            synchronize = (i + 1) % args.accumulate == 0 or i + 1 == len(loader)
            group_size = min(args.accumulate, len(loader) - i // args.accumulate * args.accumulate)
            context = model.no_sync() if world > 1 and not synchronize else nullcontext()
            labels = item["baseline"]
            head = int(item["head"])
            prompts = prepare_prompts(labels, item["probability"], head, decoder.mask_input_size)
            with context:
                with autocast(device):
                    logits = model(item["features"], prompts, tuple(labels.shape))
                batch_loss = loss(logits, item["target"], (labels > head)[None, None], encoder["sam_source"],
                                  args.edit_weight, args.keep_weight)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(batch_loss / group_size).backward()
            stats[:2] += [float(batch_loss.detach()), 1]
            if synchronize:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < scale
                stats[2:] += [not skipped, skipped]
                optimizer.zero_grad(set_to_none=True)
            if rank == 0 and (i == 0 or (i + 1) % 100 == 0):
                print(f"epoch {epoch + 1}/{args.epochs} rank step {i + 1}/{len(loader)} "
                      f"loss={stats[0] / stats[1]:.6f} updates={int(stats[2])} skipped={int(stats[3])}", flush=True)
        totals = torch.as_tensor(stats, device=device)
        if world > 1:
            dist.all_reduce(totals)
        ls, steps, updates, skips = (float(x) for x in totals)
        if updates == 0:
            raise RuntimeError("No successful optimizer steps in the epoch")
        history.append({"epoch": epoch + 1, "loss": ls / steps, "samples": int(steps),
                        "optimizer_steps_per_rank": int(updates) // world,
                        "overflows_across_ranks": int(skips)})
        if rank == 0:
            state = {"schema": SCHEMA, "encoder": encoder, "implementation": code,
                     "configuration": decoder.configuration(), "training": config, "history": history,
                     "state_dict": {n: p.detach().cpu() for n, p in decoder.state_dict().items()},
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
            atomic_save(state, checkpoint)
            write_json(output / "history.json", history)
        if world > 1:
            dist.barrier()
    checksum = sha256()
    for name, value in sorted(decoder.state_dict().items()):
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().contiguous().numpy().tobytes())
    hashes = [checksum.hexdigest()]
    if world > 1:
        hashes = [None] * world
        dist.all_gather_object(hashes, checksum.hexdigest())
        if len(set(hashes)) != 1:
            raise ValueError("DDP replicas diverged")
    write_json(output / f"resources_rank{rank}.json", {
        "rank": rank, "world": world, "device": str(device), "replicas_identical": len(set(hashes)) == 1,
        "model_sha256": hashes[rank],
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.,
    })
    if rank == 0:
        write_json(output / "completed.json", {"checkpoint": file_identity(checkpoint), "training": config,
                                               "epochs": len(history)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-slices", type=int, default=3)
    p.add_argument("--per-case", type=int, default=6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--edit-weight", type=float, default=1.)
    p.add_argument("--keep-weight", type=float, default=.5)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
