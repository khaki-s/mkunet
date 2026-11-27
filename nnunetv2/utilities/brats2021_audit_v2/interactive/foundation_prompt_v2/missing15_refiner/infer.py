"""Run the trained decoder on cached native planes using deployment manifests."""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..common import check_file, file_identity, read_json, write_json
from ..sam_domain_data import read_features
from .cache_features import packet_binding, write_npz
from .dataset import PROJECT, catalog, packet_path, read_packet
from .model import autocast, prepare_prompts, restore


def score_path(output, split, case_id):
    return Path(output) / "sam_scores" / split / f"{case_id}.json"


@lru_cache(maxsize=8)
def _feature(path, sha256):
    return read_features({"path": path, "sha256": sha256})


def score_patient(root, split, patient, packet, network, device, checkpoint, active_heads=None):
    output = score_path(root, split, patient)
    binding = {"checkpoint": checkpoint, "packet": file_identity(packet_path(root, split, patient)),
               "heads": None if active_heads is None else {str(c): list(h) for c, h in active_heads.items()},
               "gt_used": False}
    if output.exists():
        previous = read_json(output)
        if previous["binding"] != binding:
            raise ValueError(f"{output}: scores belong to a different method")
        for group in previous["pairs"]:
            if group["records"]:
                check_file(group["scores"])
        return
    result = []
    for group in packet["pairs"]:
        heads = None if active_heads is None else active_heads[group["code"]]
        indices = [i for i, r in enumerate(group["records"]) if heads is None or r["head"] in heads]
        records = [group["records"][i] for i in indices]
        info = {"code": group["code"], "baseline": group["baseline"], "records": records}
        if records:
            check_file(group["slices"])
            scores = []
            with np.load(group["slices"]["path"], allow_pickle=False) as archive:
                masks, prior = archive["labels"], archive["probability"]
            for i, record in zip(indices, records):
                feature = _feature(record["feature"]["path"], record["feature"]["sha256"])
                prompt = prepare_prompts(masks[i], prior[i], record["head"], network.mask_input_size)
                with autocast(device):
                    logit = network.forward_cached(feature, prompt, masks[i].shape)
                scores.append(logit.float().sigmoid()[0, 0].cpu().numpy().astype(np.float16))
            path = output.parent / patient / f"mask_{group['code']:02d}.npz"
            write_npz(path, score=np.stack(scores), probability=prior[indices])
            info["scores"] = file_identity(path)
        result.append(info)
    write_json(output, {"binding": binding, "pairs": result})


def load_scores(root, split, patient, checkpoint):
    path = score_path(root, split, patient)
    value = read_json(path)
    if value["binding"]["gt_used"] is not False or value["binding"]["checkpoint"] != checkpoint:
        raise ValueError(f"{path}: invalid prediction receipt")
    check_file(value["binding"]["packet"])
    return {group["code"]: group for group in value["pairs"]}


def run(args):
    output = args.output.resolve()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid inference shard")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    encoder = read_json(output / "encoder.json")
    checkpoint = read_json(output / "training" / "completed.json")["checkpoint"]
    active = None
    if args.split == "test":
        from .select_val80 import load_frozen
        frozen = load_frozen(output)
        if (args.scope, args.max_slices, checkpoint) != (frozen["scope"], frozen["max_slices"], frozen["checkpoint"]):
            raise ValueError("Test inference settings differ from the frozen policy")
        active = {c: p.heads for c, p in frozen["policies"].items() if p.rules}
        if not active:
            return
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    binding = packet_binding(encoder, args.split, args.scope, args.max_slices)
    rows = catalog(args.project, args.split, codes=tuple(active) if active is not None else tuple(range(1, 16)),
                   limit_cases=args.limit_cases)
    network, metadata = restore(checkpoint, encoder, device)
    if args.split == "test" and args.limit_cases:
        raise ValueError("Frozen evaluation requires the full test cohort")
    if next(iter(rows.values()))[0].baseline_model != metadata["training"]["baseline_model"]:
        raise ValueError("Inference and training baseline checkpoints differ")
    subset = sorted(rows)[args.shard::args.shards]
    with torch.inference_mode():
        for i, patient in enumerate(subset, 1):
            packet = read_packet(packet_path(output, args.split, patient), binding)
            expected = {(p.code, p.mask["sha256"], p.probability["sha256"]) for p in rows[patient]}
            prepared = {(g["code"], g["baseline"]["sha256"], g["probability"]["sha256"]) for g in packet["pairs"]}
            if expected != prepared:
                raise ValueError(f"{patient}: cache uses different combinations or baseline")
            score_patient(output, args.split, patient, packet, network, device, checkpoint, active)
            print(f"{args.split} inference {i}/{len(subset)} {patient}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--scope", choices=("all", "priority"), default="priority")
    p.add_argument("--max-slices", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
