"""Sliding-window inference of the residual corrector over each cached window.

Produces, per (case, combination), the three nested region score volumes r_WT,
r_TC, r_ET on the window grid (float16), plus the baseline region probabilities
q so the policy layer can blend arm P and arm R on identical geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...common import check_file, file_identity, read_json, write_json
from ..dataset import catalog
from . import data3d as D
from .model3d import regions_from_softmax_t, restore


def gaussian_weight(patch, sigma_scale=0.125, device="cpu"):
    center = (patch - 1) / 2
    axis = torch.arange(patch, dtype=torch.float32, device=device) - center
    g1 = torch.exp(-(axis ** 2) / (2 * (patch * sigma_scale) ** 2))
    weight = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    return (weight / weight.max()).clamp_min(1e-3)


def sliding_window(model, features, prior, patch, step, device, mirror=True):
    """Gaussian-weighted sliding window with optional x-axis mirror TTA; returns r (3, X, Y, Z)."""
    shape = features.shape[1:]
    pad = [max(patch - s, 0) for s in shape]
    features = np.pad(features, [(0, 0)] + [(0, p) for p in pad])
    prior = np.pad(prior, [(0, 0)] + [(0, p) for p in pad])
    padded = features.shape[1:]
    logits_sum = torch.zeros((4, *padded), device=device)
    weight_sum = torch.zeros((1, *padded), device=device)
    window = gaussian_weight(patch, device=device)
    starts = [sorted({*range(0, max(dim - patch, 0) + 1, step), max(dim - patch, 0)})
              for dim in padded]
    feat_t = torch.from_numpy(features).to(device)
    prior_t = torch.from_numpy(prior).to(device)
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                sl = (slice(None), slice(x, x + patch), slice(y, y + patch), slice(z, z + patch))
                f = feat_t[sl][None]
                p = prior_t[sl][None]
                with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
                    out = model(f, p)
                    if mirror:
                        out = out + torch.flip(model(torch.flip(f, dims=[2]),
                                                     torch.flip(p, dims=[2])), dims=[2])
                logits = out.float()[0] / (2 if mirror else 1)
                logits_sum[(slice(None), *sl[1:])] += logits * window
                weight_sum[(slice(None), *sl[1:])] += window
    posterior = torch.softmax(logits_sum / weight_sum.clamp_min(1e-6), dim=0)
    region = regions_from_softmax_t(posterior[None])[0]
    crop = (slice(None), *(slice(0, s) for s in shape))
    return region[crop].cpu().numpy().astype(np.float16)


def infer_patient(root, split, case_id, packet, model, device, checkpoint, args):
    output = Path(root) / "r3d_scores" / split / f"{case_id}.json"
    binding = {"checkpoint": checkpoint, "window_packet": file_identity(D.packet_path(root, split, case_id)),
               "step": args.step, "mirror": not args.no_mirror, "gt_used": False}
    if output.exists():
        previous = read_json(output)
        if previous["binding"] != binding:
            raise ValueError(f"{output}: scores belong to a different method")
        for entry in previous["pairs"]:
            check_file(entry["scores"])
        return
    pairs = []
    for entry in packet["pairs"]:
        image = np.asarray(D.load_window(entry, "image"), dtype=np.float32)
        prob = np.asarray(D.load_window(entry, "prob"), dtype=np.float32)
        base = np.asarray(D.load_window(entry, "base"), dtype=np.int64)
        features = D.assemble_input(image, prob, base, entry["availability"])
        prior = D.prior_logits(prob)
        region = sliding_window(model, features, prior, args.patch, args.step, device,
                                mirror=not args.no_mirror)
        path = output.parent / case_id / f"mask_{entry['code']:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, region=region,
                     prior_region=D.nested_projection_region(prob).astype(np.float16))
        temporary.replace(path)
        pairs.append({"code": entry["code"], "baseline": entry["baseline"],
                      "probability": entry["probability"], "box": entry["box"],
                      "shape": entry["shape"], "scores": file_identity(path)})
    write_json(output, {"binding": binding, "pairs": pairs})


def run(args):
    root = args.output.resolve()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu" if args.cpu else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    completed = read_json(root / "training3d" / "completed.json")
    checkpoint = completed["checkpoint"]
    model, _ = restore(checkpoint, device)
    prepare = read_json(root / "prepare.json")
    binding = D.packet_binding(args.split, prepare["prepare_sha256"])
    codes = tuple(args.codes) if args.codes else tuple(range(1, 16))
    cases = catalog(args.project, args.split, codes=codes, limit_cases=args.limit_cases)
    if next(iter(cases.values()))[0].baseline_model != completed["baseline_model"]:
        raise ValueError("Inference and training baseline checkpoints differ")
    ids = sorted(cases)[args.shard::args.shards]
    with torch.inference_mode():
        for i, case_id in enumerate(ids, 1):
            packet = D.read_packet(D.packet_path(root, args.split, case_id), binding)
            expected = {(p.code, p.mask["sha256"]) for p in cases[case_id]}
            actual = {(e["code"], e["baseline"]["sha256"]) for e in packet["pairs"]}
            if expected != actual:
                raise ValueError(f"{case_id}: window cache uses different combinations or baseline")
            infer_patient(root, args.split, case_id, packet, model, device, checkpoint, args)
            print(f"{args.split} r3d inference {i}/{len(ids)} {case_id}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=D.PROJECT)
    p.add_argument("--output", type=Path, default=D.DEFAULT_OUTPUT)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--codes", type=int, nargs="+", default=[])
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--cpu", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
