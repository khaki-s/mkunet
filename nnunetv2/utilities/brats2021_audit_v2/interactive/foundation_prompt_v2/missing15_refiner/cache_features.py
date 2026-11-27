"""Shard label-free SAM3 encoding by patient and cache the official FP16 features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..common import MODALITIES, check_file, file_identity, object_digest, read_json, same_grid, write_json
from ..sam_domain_data import FEATURE_KEYS, slice_rgb
from ..sam_domain_model import FrozenSamDomainEncoder, SAM_SOURCE, SAM_CHECKPOINT, source_identity
from ..sam_domain_pipeline import memory_preflight
from .dataset import (
    FORMAT_VERSION, PROJECT, catalog, channel, heads_for, load_mask, load_probability,
    packet_path, slice_schedule,
)


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encoder_contract(source, checkpoint):
    return {
        "sam_checkpoint": file_identity(checkpoint), "sam_source": str(Path(source).resolve()),
        "sources": source_identity(source), "input_size": 1008,
        "normalization": "volume_nonzero_percentile_1_99_gray_rgb_native_v1",
        "precision": "float16", "features": list(FEATURE_KEYS),
    }


def initialize(output, source, checkpoint):
    path = output / "encoder.json"
    value = encoder_contract(source, checkpoint)
    if path.exists() and read_json(path) != value:
        raise ValueError("Encoder configuration differs from the cache's encoder.json")
    if not path.exists():
        write_json(path, value)
    return value


def packet_binding(encoder, split, scope, max_slices):
    scheduler = file_identity(Path(__file__).with_name("dataset.py"))["sha256"]
    return {"format": FORMAT_VERSION, "split": split, "scope": scope, "max_slices": max_slices,
            "encoder_sha256": object_digest(encoder), "scheduler": scheduler, "gt_used": False}


def read_mri(identity, reference):
    check_file(identity)
    image = nib.load(identity["path"])
    same_grid(reference, image)
    volume = image.get_fdata(dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Non-finite image: {identity['path']}")
    nz = volume[volume != 0]
    lo, hi = np.percentile(nz, (1, 99)) if nz.size else (0., 1.)
    return volume, (float(lo), max(float(hi - lo), 1e-6))


def encode_patient(pairs, output, register, binding):
    case_id = pairs[0].case_id
    split = binding["split"]
    path = packet_path(output, split, case_id)
    plans = []
    feature_dir = output / "features" / split / case_id
    requirements = {}
    reference = None
    for pair in pairs:
        heads = heads_for(pair.code, binding["scope"])
        if not heads:
            plans.append({"code": pair.code, "baseline": pair.mask, "probability": pair.probability, "records": []})
            continue
        labels, ref = load_mask(pair.mask)
        if reference is None:
            reference = ref
        else:
            same_grid(reference, ref)
        probability = load_probability(pair, labels.shape)
        plans.append(build_pair_packet(pair, labels, probability, output, binding, requirements))
    features = {}
    for modality, needs in requirements.items():
        volume, ranges = None, None
        for z, identity in sorted(needs.items()):
            name = f"m{modality}_z{z:03d}_{identity['sha256'][:16]}"
            target = feature_dir / f"{name}.npz"
            receipt = feature_dir / f"{name}.json"
            if receipt.exists():
                value = read_json(receipt)
                if value["input"] != identity or value["file"]["path"] != str(target.resolve()):
                    raise ValueError(f"Wrong feature identity: {receipt}")
                check_file(value["file"])
            else:
                if volume is None:
                    volume, ranges = read_mri(identity, reference)
                rgb = slice_rgb(volume[None], 0, z, [ranges])
                encoded = register().encode(rgb)
                write_npz(target, **{k: v.numpy() for k, v in encoded.items()})
                value = {"input": identity, "file": file_identity(target)}
                write_json(receipt, value)
            features[(modality, z)] = value["file"]
    for group in plans:
        for record in group["records"]:
            record["feature"] = features[(record["modality"], record["z"])]
    write_json(path, {"binding": binding, "pairs": plans})
    return len(features), sum(len(p["records"]) for p in plans)


def build_pair_packet(pair, labels, probability, output, binding, requirements):
    heads = heads_for(pair.code, binding["scope"])
    schedule = slice_schedule(labels, probability, heads, binding["max_slices"], joint=True)
    records = []
    for h, z in schedule:
        modality = channel(h, pair.code)
        needs = requirements.setdefault(modality, {})
        identity = pair.modalities[MODALITIES[modality]]
        if z in needs and needs[z] != identity:
            raise ValueError("The same patient and modality have different source files")
        needs[z] = identity
        records.append({"head": h, "z": z, "modality": modality})
    packet = {"code": pair.code, "baseline": pair.mask, "probability": pair.probability, "records": records}
    if records:
        path = output / "slices" / binding["split"] / pair.case_id / f"mask_{pair.code:02d}.npz"
        write_npz(path, labels=np.stack([labels[:, :, z] for _, z in schedule]),
                  probability=np.stack([probability[h, :, :, z] for h, z in schedule]).astype(np.float16))
        packet["slices"] = file_identity(path)
    return packet


def run(args):
    output = args.output.resolve()
    if args.initialize:
        initialize(output, args.source, args.checkpoint)
        return
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.max_slices < 0:
        raise ValueError("Invalid shard or slice budget")
    torch.set_num_threads(args.cpu_threads)
    contract = read_json(output / "encoder.json")
    if contract["sam_checkpoint"]["path"] != str(Path(args.checkpoint).resolve()) or contract["sam_source"] != str(Path(args.source).resolve()):
        raise ValueError("Encoder paths disagree with encoder.json")
    binding = packet_binding(contract, args.split, args.scope, args.max_slices)
    all_pairs = catalog(args.project, args.split, per_case=args.per_case, codes=args.codes,
                        limit_cases=args.limit_cases)
    encoder = None

    def register():
        nonlocal encoder
        if encoder is None:
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; launch on a CUDA host")
            memory_preflight(args.device, 8.)
            encoder = FrozenSamDomainEncoder(args.source, args.checkpoint, args.device)
        return encoder

    ids = sorted(all_pairs)[args.shard::args.shards]
    for index, case_id in enumerate(ids, 1):
        path = packet_path(output, args.split, case_id)
        if path.exists():
            packet = read_json(path)
            expected = {(p.code, p.mask["sha256"], p.probability["sha256"]) for p in all_pairs[case_id]}
            actual = {(g["code"], g["baseline"]["sha256"], g["probability"]["sha256"]) for g in packet["pairs"]}
            if packet["binding"] != binding or actual != expected:
                raise ValueError(f"{path}: cached patient is from another run")
            continue
        encodes, samples = encode_patient(all_pairs[case_id], output, register, binding)
        print(f"{args.split} {index}/{len(ids)} {case_id}: {encodes} features, {samples} samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=SAM_SOURCE)
    parser.add_argument("--checkpoint", default=SAM_CHECKPOINT)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scope", choices=("all", "priority"), default="all")
    parser.add_argument("--max-slices", type=int, default=0, help="0 schedules every eligible plane")
    parser.add_argument("--per-case", type=int, default=6)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--codes", type=int, nargs="+", default=list(range(1, 16)))
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
