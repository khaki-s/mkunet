"""Build the per-(case, combination) 3D window cache from the v1 export.

CPU-only and shardable by patient. For each pair it crops a support window, then
writes float16 image/prob, uint8 baseline labels, and a bool ROI as ``.npy``.
The training split additionally writes the GT four-class target and the ROI
baseline-error coordinates used for patch sampling. val/test caches never read GT.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from ...common import check_file, file_identity, read_json, same_grid, write_json
from ..dataset import MODALITIES, catalog, load_mask, load_probability, split_manifest
from . import data3d as D


def save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:   # a handle stops np.save appending ".npy"
            np.save(handle, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_mri(modalities, availability, reference):
    """Stack the four modality volumes, zeroing (and never opening) missing channels."""
    shape = reference.shape
    volume = np.zeros((4, *shape), dtype=np.float32)
    for i, name in enumerate(MODALITIES):
        if not availability[i]:
            continue
        identity = modalities[name]
        check_file(identity)
        image = nib.load(identity["path"])
        same_grid(image, reference)
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite MRI: {identity['path']}")
        volume[i] = data
    return volume


def build_pair(pair, output, split, truth_lookup):
    labels, reference = load_mask(pair.mask)
    probability = load_probability(pair, labels.shape)   # (3, X, Y, Z)
    present = D.availability(pair.code)
    volume = read_mri(pair.modalities, present, reference)
    zscored = D.zscore_mri(volume, present)
    box = D.support_box(zscored, probability[0], labels)
    sel_img = (slice(None), *box)
    image = zscored[sel_img].astype(np.float16)
    prob = probability[sel_img].astype(np.float16)
    base = labels[box].astype(np.uint8)
    roi = D.region_of_interest(labels, probability[0])[box]
    directory = D.window_dir(output, split, pair.case_id, pair.code)
    arrays = {}
    arrays["image"] = store(directory, "image", image)
    arrays["prob"] = store(directory, "prob", prob)
    arrays["base"] = store(directory, "base", base)
    arrays["roi"] = store(directory, "roi", roi)
    entry = {"code": pair.code, "baseline": pair.mask, "probability": pair.probability,
             "availability": present, "native_shape": list(labels.shape),
             "shape": list(base.shape), "box": [[s.start, s.stop] for s in box],
             "arrays": arrays}
    if split == "train":
        truth = truth_lookup(pair.case_id, reference)
        target = truth[box].astype(np.uint8)
        errors = np.argwhere(roi & (base != target)).astype(np.int32)
        arrays["target"] = store(directory, "target", target)
        arrays["errors"] = store(directory, "errors", errors)
        entry["error_count"] = int(len(errors))
    return entry


def store(directory, name, array):
    path = directory / f"{name}.npy"
    save_npy(path, array)
    return file_identity(path)


def truth_loader(truth_rows):
    def load(case_id, reference):
        identity = truth_rows[case_id]
        check_file(identity)
        image = nib.load(identity["path"])
        same_grid(image, reference)
        from ...common import read_labels
        return read_labels(image)
    return load


def encode_patient(pairs, output, split, binding, truth_lookup):
    case_id = pairs[0].case_id
    path = D.packet_path(output, split, case_id)
    if path.exists():
        packet = read_json(path)
        expected = {(p.code, p.mask["sha256"], p.probability["sha256"]) for p in pairs}
        actual = {(e["code"], e["baseline"]["sha256"], e["probability"]["sha256"]) for e in packet["pairs"]}
        if packet["binding"] != binding or actual != expected:
            raise ValueError(f"{path}: cached patient is from another run")
        return 0
    entries = [build_pair(pair, output, split, truth_lookup) for pair in pairs]
    write_json(path, {"binding": binding, "pairs": entries})
    return len(entries)


def run(args):
    output = args.output.resolve()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    prepare_sha = file_identity(Path(__file__))["sha256"]
    binding = D.packet_binding(args.split, prepare_sha)
    contract_path = output / "prepare.json"
    if args.shard == 0 and not contract_path.exists():
        write_json(contract_path, {"prepare_sha256": prepare_sha, "format": D.FORMAT_VERSION})
    cases = catalog(args.project, args.split, per_case=args.per_case, codes=tuple(args.codes),
                    limit_cases=args.limit_cases)
    truth_lookup = None
    if args.split == "train":
        rows = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "train", truth=True)}
        truth_lookup = truth_loader(rows)
    ids = sorted(cases)[args.shard::args.shards]
    for index, case_id in enumerate(ids, 1):
        written = encode_patient(cases[case_id], output, args.split, binding, truth_lookup)
        print(f"{args.split} {index}/{len(ids)} {case_id}: {written} windows", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=D.PROJECT)
    p.add_argument("--output", type=Path, default=D.DEFAULT_OUTPUT)
    p.add_argument("--split", choices=("train", "validation", "test"), default="train")
    p.add_argument("--per-case", type=int, default=6)
    p.add_argument("--codes", type=int, nargs="+", default=list(range(1, 16)))
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.cpu_threads)
    run(args)


if __name__ == "__main__":
    main()
