"""Missing-modality case index and label-free slice schedule."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d

from nnunetv2.utilities.brats2021_audit.missing_modality import MASK_CODES

from ..common import MODALITIES, check_file, manifest, read_json, read_labels, same_grid
from ..sam_domain_data import read_features

PROJECT = Path("/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113")
EXPORTS = "sam_protocol/legacy_prob_missing15"
HEADS = ("WT", "TC", "ET")
PRIORITY = {2: (0,), 3: (0,), **{code: (1, 2) for code in (1, 4, 5, 8, 9, 12, 13)}}
PREFERENCE = ((3, 2, 1, 0), (1, 0, 2, 3), (1, 0, 2, 3))
MANIFESTS = {"train": "train320", "validation": "val80", "test": "test100"}
PROB_FLOOR = 0.20
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Pair:
    case_id: str
    code: int
    modalities: dict
    mask: dict
    probability: dict
    baseline_model: dict


def split_manifest(project, split, *, truth=False):
    path = Path(project) / "manifests" / (MANIFESTS[split] + (".json" if truth else ".deployment.json"))
    return manifest(path, deployment=not truth, allowed=(split,))["cases"]


def catalog(project, split, *, per_case=6, codes=tuple(range(1, 16)), limit_cases=0):
    """Read exactly the exporter schedule. A split limit is a diagnostic subset."""
    rows = split_manifest(project, split)
    codes = tuple(sorted(set(codes)))
    if not (0 <= per_case <= 15 and 0 <= limit_cases <= len(rows)):
        raise ValueError("Invalid number of combinations or patients")
    if not codes or any(type(c) is not int or not 1 <= c <= 15 for c in codes):
        raise ValueError("Invalid mask code")
    target = Path(project) / EXPORTS / split
    pairs = defaultdict(list)
    common_model = None
    for index, row in enumerate(rows[:limit_cases or None]):
        case_id = row["case_id"]
        assigned = range(1, 16)
        if split == "train" and per_case:
            assigned = {MASK_CODES[(index * per_case + offset) % len(MASK_CODES)] for offset in range(per_case)}
        for code in codes:
            if code not in assigned:
                continue
            path = target / f"mask_{code:02d}" / f"{case_id}.json"
            record = read_json(path)
            availability = [(code >> i) & 1 for i in range(4)]
            modalities = {name: row["modalities"][name] for i, name in enumerate(MODALITIES) if availability[i]}
            expected = dict(case_id=case_id, modalities=modalities, availability=availability, mask_code=code)
            binding = record["binding"]
            if (record["gt_used"] is not False or binding["row"] != expected
                    or binding["mirroring"] is not True or binding["probabilities_exported"] is not True):
                raise ValueError(f"{path}: exporter contract mismatch")
            model = binding["model"]
            if common_model is None:
                common_model = model
            if model != common_model:
                raise ValueError(f"{path}: inconsistent baseline checkpoints")
            pairs[case_id].append(Pair(case_id, code, modalities, record["mask"], record["probabilities"], model))
    if not pairs:
        raise ValueError(f"{split}: empty case catalog")
    return dict(pairs)


def load_mask(identity):
    check_file(identity)
    image = nib.load(identity["path"])
    return read_labels(image), image


def load_probability(pair, shape):
    check_file(pair.probability)
    with np.load(pair.probability["path"], allow_pickle=False) as archive:
        p = archive["probabilities"].transpose(0, 3, 2, 1)
    if p.shape != (3, *shape) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError(f"Invalid probability shape or range for {pair.case_id}/{pair.code}")
    return p


def channel(head, code):
    return next(i for i in PREFERENCE[head] if code & (1 << i))


def _budget(indices, rank, limit):
    if not limit or len(indices) <= limit:
        return indices
    uniform_count = (limit + 1) // 2
    uniform = indices[np.rint(np.linspace(0, len(indices) - 1, uniform_count)).astype(int)]
    ranked = indices[np.argsort(-rank[indices], kind="stable")]
    extra = ranked[~np.isin(ranked, uniform)][:limit - len(uniform)]
    return np.sort(np.concatenate((uniform, extra)))


def slice_schedule(labels, probability, heads, limit, *, joint=False):
    """Spread queries across axial extent and prioritize likely edits using baseline evidence."""
    records = {}
    ranks = {}
    for h in heads:
        base = labels > h
        p = probability[h]
        candidate = (~base) & (p >= PROB_FLOOR)
        if h:
            candidate &= labels >= h
        # Adjacent-plane smoothing affects scheduling only; no masks are propagated.
        active = base.any((0, 1)) | candidate.any((0, 1))
        ranks[h] = gaussian_filter1d(
            ((base & (p < .8)) | candidate).sum((0, 1)).astype(float), sigma=1.)
        records[h] = np.flatnonzero(active)
    if joint and 1 in records and 2 in records:
        # Shared planes make ET->ED assessable by both region heads.
        z = np.union1d(records[1], records[2])
        keep = _budget(z, ranks[1] + ranks[2], limit)
        records[1], records[2] = keep, keep
    return [(h, int(z)) for h in heads for z in
            (records[h] if joint and h in (1, 2) and 1 in records and 2 in records
             else _budget(records[h], ranks[h], limit))]


def heads_for(code, scope):
    return tuple(range(3)) if scope == "all" else PRIORITY.get(code, ())


def packet_path(output, split, case_id):
    return Path(output) / "packets" / split / f"{case_id}.json"


def read_packet(path, expected):
    packet = read_json(path)
    if any(packet["binding"].get(k) != value for k, value in expected.items()):
        raise ValueError(f"{path}: packet was generated by a different refiner contract")
    return packet


@lru_cache(maxsize=16)
def _arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=16)
def truth_volume(path):
    return load_mask({"path": path[0], "sha256": path[1]})


def label_truth(identity):
    return truth_volume((identity["path"], identity["sha256"]))


class CachedDataset(Dataset):
    """Only this training dataset attaches GT to the otherwise label-free records."""

    def __init__(self, output, catalogs, binding, truth_rows):
        self.records = []
        self.truth = {r["case_id"]: r["gt"] for r in truth_rows}
        self.counts = Counter()
        for case_id, pairs in catalogs.items():
            packet = read_packet(packet_path(output, "train", case_id), binding)
            by_code = {pair.code: pair for pair in pairs}
            if set(by_code) != {entry["code"] for entry in packet["pairs"]}:
                raise ValueError(f"{case_id}: incomplete training packet")
            truth_identity = self.truth[case_id]
            check_file(truth_identity)
            truth_ref = nib.load(truth_identity["path"])
            for group in packet["pairs"]:
                pair = by_code[group["code"]]
                if pair.mask != group["baseline"] or pair.probability != group["probability"]:
                    raise ValueError(f"{case_id}: export changed after cache generation")
                same_grid(truth_ref, nib.load(pair.mask["path"]))
                if not group["records"]:
                    continue
                check_file(group["slices"])
                for i, record in enumerate(group["records"]):
                    self.records.append((case_id, pair.code, group["slices"]["path"], i, record))
                    self.counts[pair.code] += 1
        if not self.records:
            raise ValueError("No trainable cached slices")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        case_id, code, path, i, record = self.records[index]
        data = _arrays(path)
        target, _ = label_truth(self.truth[case_id])
        head, z = record["head"], record["z"]
        return {
            "features": read_features(record["feature"], verify=False),
            "baseline": data["labels"][i], "probability": data["probability"][i],
            "head": head, "code": code,
            "target": torch.from_numpy((target[:, :, z] > head)[None, None].copy()),
        }
