"""Label-free 3D cache geometry, baseline-prior projection, and patch dataset.

Everything here reuses the v1 catalog (``..dataset``) so the exporter schedule,
hash binding, and baseline checkpoint are identical across v1 and v2. The 3D
cache stores one cropped window per (case, combination): available MRI, baseline
region probabilities, baseline four-class labels, and an ROI mask. Only the
training cache additionally stores GT and the ROI baseline-error coordinates;
val/test caches never contain GT.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from ..dataset import (  # noqa: F401  (re-exported for v2 callers)
    HEADS, MANIFESTS, PROJECT, Pair, catalog, channel, load_mask,
    load_probability, split_manifest,
)

# Products live beside the v1 refiner cache but under a separate v2 root.
DEFAULT_OUTPUT = PROJECT / "sam_protocol" / "refiner" / "residual3d_v2"
FORMAT_VERSION = 1

WINDOW_PAD = 8            # voxels added around the support bounding box
WINDOW_PROB = 0.05        # p_WT threshold that enlarges the window
ROI_PROB = 0.10           # p_WT threshold that enlarges the editable ROI
ROI_DILATION = 4          # 6-neighbour dilations of baseline WT for the ROI
PRIOR_FLOOR = 1e-3        # clamp on the nested prior before taking its log
CLIP = 5.0                # z-score clip for cached MRI
REGIONS = ("WT", "TC", "ET")
LABEL_ORDER = ("BG", "ED", "NCR", "ET")


def availability(code):
    """Four modality-present flags (T1, T1ce, T2, FLAIR) from a 1..15 mask code."""
    return [(code >> i) & 1 for i in range(4)]


def nested_projection(prob):
    """Project the three region heads to a legal nested four-class prior.

    ``prob`` is ``(3, ...)`` with p_WT, p_TC, p_ET in [0, 1]. Returns ``q4`` with
    channels (BG, ED, NCR, ET) that sum to one and satisfy ET<=TC<=WT.
    """
    p = np.asarray(prob, dtype=np.float32)
    q_wt = p[0]
    q_tc = np.minimum(p[1], q_wt)
    q_et = np.minimum(p[2], q_tc)
    q4 = np.stack([1.0 - q_wt, q_wt - q_tc, q_tc - q_et, q_et], axis=0)
    return np.clip(q4, 0.0, 1.0)


def prior_logits(prob):
    """Log of the clamped nested prior; R3D adds a zero-initialised residual to it."""
    return np.log(np.clip(nested_projection(prob), PRIOR_FLOOR, None)).astype(np.float32)


def regions_from_softmax(posterior):
    """Nested region probabilities r_WT/r_TC/r_ET from a four-class posterior ``(4, ...)``."""
    p = np.asarray(posterior, dtype=np.float32)
    r_wt = 1.0 - p[0]
    r_tc = p[2] + p[3]
    r_et = p[3]
    return np.stack([r_wt, r_tc, r_et], axis=0)


def one_hot(labels):
    """Four-class one-hot ``(4, ...)`` float32 from internal labels BG/ED/NCR/ET."""
    labels = np.asarray(labels)
    return np.stack([(labels == c) for c in range(4)], axis=0).astype(np.float32)


def support_box(image, prob_wt, baseline):
    """Outer box of available-MRI support, baseline WT, and p_WT>=WINDOW_PROB, padded."""
    support = (image != 0).any(axis=0) | (baseline > 0) | (prob_wt >= WINDOW_PROB)
    coords = np.argwhere(support)
    if coords.size == 0:
        return tuple(slice(0, s) for s in baseline.shape)
    lo = np.maximum(coords.min(0) - WINDOW_PAD, 0)
    hi = np.minimum(coords.max(0) + WINDOW_PAD + 1, baseline.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def region_of_interest(baseline, prob_wt):
    """Editable ROI (no GT): dilated baseline WT unioned with p_WT>=ROI_PROB."""
    structure = generate_binary_structure(3, 1)
    dilated = binary_dilation(baseline > 0, structure, iterations=ROI_DILATION)
    return dilated | (prob_wt >= ROI_PROB)


def zscore_mri(volume, availability):
    """Per-modality z-score on nonzero voxels, clipped; missing modalities stay zero."""
    volume = np.asarray(volume, dtype=np.float32)
    out = np.zeros_like(volume)
    for i, present in enumerate(availability):
        if not present:
            continue
        channel_volume = volume[i]
        nonzero = channel_volume[channel_volume != 0]
        if nonzero.size:
            mean, std = float(nonzero.mean()), float(nonzero.std())
            std = std if std > 1e-6 else 1.0
            out[i] = np.clip((channel_volume - mean) / std, -CLIP, CLIP)
            out[i][channel_volume == 0] = 0.0
    return out


# --- window cache layout -------------------------------------------------

WINDOW_ARRAYS = ("image", "prob", "base", "roi")
TRAIN_ARRAYS = ("target", "errors")


def packet_path(output, split, case_id):
    return Path(output) / "windows" / split / f"{case_id}.json"


def window_dir(output, split, case_id, code):
    return Path(output) / "windows" / split / case_id / f"mask_{code:02d}"


def packet_binding(split, prepare_sha):
    """Bind the window cache to the split and the generator source hash."""
    return {"format": FORMAT_VERSION, "split": split, "prepare_sha256": prepare_sha,
            "window_pad": WINDOW_PAD, "window_prob": WINDOW_PROB, "roi_prob": ROI_PROB,
            "roi_dilation": ROI_DILATION, "gt_used": split == "train"}


@lru_cache(maxsize=64)
def _mmap(path):
    return np.load(path, mmap_mode="r")


def load_window(entry, name):
    """Memory-mapped read of a cached window array; shared across workers via page cache."""
    return _mmap(entry["arrays"][name]["path"])


def read_packet(path, expected):
    from ..dataset import read_json
    packet = read_json(path)
    binding = packet["binding"]
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{path}: window packet is from a different cache contract")
    return packet


# --- training patch dataset ----------------------------------------------

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from ...common import check_file  # noqa: E402


def assemble_input(image, prob, base, availability):
    """Stack the 15 R3D input channels: MRI 4, availability 4, p 3, baseline one-hot 4."""
    avail = np.asarray(availability, dtype=np.float32).reshape(4, *([1] * (image.ndim - 1)))
    avail = np.broadcast_to(avail, image.shape).astype(np.float32)
    channels = np.concatenate([image, avail, prob.astype(np.float32), one_hot(base)], axis=0)
    return np.ascontiguousarray(channels, dtype=np.float32)


def labels_from_prob(prob):
    """nnU-Net nested region rule: paint WT, then TC, then ET where p>0.5 (ET<=TC<=WT)."""
    labels = np.zeros(prob.shape[1:], dtype=np.int64)
    labels[prob[0] > 0.5] = 1
    labels[prob[1] > 0.5] = 2
    labels[prob[2] > 0.5] = 3
    return labels


def _smooth_field(shape, rng, scale=8.0):
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma=scale)
    # Smoothing shrinks the std to ~0.01; rescale so the amplitude U(0, 1.5) is in logit units.
    return field / max(float(field.std()), 1e-6)


def simulate_baseline_error(prob, rng):
    """Self-consistent baseline perturbation: sharpen/soften logits, add a smooth field, re-threshold."""
    logit = np.log(np.clip(prob, 1e-4, 1 - 1e-4) / np.clip(1 - prob, 1e-4, 1 - 1e-4))
    temperature = rng.uniform(1.0, 1.5)
    amplitude = rng.uniform(0.0, 1.5)
    noise = amplitude * _smooth_field(prob.shape[1:], rng)
    perturbed = 1.0 / (1.0 + np.exp(-(logit / temperature + noise)))
    perturbed = nested_projection_region(perturbed)
    return perturbed.astype(np.float32), labels_from_prob(perturbed)


def nested_projection_region(prob):
    """Enforce ET<=TC<=WT directly on the three region channels."""
    p = np.asarray(prob, dtype=np.float32)
    wt = p[0]
    tc = np.minimum(p[1], wt)
    et = np.minimum(p[2], tc)
    return np.stack([wt, tc, et], axis=0)


class WindowPatchDataset(Dataset):
    """Random 128^3 patches from the cached training windows (GT attached here only)."""

    def __init__(self, output, catalogs, binding, patch=128, steps=30000,
                 roi_error_frac=0.5, roi_any_frac=0.3, seed=0, augment=True,
                 simulate_prob=0.3):
        self.augment = augment
        self.simulate_prob = simulate_prob
        from ..dataset import read_json
        self.patch = patch
        self.steps = int(steps)
        self.fractions = (roi_error_frac, roi_any_frac)
        self.seed = seed
        self.records = []
        self.counts = {}
        for case_id, pairs in sorted(catalogs.items()):
            packet = read_packet(packet_path(output, "train", case_id), binding)
            by_code = {entry["code"]: entry for entry in packet["pairs"]}
            for pair in pairs:
                if pair.code not in by_code:
                    raise ValueError(f"{case_id}: window missing combination {pair.code}")
                entry = by_code[pair.code]
                if entry["baseline"] != pair.mask or entry["probability"] != pair.probability:
                    raise ValueError(f"{case_id}: export changed after window cache generation")
                for name in (*WINDOW_ARRAYS, *TRAIN_ARRAYS):
                    check_file(entry["arrays"][name])
                self.records.append(entry)
                self.counts[pair.code] = self.counts.get(pair.code, 0) + 1
        if not self.records:
            raise ValueError("No training windows found")

    def __len__(self):
        return self.steps

    def _center(self, rng, entry):
        shape = np.array(entry["shape"], dtype=np.int64)
        error_frac, any_frac = self.fractions
        draw = rng.random()
        errors = entry.get("error_count", 0)
        if draw < error_frac and errors:
            coords = load_window(entry, "errors")
            return np.asarray(coords[rng.integers(0, len(coords))], dtype=np.int64)
        if draw < error_frac + any_frac:
            roi = np.asarray(load_window(entry, "roi"))
            inside = np.argwhere(roi)
            if len(inside):
                return np.asarray(inside[rng.integers(0, len(inside))], dtype=np.int64)
        return rng.integers(0, np.maximum(shape, 1))

    def _slices(self, center, shape):
        half = self.patch // 2
        lo = np.clip(center - half, 0, np.maximum(shape - self.patch, 0))
        return tuple(slice(int(a), int(a + self.patch)) for a in lo)

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, index))
        entry = self.records[rng.integers(0, len(self.records))]
        shape = np.array(entry["shape"], dtype=np.int64)
        center = self._center(rng, entry)
        sel = self._slices(center, shape)
        image = np.asarray(load_window(entry, "image"))[(slice(None), *sel)].astype(np.float32)
        prob = np.asarray(load_window(entry, "prob"))[(slice(None), *sel)].astype(np.float32)
        base = np.asarray(load_window(entry, "base"))[sel].astype(np.int64)
        roi = np.asarray(load_window(entry, "roi"))[sel]
        target = np.asarray(load_window(entry, "target"))[sel].astype(np.int64)
        pad = [(0, self.patch - s) for s in base.shape]
        if any(b for _, b in pad):
            image = np.pad(image, [(0, 0), *pad])
            prob = np.pad(prob, [(0, 0), *pad])
            base = np.pad(base, pad)
            roi = np.pad(roi, pad)
            target = np.pad(target, pad)
        availability = entry["availability"]
        if self.augment:
            for axis in (1, 2, 3):
                if rng.random() < 0.5:
                    image, prob = np.flip(image, axis), np.flip(prob, axis)
                    base, roi, target = (np.flip(a, axis - 1) for a in (base, roi, target))
            for i, present in enumerate(availability):
                if present:
                    image[i] = image[i] * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
            if rng.random() < self.simulate_prob:
                prob, base = simulate_baseline_error(prob, rng)
        image = np.ascontiguousarray(image)
        prob = np.ascontiguousarray(prob)
        base = np.ascontiguousarray(base)
        features = assemble_input(image, prob, base, availability)
        return {
            "features": torch.from_numpy(features),
            "prior": torch.from_numpy(prior_logits(prob)),
            "base": torch.from_numpy(base.copy()),
            "roi": torch.from_numpy(np.ascontiguousarray(roi).astype(np.bool_)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
        }
