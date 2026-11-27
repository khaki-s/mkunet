"""GT-free SAM3 prompt construction for the mask refiner.

One slice, one region. Inputs are the baseline binary mask of that region on the
slice and the baseline probability of that region (both native H x W). No GT.

Borrowed from the reference repos (third_party_src_20261125):
- SAMRefiner: distance-centre positive point, in-box farthest background negative
  point, bounding box, and a signed dense-mask prior (utils.extract_points / extract_mask).
- SAMedEnhancer: condition the dense prior on the baseline *probability*, not only
  on the binary mask, so sub-threshold evidence still reaches SAM.
- MaskSAM: box derived from the mask, used together with the dense prior.

Coordinates are native (x, y) = (column, row); SAM scaling to 1008 happens in model.py.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label

MASK_INPUT_SIZE = (288, 288)   # SAM3 prompt_encoder.mask_input_size
EMPTY_SDF = -12.0              # deterministic signed value for an all-background slice


def _distance_centre_points(mask, max_points=3):
    """Farthest-inside points, one per connected component (largest first)."""
    components, n = label(mask)
    if n == 0:
        return np.zeros((0, 2), np.float32)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    points = []
    for index in np.argsort(-sizes):
        if sizes[index] == 0 or len(points) >= max_points:
            break
        dist = distance_transform_edt(np.pad(components == index, 1))[1:-1, 1:-1]
        row, col = np.unravel_index(int(np.argmax(dist)), mask.shape)
        points.append([float(col), float(row)])
    return np.asarray(points, np.float32)


def _negative_point(mask):
    """Farthest background point inside the mask's bounding box (SAMRefiner neg point)."""
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    box_bg = np.zeros_like(mask)
    box_bg[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1] = True
    box_bg &= ~mask
    if not box_bg.any():
        return np.zeros((0, 2), np.float32)
    dist = distance_transform_edt(box_bg)
    row, col = np.unravel_index(int(np.argmax(dist)), mask.shape)
    return np.asarray([[float(col), float(row)]], np.float32)


def _box(mask, shape):
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0)
    return np.asarray([lo[1], lo[0], hi[1], hi[0]], np.float32)   # x1,y1,x2,y2


def dense_prior(mask, prob):
    """Signed dense prior at MASK_INPUT_SIZE: baseline probability steers SAM.

    Inside the mask the value is positive, outside negative; magnitude grows with
    |p-0.5| so confident regions push harder. Sub-threshold probability (SAMedEnhancer)
    keeps a mild positive bias so near-miss voxels are reachable.
    """
    prob = np.clip(np.asarray(prob, np.float32), 0.0, 1.0)
    signed = (prob - 0.5) * 12.0                    # [-6, 6]
    if mask.any():
        sdf = distance_transform_edt(mask) - distance_transform_edt(~mask)
        signed = np.clip(signed + np.clip(sdf / 2.0, -6.0, 6.0), -12.0, 12.0) * 0.5
    else:
        signed = np.where(prob > 0.1, signed, EMPTY_SDF)
    return signed.astype(np.float32)


def slice_prompt(mask, prob, *, max_points=3):
    """Return native prompts for one region on one slice (no GT).

    Keys: points [N,2] xy, point_labels [N] (1 pos / 0 neg), box [4] xyxy,
    dense [H,W] signed prior. For an empty baseline mask a whole-slice box and a
    negative point encode "nothing here unless probability says otherwise".
    """
    mask = np.asarray(mask, bool)
    prob = np.asarray(prob, np.float32)
    if mask.shape != prob.shape or mask.ndim != 2:
        raise ValueError('mask and prob must be the same 2D native shape')
    dense = dense_prior(mask, prob)
    if not mask.any():
        h, w = mask.shape
        return {'points': np.zeros((1, 2), np.float32), 'point_labels': np.array([-1], np.int64),
                'box': np.array([0, 0, w - 1, h - 1], np.float32), 'dense': dense}
    pos = _distance_centre_points(mask, max_points)
    neg = _negative_point(mask)
    points = np.concatenate([pos, neg], 0) if len(neg) else pos
    labels = np.concatenate([np.ones(len(pos), np.int64), np.zeros(len(neg), np.int64)])
    return {'points': points.astype(np.float32), 'point_labels': labels,
            'box': _box(mask, mask.shape), 'dense': dense}
