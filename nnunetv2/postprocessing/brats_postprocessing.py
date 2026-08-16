"""
brats_postprocess.py
=====================
Post-processing for BraTS 2021 predictions that enforces the
label hierarchy  ET ⊂ TC ⊂ WT.

Label convention is NOT hardcoded: it comes from `region_ids`, defaulting to
the nnUNet region-based table (WT={1,2,3}, TC={2,3}, ET={3}) that
Dataset137_BraTS2021 declares in its dataset.json. Note this SWAPS labels 1
and 2 relative to raw BraTS (where TC={1,3}) -- pass REGION_IDS_BRATS_RAW or
region_ids_from_dataset_json(...) for other datasets.

Usage (inference pipeline):
    python brats_postprocess.py \
        --pred_dir  /path/to/raw_predictions \
        --val_dir   /path/to/val_labels \
        --out_dir   /path/to/corrected_predictions \
        --search_thresholds          # run threshold search on val set

    After threshold search, call again without --search_thresholds to
    apply the found thresholds to a test set.

Algorithm (applied per 3-D volume, assembled from 2D slices):
    Step 1 – Hierarchy enforcement
        TC  = TC  ∩ WT           (TC cannot exist outside WT)
        ET  = ET  ∩ TC           (ET cannot exist outside TC)
    Step 2 – Small connected-component filtering
        Remove ET CCs  smaller than et_min_size  voxels
        Remove TC CCs  smaller than tc_min_size  voxels
        Remove WT CCs  smaller than wt_min_size  voxels
    Step 3 – ET-outside-WT hard removal
        Any remaining ET voxel outside WT is zeroed (second pass after CC filter)
    Step 4 – TC-free WT removal  (optional, --drop_wt_without_tc)
        Delete WT CCs containing no TC voxel at all.
    Step 5 – Threshold search  (validation set only)
        Grid-search over {wt_min_size, tc_min_size, et_min_size} using Dice on
        val labels. Saves best thresholds to a JSON file. WT is only swept if
        you pass --search_wt; otherwise it is fixed and the search cannot
        improve WT at all.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

# ── Optional: nibabel for .nii.gz I/O ────────────────────────────────────────
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    warnings.warn("nibabel not found; .nii.gz I/O disabled. "
                  "pip install nibabel")

# ── Optional: SimpleITK for .nii.gz I/O ──────────────────────────────────────
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


# ============================================================================
#  Core correction function (pure numpy, no I/O)
# ============================================================================

def enforce_brats_hierarchy(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hard-enforce ET ⊂ TC ⊂ WT on binary masks.
    Returns corrected (wt, tc, et) — all uint8.

    Args:
        wt, tc, et : binary numpy arrays (same shape), dtype bool or uint8
    """
    wt = wt.astype(bool)
    tc = tc.astype(bool) & wt     # TC must lie inside WT
    et = et.astype(bool) & tc     # ET must lie inside TC
    return wt.astype(np.uint8), tc.astype(np.uint8), et.astype(np.uint8)


def label_components(
    mask: np.ndarray,
    connectivity: int = 2,       # in 3D: 1=6-conn, 2=18-conn, 3=26-conn
) -> Tuple[np.ndarray, np.ndarray]:
    """Label connected components once and return per-component sizes.

    Split out of remove_small_components so that a threshold *sweep* (many
    min_size values on the same mask) can label the mask a single time and
    then just re-filter the (cheap, vectorized) size array -- ndimage.label
    is the expensive step on ~9M-voxel BraTS volumes, and it does not depend
    on min_size at all.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32), np.array([])

    struct = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labeled, n_cc = ndimage.label(mask, structure=struct)
    if n_cc == 0:
        return labeled, np.array([])
    sizes = np.asarray(ndimage.sum(mask, labeled, range(1, n_cc + 1)))
    return labeled, sizes


def filter_labeled_by_size(
    labeled: np.ndarray,
    sizes: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Re-filter an already-labeled mask by a new min_size -- O(volume),
    no relabeling. Use with label_components() when sweeping thresholds."""
    if min_size <= 0 or sizes.size == 0:
        return (labeled > 0).astype(np.uint8)
    keep = sizes >= min_size
    lut = np.zeros(sizes.size + 1, dtype=bool)   # lut[0] = background = False
    lut[1:] = keep
    return lut[labeled].astype(np.uint8)


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove connected components smaller than `min_size` voxels.
    Works on 2D or 3D arrays.

    Args:
        mask        : binary array (bool or uint8)
        min_size    : minimum CC size to keep (voxels). 0 → no filtering.
        connectivity: passed to ndimage.label via structure
    Returns:
        Cleaned binary mask (uint8).
    """
    if min_size <= 0:
        return mask.astype(np.uint8)
    if not mask.any():
        return mask.astype(np.uint8)

    labeled, sizes = label_components(mask, connectivity)
    return filter_labeled_by_size(labeled, sizes, min_size)


def remove_wt_components_without_tc(
    wt: np.ndarray,
    tc: np.ndarray,
    connectivity: int = 2,
    max_size: int = 0,
) -> np.ndarray:
    """Delete WT connected components that contain no TC voxel at all.

    Targets non-tumour FLAIR hyperintensity (gliosis, old infarcts) that the
    network calls oedema: real oedema almost always accompanies a core, so a
    WT island with zero core inside is usually a false positive.

    `max_size` bounds the downside: only drop TC-free components at or below
    that many voxels (0 = no limit). Measured on val, dropping any size was
    ~neutral overall (+0.0005 WT, ~0.3 sigma) but swung individual cases hard
    (+0.082 on 00493, -0.120 on 00638), so capping is the safer default if
    you enable this at all.
    """
    wt = wt.astype(bool)
    if not wt.any():
        return wt.astype(np.uint8)

    labeled, sizes = label_components(wt, connectivity)
    n_cc = int(sizes.size)
    if n_cc <= 0:
        return np.zeros_like(wt, dtype=np.uint8)

    tc_b = tc.astype(bool)
    labels = np.arange(1, n_cc + 1)
    # TC voxels per component; 0 -> no core inside
    tc_per_cc = ndimage.sum(tc_b, labeled, labels)

    drop = np.asarray(tc_per_cc) == 0
    if max_size > 0:
        drop &= np.asarray(sizes) <= max_size

    if not drop.any():
        return wt.astype(np.uint8)

    return np.isin(labeled, labels[~drop]).astype(np.uint8)


def postprocess_volume(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
    drop_wt_without_tc: bool = False,
    wt_without_tc_max_size: int = 0,
    connectivity: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline for a single BraTS volume.

    Args:
        wt, tc, et   : predicted binary masks (H, W) or (D, H, W)
        et_min_size  : ET connected components smaller than this are removed
        tc_min_size  : TC connected components smaller than this are removed
        wt_min_size  : WT connected components smaller than this are removed
        drop_wt_without_tc     : also delete WT CCs containing no TC voxel
        wt_without_tc_max_size : size cap for the above (0 = no cap)

    Returns:
        Corrected (wt, tc, et) as uint8 arrays.
    """
    # ── Step 1: hierarchy enforcement ────────────────────────────────────────
    wt, tc, et = enforce_brats_hierarchy(wt, tc, et)

    # ── Step 2: remove small connected components ─────────────────────────────
    wt = remove_small_components(wt, wt_min_size, connectivity)
    tc = remove_small_components(tc, tc_min_size, connectivity)
    et = remove_small_components(et, et_min_size, connectivity)

    # ── Step 3: second-pass hierarchy enforcement after CC filter ─────────────
    # (CC filter on WT may have removed WT voxels that TC/ET sat on)
    wt, tc, et = enforce_brats_hierarchy(wt, tc, et)

    # ── Step 4: drop WT components with no core inside ────────────────────────
    if drop_wt_without_tc:
        wt = remove_wt_components_without_tc(
            wt, tc, connectivity, max_size=wt_without_tc_max_size)
        wt, tc, et = enforce_brats_hierarchy(wt, tc, et)

    return wt, tc, et


# ============================================================================
#  nnUNet-format label array helpers
# ============================================================================

# nnUNet region-based datasets (e.g. Dataset137_BraTS2021) declare their own
# label->region table in dataset.json, and Dataset137 SWAPS labels 1 and 2
# relative to the raw BraTS convention: its TC is {2,3}, not {1,3}. Hardcoding
# the raw convention silently computes the wrong TC on that dataset, which in
# turn corrupts every "does this WT component contain TC" decision.
REGION_IDS_NNUNET_REGIONS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,)}
REGION_IDS_BRATS_RAW      = {'WT': (1, 2, 3), 'TC': (1, 3), 'ET': (3,)}

DEFAULT_REGION_IDS = REGION_IDS_NNUNET_REGIONS


def region_ids_from_dataset_json(dataset_json: dict) -> Dict[str, Tuple[int, ...]]:
    """Read the label->region table straight out of a dataset.json.

    Safer than guessing: matches on the region names BraTS datasets use.
    """
    labels = dataset_json['labels']
    alias = {'whole tumor': 'WT', 'whole_tumor': 'WT', 'wt': 'WT',
             'tumor core': 'TC', 'tumor_core': 'TC', 'tc': 'TC',
             'enhancing tumor': 'ET', 'enhancing_tumor': 'ET', 'et': 'ET'}
    out = {}
    for name, ids in labels.items():
        key = alias.get(str(name).strip().lower())
        if key is None:
            continue
        out[key] = tuple(ids) if isinstance(ids, (list, tuple)) else (int(ids),)
    missing = {'WT', 'TC', 'ET'} - set(out)
    if missing:
        raise ValueError(f"dataset.json labels {list(labels)} do not define {missing}")
    return out


def pred_to_masks(
    seg: np.ndarray,
    region_ids: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a nnUNet BraTS segmentation array to (wt, tc, et) binary masks.

    Args:
        region_ids : label ids per region. Defaults to the nnUNet region-based
                     convention (TC={2,3}); pass REGION_IDS_BRATS_RAW for raw
                     BraTS labels, or region_ids_from_dataset_json(...).
    """
    r = region_ids or DEFAULT_REGION_IDS
    wt = np.isin(seg, r['WT']).astype(np.uint8)
    tc = np.isin(seg, r['TC']).astype(np.uint8)
    et = np.isin(seg, r['ET']).astype(np.uint8)
    return wt, tc, et


def masks_to_pred(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    original_seg: Optional[np.ndarray] = None,
    region_ids: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> np.ndarray:
    """
    Reconstruct a nnUNet BraTS label array from (wt, tc, et) binary masks.

    Label values are derived from `region_ids` by set difference (WT-only =
    oedema, TC-only = non-enhancing core, ET) rather than hardcoded, so this
    round-trips correctly regardless of which convention the dataset uses.
    """
    r = region_ids or DEFAULT_REGION_IDS
    wt_only = sorted(set(r['WT']) - set(r['TC']))
    tc_only = sorted(set(r['TC']) - set(r['ET']))
    et_ids  = sorted(r['ET'])
    if not (wt_only and tc_only and et_ids):
        raise ValueError(f"region_ids {r} must be strictly nested ET < TC < WT")

    out = np.zeros_like(wt, dtype=np.uint8)
    out[wt == 1] = wt_only[0]    # oedema: in WT, not in TC
    out[tc == 1] = tc_only[0]    # non-enhancing core: in TC, not in ET
    out[et == 1] = et_ids[0]     # enhancing tumour
    return out


def postprocess_pred_array(
    seg: np.ndarray,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
    drop_wt_without_tc: bool = False,
    wt_without_tc_max_size: int = 0,
    region_ids: Optional[Dict[str, Tuple[int, ...]]] = None,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Convenience wrapper: takes a nnUNet BraTS seg array, post-processes it,
    and returns a corrected seg array.
    """
    wt, tc, et = pred_to_masks(seg, region_ids)
    wt, tc, et = postprocess_volume(wt, tc, et, et_min_size, tc_min_size,
                                    wt_min_size,
                                    drop_wt_without_tc=drop_wt_without_tc,
                                    wt_without_tc_max_size=wt_without_tc_max_size,
                                    connectivity=connectivity)
    return masks_to_pred(wt, tc, et, seg, region_ids)


# ============================================================================
#  Dice metric (numpy)
# ============================================================================

def dice_from_counts(
    n_pred: float,
    n_gt: float,
    n_inter: float,
    empty_score: float = 1.0,
) -> float:
    """Dice from raw counts, with the BraTS empty-case convention.

    If GT and prediction are both empty the region was correctly predicted
    absent, which BraTS scores as 1.0 -- not 0.0. This matters for ET, where
    a nontrivial number of BraTS cases have no enhancing tumour at all: the
    old `2*0/(0+0+eps)` returned 0.0 and dragged the mean ET down, making the
    threshold search prefer whatever setting happened to hallucinate a few ET
    voxels in those cases.
    """
    denom = n_pred + n_gt
    if denom == 0:
        return float(empty_score)
    return float(2.0 * n_inter / denom)


def dice(pred: np.ndarray, gt: np.ndarray, empty_score: float = 1.0) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    return dice_from_counts(p.sum(), g.sum(), (p & g).sum(), empty_score)


def brats_dice(
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
) -> Dict[str, float]:
    """Compute WT/TC/ET Dice from nnUNet label arrays."""
    p_wt, p_tc, p_et = pred_to_masks(pred_seg)
    g_wt, g_tc, g_et = pred_to_masks(gt_seg)
    return {
        'WT': dice(p_wt, g_wt),
        'TC': dice(p_tc, g_tc),
        'ET': dice(p_et, g_et),
    }


# ============================================================================
#  Threshold search on validation set
# ============================================================================

def _union_bbox_slices(*masks: np.ndarray, margin: int = 1):
    """Bounding box around the union of all masks, padded by `margin`.

    BraTS volumes are ~155x240x240 = 8.9M voxels but the tumour plus its GT
    typically occupies a few percent of that. Connected-component labeling
    cost scales with the array we hand it, so cropping to this box before
    labeling is the single biggest speedup available -- and it is exact, not
    an approximation: every foreground voxel of every mask is inside the box,
    so no component can be split or merged by the crop.
    """
    union = np.zeros(masks[0].shape, dtype=bool)
    for m in masks:
        union |= m.astype(bool)
    if not union.any():
        return tuple(slice(0, 0) for _ in union.shape)

    sl = []
    for ax in range(union.ndim):
        other = tuple(a for a in range(union.ndim) if a != ax)
        proj = union.any(axis=other)
        idx = np.nonzero(proj)[0]
        lo = max(0, int(idx[0]) - margin)
        hi = min(union.shape[ax], int(idx[-1]) + 1 + margin)
        sl.append(slice(lo, hi))
    return tuple(sl)


def _component_stats(
    mask: np.ndarray,
    gt_mask: np.ndarray,
    connectivity: int,
    parent_labeled: Optional[np.ndarray] = None,
):
    """Label `mask` and return per-component (size, GT-intersection, parent).

    `parent_labeled` is the labeled array of the enclosing region (e.g. the
    labeled WT for TC components). Because postprocess_volume enforces the
    hierarchy BEFORE any size filtering, every component here is fully
    contained in exactly one parent component, so a single ndimage.maximum
    per component recovers the parent id exactly.
    """
    labeled, sizes = label_components(mask, connectivity)
    n = int(sizes.size)
    if n == 0:
        z = np.zeros(0)
        return labeled, z, z, np.zeros(0, dtype=np.int64)

    # GT intersection per component: bincount over labels at GT voxels
    inter = np.bincount(labeled[gt_mask.astype(bool)].ravel(),
                        minlength=n + 1)[1:].astype(np.float64)

    if parent_labeled is None:
        parent = np.zeros(n, dtype=np.int64)
    else:
        parent = np.asarray(
            ndimage.maximum(parent_labeled, labeled, np.arange(1, n + 1))
        ).astype(np.int64).reshape(-1)

    return labeled, sizes.astype(np.float64), inter, parent


def _precompute_volume_for_search(
    pred_file: Path,
    gt_file: Path,
    connectivity: int,
    load_fn,
    region_ids: Optional[Dict[str, Tuple[int, ...]]],
):
    """Reduce one volume to a handful of small per-component arrays.

    Once we have (size, GT-intersection, parent) per component, every grid
    point's Dice is exact arithmetic on arrays of length n_components (<100
    typically) -- the grid loop never touches a voxel again. That turns the
    sweep from O(n_grid x volume) into O(n_grid x n_components).
    """
    pred = load_fn(pred_file)
    gt   = load_fn(gt_file)

    wt0, tc0, et0 = pred_to_masks(pred, region_ids)
    wt0, tc0, et0 = enforce_brats_hierarchy(wt0, tc0, et0)
    g_wt, g_tc, g_et = pred_to_masks(gt, region_ids)

    # crop everything to the union bbox before labeling (exact, see above)
    sl = _union_bbox_slices(wt0, g_wt, margin=1)
    wt0, tc0, et0 = wt0[sl], tc0[sl], et0[sl]
    cg_wt, cg_tc, cg_et = g_wt[sl], g_tc[sl], g_et[sl]

    lab_wt, sz_wt, in_wt, _ = _component_stats(wt0, cg_wt, connectivity)
    lab_tc, sz_tc, in_tc, par_tc = _component_stats(tc0, cg_tc, connectivity,
                                                    parent_labeled=lab_wt)
    _,      sz_et, in_et, par_et = _component_stats(et0, cg_et, connectivity,
                                                    parent_labeled=lab_tc)

    return {
        'sz_wt': sz_wt, 'in_wt': in_wt,
        'sz_tc': sz_tc, 'in_tc': in_tc, 'par_tc': par_tc,
        'sz_et': sz_et, 'in_et': in_et, 'par_et': par_et,
        'n_gt_wt': float(g_wt.sum()),
        'n_gt_tc': float(g_tc.sum()),
        'n_gt_et': float(g_et.sum()),
    }


def _eval_grid_point(
    v: dict,
    wt_sz: int,
    tc_sz: int,
    et_sz: int,
    drop_wt_without_tc: bool = False,
    wt_without_tc_max_size: int = 0,
) -> Tuple[float, float, float]:
    """Exact WT/TC/ET Dice for one volume at one grid point, O(n_components).

    Mirrors postprocess_volume's semantics: size-filter each region, then
    re-enforce the hierarchy (a child component survives only if its parent
    did), then optionally drop TC-free WT components.
    """
    wt_keep = v['sz_wt'] >= wt_sz if wt_sz > 0 else np.ones(v['sz_wt'].size, bool)
    tc_keep = v['sz_tc'] >= tc_sz if tc_sz > 0 else np.ones(v['sz_tc'].size, bool)
    et_keep = v['sz_et'] >= et_sz if et_sz > 0 else np.ones(v['sz_et'].size, bool)

    # hierarchy: child survives only inside a surviving parent
    if tc_keep.size:
        tc_keep &= wt_keep[v['par_tc'] - 1] if wt_keep.size else False
    if et_keep.size:
        et_keep &= tc_keep[v['par_et'] - 1] if tc_keep.size else False

    if drop_wt_without_tc and wt_keep.size:
        has_tc = np.zeros(wt_keep.size, dtype=bool)
        if tc_keep.any():
            has_tc[v['par_tc'][tc_keep] - 1] = True
        drop = wt_keep & ~has_tc
        if wt_without_tc_max_size > 0:
            drop &= v['sz_wt'] <= wt_without_tc_max_size
        wt_keep = wt_keep & ~drop
        # re-enforce hierarchy after the drop
        if tc_keep.size:
            tc_keep &= wt_keep[v['par_tc'] - 1]
        if et_keep.size:
            et_keep &= tc_keep[v['par_et'] - 1]

    d_wt = dice_from_counts(v['sz_wt'][wt_keep].sum(), v['n_gt_wt'],
                            v['in_wt'][wt_keep].sum())
    d_tc = dice_from_counts(v['sz_tc'][tc_keep].sum(), v['n_gt_tc'],
                            v['in_tc'][tc_keep].sum())
    d_et = dice_from_counts(v['sz_et'][et_keep].sum(), v['n_gt_et'],
                            v['in_et'][et_keep].sum())
    return d_wt, d_tc, d_et


def search_thresholds(
    pred_files: list[Path],
    gt_files: list[Path],
    et_min_sizes: list[int] = None,
    tc_min_sizes: list[int] = None,
    wt_min_size: int = 100,
    wt_min_sizes: list[int] = None,   # if given, WT is swept too
    load_fn=None,
    connectivity: int = 2,
    region_ids: Optional[Dict[str, Tuple[int, ...]]] = None,
    num_processes: Optional[int] = None,
    drop_wt_without_tc: bool = False,
    wt_without_tc_max_size: int = 0,
    region_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict:
    """
    Grid search over CC size thresholds to maximise weighted mean Dice on a
    labelled split.

    The search must be configured the same way as deployment: pass the same
    drop_wt_without_tc / wt_without_tc_max_size you intend to apply, or the
    thresholds are optimal for a pipeline you are not running.

    If wt_min_sizes is None, WT stays fixed at wt_min_size -- in that case the
    search cannot improve WT at all, it only adds a constant to the objective.
    Pass a list (or use --search_wt on the CLI) to actually tune it.

    Returns:
        dict with best thresholds, per-region Dice at the chosen point, the
        no-filter baseline for comparison, and the full grid.
    """
    if et_min_sizes is None:
        et_min_sizes = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    if tc_min_sizes is None:
        tc_min_sizes = [0, 10, 25, 50, 75, 100, 150, 200, 300]
    if wt_min_sizes is None:
        wt_min_sizes = [wt_min_size]

    if load_fn is None:
        load_fn = _default_load
    if num_processes is None:
        try:
            from nnunetv2.configuration import default_num_processes
            num_processes = default_num_processes
        except ImportError:
            num_processes = os.cpu_count() or 1

    n_combos = len(et_min_sizes) * len(tc_min_sizes) * len(wt_min_sizes)
    print(f"Threshold search: {len(pred_files)} volumes, "
          f"{len(wt_min_sizes)}×{len(tc_min_sizes)}×{len(et_min_sizes)} "
          f"(WT×TC×ET) = {n_combos} combinations.\n"
          f"  Labeling each volume once (connectivity={connectivity}, "
          f"bbox-cropped) across {num_processes} processes; the grid loop then "
          f"runs on per-component arrays only, never on voxels.")

    import multiprocessing
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        cached = pool.starmap(
            _precompute_volume_for_search,
            [(pf, gf, connectivity, load_fn, region_ids)
             for pf, gf in zip(pred_files, gt_files)]
        )
    print(f"  Precompute done: "
          f"{sum(v['sz_wt'].size for v in cached)} WT / "
          f"{sum(v['sz_tc'].size for v in cached)} TC / "
          f"{sum(v['sz_et'].size for v in cached)} ET components total.")

    w_wt, w_tc, w_et = region_weights
    w_sum = w_wt + w_tc + w_et

    best_score = -1.0
    best = {'wt': wt_min_sizes[0], 'tc': tc_min_sizes[0], 'et': et_min_sizes[0]}
    grid_results = []

    for wt_sz in wt_min_sizes:
        for tc_sz in tc_min_sizes:
            for et_sz in et_min_sizes:
                d = np.array([
                    _eval_grid_point(v, wt_sz, tc_sz, et_sz,
                                     drop_wt_without_tc, wt_without_tc_max_size)
                    for v in cached
                ])
                mean_wt, mean_tc, mean_et = (float(x) for x in d.mean(axis=0))
                score = (w_wt * mean_wt + w_tc * mean_tc + w_et * mean_et) / w_sum

                grid_results.append({
                    'et_min_size': et_sz,
                    'tc_min_size': tc_sz,
                    'wt_min_size': wt_sz,
                    'WT': mean_wt, 'TC': mean_tc, 'ET': mean_et,
                    'mean': score,
                })

                if score > best_score:
                    best_score = score
                    best = {'wt': wt_sz, 'tc': tc_sz, 'et': et_sz}
                    print(f"  → New best: WT_sz={wt_sz:4d}  TC_sz={tc_sz:4d}  "
                          f"ET_sz={et_sz:4d}  WT={mean_wt:.4f}  TC={mean_tc:.4f}  "
                          f"ET={mean_et:.4f}  score={score:.4f}")

    # Report the no-filtering baseline and the per-region optima so the chosen
    # point can be sanity-checked -- a single weighted mean will happily trade
    # WT away for ET, which is the opposite of what we want here.
    base = next((g for g in grid_results
                 if g['et_min_size'] == 0 and g['tc_min_size'] == 0
                 and g['wt_min_size'] == 0), None)
    if base is not None:
        print(f"\n  no-filter baseline (0/0/0): WT={base['WT']:.4f}  "
              f"TC={base['TC']:.4f}  ET={base['ET']:.4f}")
    else:
        print("\n  (no-filter baseline not in grid -- add 0 to every grid, or "
              "use --search_wt, to see whether filtering helps at all)")
    for reg in ('WT', 'TC', 'ET'):
        g = max(grid_results, key=lambda r: r[reg])
        print(f"  best {reg} alone: {g[reg]:.4f} at "
              f"WT_sz={g['wt_min_size']} TC_sz={g['tc_min_size']} "
              f"ET_sz={g['et_min_size']}  "
              f"(WT={g['WT']:.4f} TC={g['TC']:.4f} ET={g['ET']:.4f})")

    chosen = next(g for g in grid_results
                  if g['wt_min_size'] == best['wt']
                  and g['tc_min_size'] == best['tc']
                  and g['et_min_size'] == best['et'])
    print(f"\nBest thresholds: wt_min_size={best['wt']}, "
          f"tc_min_size={best['tc']}, et_min_size={best['et']}  "
          f"(WT={chosen['WT']:.4f}  TC={chosen['TC']:.4f}  "
          f"ET={chosen['ET']:.4f}  score={best_score:.4f})")

    return {
        'best_et_min_size': best['et'],
        'best_tc_min_size': best['tc'],
        'best_wt_min_size': best['wt'],
        'best_dice_mean':   best_score,
        'best_dice_per_region': {'WT': chosen['WT'], 'TC': chosen['TC'],
                                 'ET': chosen['ET']},
        'baseline_no_filter': base,
        'search_config': {
            'connectivity': connectivity,
            'drop_wt_without_tc': drop_wt_without_tc,
            'wt_without_tc_max_size': wt_without_tc_max_size,
            'region_weights': list(region_weights),
        },
        'grid_results':     grid_results,
    }


# ============================================================================
#  File I/O helpers
# ============================================================================

def _default_load(path: Path) -> np.ndarray:
    path = Path(path)
    if HAS_NIBABEL:
        return np.asarray(nib.load(str(path)).dataobj).astype(np.uint8)
    elif HAS_SITK:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.uint8)
    elif path.suffix == '.npy':
        return np.load(str(path)).astype(np.uint8)
    else:
        raise RuntimeError("Neither nibabel nor SimpleITK available. "
                           "Install one: pip install nibabel")


def _default_save(arr: np.ndarray, path: Path, ref_path: Path) -> None:
    path = Path(path)
    if HAS_NIBABEL:
        ref = nib.load(str(ref_path))
        img = nib.Nifti1Image(arr, ref.affine, ref.header)
        nib.save(img, str(path))
    elif HAS_SITK:
        ref = sitk.ReadImage(str(ref_path))
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref)
        sitk.WriteImage(img, str(path))
    else:
        np.save(str(path.with_suffix('.npy')), arr)


# ============================================================================
#  nnUNetv2 post-processing hook
#  (call this from your inference script after nnUNetv2_predict)
# ============================================================================

def _postprocess_one_file(
    fpath: Path,
    opath: Path,
    et_min_size: int,
    tc_min_size: int,
    wt_min_size: int,
    drop_wt_without_tc: bool,
    wt_without_tc_max_size: int,
    region_ids: Optional[Dict[str, Tuple[int, ...]]],
    connectivity: int,
):
    seg  = _default_load(fpath)
    corr = postprocess_pred_array(seg, et_min_size, tc_min_size, wt_min_size,
                                  drop_wt_without_tc=drop_wt_without_tc,
                                  wt_without_tc_max_size=wt_without_tc_max_size,
                                  region_ids=region_ids,
                                  connectivity=connectivity)
    _default_save(corr, opath, fpath)


def apply_postprocess_to_folder(
    pred_dir: str | Path,
    out_dir:  str | Path,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
    suffix: str = '.nii.gz',
    drop_wt_without_tc: bool = False,
    wt_without_tc_max_size: int = 0,
    region_ids: Optional[Dict[str, Tuple[int, ...]]] = None,
    connectivity: int = 2,
    num_processes: Optional[int] = None,
):
    """
    Apply post-processing to every prediction file in pred_dir.
    Corrected files are written to out_dir (created if needed).

    Files are processed in parallel with a process pool (like
    remove_connected_components.py's apply_postprocessing_to_folder) instead
    of a serial per-file loop, since each file's ndimage.label calls are
    independent and CC labeling on full-resolution BraTS volumes dominates.

    Args:
        pred_dir    : folder produced by nnUNetv2_predict
        out_dir     : where to write corrected predictions
        et_min_size : ET CC size threshold (from threshold search)
        tc_min_size : TC CC size threshold (from threshold search)
        wt_min_size : WT CC size threshold
        suffix      : file extension to glob
        num_processes : worker processes; defaults to nnunetv2's
                        default_num_processes (falls back to cpu_count()).
    """
    pred_dir = Path(pred_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob(f'*{suffix}'))
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {pred_dir}")

    if num_processes is None:
        try:
            from nnunetv2.configuration import default_num_processes
            num_processes = default_num_processes
        except ImportError:
            num_processes = os.cpu_count() or 1

    extra = ''
    if drop_wt_without_tc:
        cap = f'≤{wt_without_tc_max_size}vox' if wt_without_tc_max_size > 0 else 'any size'
        extra = f'  drop TC-free WT CCs ({cap})'
    print(f"Post-processing {len(files)} files across {num_processes} processes: "
          f"ET≥{et_min_size}  TC≥{tc_min_size}  WT≥{wt_min_size}{extra}")

    import multiprocessing
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        pool.starmap(
            _postprocess_one_file,
            [(fpath, out_dir / fpath.name, et_min_size, tc_min_size, wt_min_size,
              drop_wt_without_tc, wt_without_tc_max_size, region_ids, connectivity)
             for fpath in files]
        )

    print(f"Done. Corrected files → {out_dir}")


# ============================================================================
#  CLI
# ============================================================================

def _build_file_pairs(pred_dir: Path, label_dir: Path, suffix='.nii.gz'):
    """Match prediction files to ground-truth by filename stem."""
    pred_files  = sorted(pred_dir.glob(f'*{suffix}'))
    gt_files    = []
    missing     = []
    for pf in pred_files:
        gt = label_dir / pf.name
        if gt.exists():
            gt_files.append(gt)
        else:
            missing.append(pf.name)

    if missing:
        warnings.warn(f"GT not found for: {missing}")
        pred_files = [p for p in pred_files
                      if (label_dir / p.name).exists()]

    return pred_files, gt_files


def main():
    parser = argparse.ArgumentParser(
        description='BraTS post-processing: hierarchy enforcement + CC filtering')
    parser.add_argument('--pred_dir',  required=True,
                        help='Folder with raw nnUNetv2_predict output (.nii.gz)')
    parser.add_argument('--out_dir',   required=True,
                        help='Where to write corrected predictions')
    parser.add_argument('--val_dir',   default=None,
                        help='Folder with validation ground-truth labels '
                             '(needed for --search_thresholds)')
    parser.add_argument('--threshold_file', default='brats_thresholds.json',
                        help='JSON file to save/load searched thresholds')
    parser.add_argument('--search_thresholds', action='store_true',
                        help='Run grid search over val set to find best CC '
                             'size thresholds. Requires --val_dir.')
    parser.add_argument('--et_min_size', type=int, default=20,
                        help='ET CC size threshold (ignored if --search_thresholds)')
    parser.add_argument('--tc_min_size', type=int, default=50,
                        help='TC CC size threshold (ignored if --search_thresholds)')
    parser.add_argument('--wt_min_size', type=int, default=100,
                        help='WT CC size threshold')
    parser.add_argument('--search_wt', action='store_true',
                        help='Also sweep wt_min_size during --search_thresholds '
                             '(otherwise WT is fixed at --wt_min_size and the '
                             'search cannot improve WT at all)')
    parser.add_argument('--suffix', default='.nii.gz',
                        help='File extension to glob (default: .nii.gz)')
    parser.add_argument('--drop_wt_without_tc', action='store_true',
                        help='Delete WT connected components that contain no TC '
                             'voxel (targets non-tumour FLAIR hyperintensity)')
    parser.add_argument('--wt_without_tc_max_size', type=int, default=0,
                        help='Only delete TC-free WT CCs at or below this many '
                             'voxels. 0 = no size limit.')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Worker processes (default: nnUNet default_num_processes)')
    parser.add_argument('--connectivity', type=int, default=2, choices=[1, 2, 3],
                        help='ndimage connectivity: in 3D 1=6-conn, 2=18-conn, '
                             '3=26-conn (default: 2)')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)

    if args.search_thresholds:
        if args.val_dir is None:
            raise ValueError("--val_dir required for --search_thresholds")
        val_dir = Path(args.val_dir)
        pred_files, gt_files = _build_file_pairs(pred_dir, val_dir, args.suffix)
        print(f"Found {len(pred_files)} matched prediction/GT pairs.")

        wt_grid = ([0, 50, 100, 200, 500, 1000, 2000]
                   if args.search_wt else None)
        results = search_thresholds(
            pred_files, gt_files,
            wt_min_size=args.wt_min_size,
            wt_min_sizes=wt_grid,
            connectivity=args.connectivity,
            num_processes=args.num_processes,
            # search the SAME pipeline we deploy below
            drop_wt_without_tc=args.drop_wt_without_tc,
            wt_without_tc_max_size=args.wt_without_tc_max_size,
        )

        tf = Path(args.threshold_file)
        with open(tf, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nThreshold results saved to {tf}")

        et_sz = results['best_et_min_size']
        tc_sz = results['best_tc_min_size']
        wt_sz = results['best_wt_min_size']
    else:
        # Load thresholds from file if it exists, else use CLI args
        tf = Path(args.threshold_file)
        if tf.exists():
            with open(tf) as f:
                saved = json.load(f)
            et_sz = saved['best_et_min_size']
            tc_sz = saved['best_tc_min_size']
            wt_sz = saved.get('best_wt_min_size', args.wt_min_size)
            print(f"Loaded thresholds from {tf}: "
                  f"ET≥{et_sz}  TC≥{tc_sz}  WT≥{wt_sz}")
        else:
            et_sz = args.et_min_size
            tc_sz = args.tc_min_size
            wt_sz = args.wt_min_size

    apply_postprocess_to_folder(
        pred_dir=pred_dir,
        out_dir=out_dir,
        et_min_size=et_sz,
        tc_min_size=tc_sz,
        wt_min_size=wt_sz,
        suffix=args.suffix,
        drop_wt_without_tc=args.drop_wt_without_tc,
        wt_without_tc_max_size=args.wt_without_tc_max_size,
        connectivity=args.connectivity,
        num_processes=args.num_processes,
    )


if __name__ == '__main__':
    main()