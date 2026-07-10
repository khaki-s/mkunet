"""
brats_postprocess.py
=====================
Post-processing for BraTS 2021 predictions that enforces the
label hierarchy  ET ⊂ TC ⊂ WT.

BraTS label convention used here (nnUNetv2 2D output):
    Channel 0 / class 0 → WT  (Whole Tumour,   includes all tumour)
    Channel 1 / class 1 → TC  (Tumour Core,    ET + NCR)
    Channel 2 / class 2 → ET  (Enhancing Tumour)

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
    Step 4 – Threshold search  (validation set only)
        Grid-search over {et_min_size, tc_min_size} using Dice on val labels.
        Saves best thresholds to a JSON file.
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


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 2,       # 2 = 8-conn in 2D, 1 = 4-conn
) -> np.ndarray:
    """
    Remove connected components smaller than `min_size` voxels.
    Works on 2D or 3D arrays.

    Args:
        mask        : binary array (bool or uint8)
        min_size    : minimum CC size to keep (voxels).
                      0 → no filtering.
        connectivity: passed to ndimage.label via structure
    Returns:
        Cleaned binary mask (uint8).
    """
    if min_size <= 0:
        return mask.astype(np.uint8)
    if not mask.any():
        return mask.astype(np.uint8)

    struct = ndimage.generate_binary_structure(mask.ndim, connectivity)
    labeled, n_cc = ndimage.label(mask, structure=struct)
    if n_cc == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    sizes = ndimage.sum(mask, labeled, range(1, n_cc + 1))
    keep  = np.array(sizes) >= min_size
    out   = np.isin(labeled, np.where(keep)[0] + 1).astype(np.uint8)
    return out


def postprocess_volume(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full post-processing pipeline for a single BraTS volume.

    Args:
        wt, tc, et   : predicted binary masks (H, W) or (D, H, W)
        et_min_size  : ET connected components smaller than this are removed
        tc_min_size  : TC connected components smaller than this are removed
        wt_min_size  : WT connected components smaller than this are removed

    Returns:
        Corrected (wt, tc, et) as uint8 arrays.
    """
    # ── Step 1: hierarchy enforcement ────────────────────────────────────────
    wt, tc, et = enforce_brats_hierarchy(wt, tc, et)

    # ── Step 2: remove small connected components ─────────────────────────────
    wt = remove_small_components(wt, wt_min_size)
    tc = remove_small_components(tc, tc_min_size)
    et = remove_small_components(et, et_min_size)

    # ── Step 3: second-pass hierarchy enforcement after CC filter ─────────────
    # (CC filter on WT may have removed WT voxels that TC/ET sat on)
    wt, tc, et = enforce_brats_hierarchy(wt, tc, et)

    return wt, tc, et


# ============================================================================
#  nnUNet-format label array helpers
# ============================================================================

def pred_to_masks(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert nnUNet BraTS segmentation array to (wt, tc, et) binary masks.

    nnUNet BraTS label convention:
        0 = background
        1 = NCR/NET  (part of TC, not ET)
        2 = ED       (oedema, part of WT but not TC)
        3 = ET       (enhancing tumour)

    Derived regions:
        WT = {1, 2, 3}
        TC = {1, 3}
        ET = {3}
    """
    wt = (seg > 0).astype(np.uint8)
    tc = np.isin(seg, [1, 3]).astype(np.uint8)
    et = (seg == 3).astype(np.uint8)
    return wt, tc, et


def masks_to_pred(
    wt: np.ndarray,
    tc: np.ndarray,
    et: np.ndarray,
    original_seg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reconstruct nnUNet BraTS label array from (wt, tc, et) binary masks.

    Mapping (priority order: ET > TC > ED > background):
        ET pixels (et=1, tc=1, wt=1) → label 3
        NC pixels (tc=1, et=0, wt=1) → label 1
        ED pixels (wt=1, tc=0, et=0) → label 2
        background (wt=0)             → label 0
    """
    out = np.zeros_like(wt, dtype=np.uint8)
    out[wt == 1] = 2     # ED (oedema)
    out[tc == 1] = 1     # NCR
    out[et == 1] = 3     # ET
    return out


def postprocess_pred_array(
    seg: np.ndarray,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
) -> np.ndarray:
    """
    Convenience wrapper: takes a nnUNet BraTS seg array, post-processes it,
    and returns a corrected seg array.
    """
    wt, tc, et = pred_to_masks(seg)
    wt, tc, et = postprocess_volume(wt, tc, et, et_min_size, tc_min_size, wt_min_size)
    return masks_to_pred(wt, tc, et, seg)


# ============================================================================
#  Dice metric (numpy)
# ============================================================================

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = (p & g).sum()
    return float(2 * inter / (p.sum() + g.sum() + eps))


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

def search_thresholds(
    pred_files: list[Path],
    gt_files: list[Path],
    et_min_sizes: list[int] = None,
    tc_min_sizes: list[int] = None,
    wt_min_size: int = 100,       # WT threshold: rarely needs tuning
    load_fn=None,
) -> Dict:
    """
    Grid search over (et_min_size, tc_min_size) to maximise mean(WT+TC+ET) Dice
    on the validation set.

    Args:
        pred_files    : list of prediction .nii.gz paths
        gt_files      : list of corresponding ground-truth paths (same order)
        et_min_sizes  : grid of ET CC size thresholds to try
        tc_min_sizes  : grid of TC CC size thresholds to try
        wt_min_size   : fixed WT CC size threshold (not searched by default)
        load_fn       : callable(path) → np.ndarray  (defaults to nib/sitk)

    Returns:
        dict with 'best_et_min_size', 'best_tc_min_size',
                  'best_dice_mean', 'grid_results'
    """
    if et_min_sizes is None:
        et_min_sizes = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    if tc_min_sizes is None:
        tc_min_sizes = [0, 10, 25, 50, 75, 100, 150, 200, 300]

    if load_fn is None:
        load_fn = _default_load

    print(f"Threshold search: {len(pred_files)} volumes, "
          f"{len(et_min_sizes)}×{len(tc_min_sizes)} grid "
          f"= {len(et_min_sizes)*len(tc_min_sizes)} combinations")

    # Pre-load all volumes once (avoids repeated disk I/O)
    preds = [load_fn(p) for p in pred_files]
    gts   = [load_fn(p) for p in gt_files]

    best_score   = -1.0
    best_et      = et_min_sizes[0]
    best_tc      = tc_min_sizes[0]
    grid_results = []

    for et_sz in et_min_sizes:
        for tc_sz in tc_min_sizes:
            dices_wt, dices_tc, dices_et = [], [], []
            for pred, gt in zip(preds, gts):
                corr = postprocess_pred_array(pred, et_sz, tc_sz, wt_min_size)
                d = brats_dice(corr, gt)
                dices_wt.append(d['WT'])
                dices_tc.append(d['TC'])
                dices_et.append(d['ET'])

            mean_wt = float(np.mean(dices_wt))
            mean_tc = float(np.mean(dices_tc))
            mean_et = float(np.mean(dices_et))
            mean_all = (mean_wt + mean_tc + mean_et) / 3.0

            grid_results.append({
                'et_min_size': et_sz,
                'tc_min_size': tc_sz,
                'wt_min_size': wt_min_size,
                'WT': mean_wt, 'TC': mean_tc, 'ET': mean_et,
                'mean': mean_all,
            })

            if mean_all > best_score:
                best_score = mean_all
                best_et    = et_sz
                best_tc    = tc_sz
                print(f"  → New best: ET_sz={et_sz:4d}  TC_sz={tc_sz:4d}  "
                      f"WT={mean_wt:.4f}  TC={mean_tc:.4f}  ET={mean_et:.4f}  "
                      f"mean={mean_all:.4f}")

    print(f"\nBest thresholds: et_min_size={best_et}, tc_min_size={best_tc}, "
          f"mean Dice={best_score:.4f}")

    return {
        'best_et_min_size': best_et,
        'best_tc_min_size': best_tc,
        'best_wt_min_size': wt_min_size,
        'best_dice_mean':   best_score,
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

def apply_postprocess_to_folder(
    pred_dir: str | Path,
    out_dir:  str | Path,
    et_min_size: int = 20,
    tc_min_size: int = 50,
    wt_min_size: int = 100,
    suffix: str = '.nii.gz',
):
    """
    Apply post-processing to every prediction file in pred_dir.
    Corrected files are written to out_dir (created if needed).

    Args:
        pred_dir    : folder produced by nnUNetv2_predict
        out_dir     : where to write corrected predictions
        et_min_size : ET CC size threshold (from threshold search)
        tc_min_size : TC CC size threshold (from threshold search)
        wt_min_size : WT CC size threshold
        suffix      : file extension to glob

    Example:
        apply_postprocess_to_folder(
            pred_dir='./raw_preds',
            out_dir='./corrected_preds',
            et_min_size=20,
            tc_min_size=50,
        )
    """
    pred_dir = Path(pred_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob(f'*{suffix}'))
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {pred_dir}")

    print(f"Post-processing {len(files)} files: "
          f"ET≥{et_min_size}  TC≥{tc_min_size}  WT≥{wt_min_size}")

    for fpath in files:
        seg   = _default_load(fpath)
        corr  = postprocess_pred_array(seg, et_min_size, tc_min_size, wt_min_size)
        opath = out_dir / fpath.name
        _default_save(corr, opath, fpath)

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
    parser.add_argument('--suffix', default='.nii.gz',
                        help='File extension to glob (default: .nii.gz)')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)

    if args.search_thresholds:
        if args.val_dir is None:
            raise ValueError("--val_dir required for --search_thresholds")
        val_dir = Path(args.val_dir)
        pred_files, gt_files = _build_file_pairs(pred_dir, val_dir, args.suffix)
        print(f"Found {len(pred_files)} matched prediction/GT pairs.")

        results = search_thresholds(pred_files, gt_files,
                                    wt_min_size=args.wt_min_size)

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
    )


if __name__ == '__main__':
    main()