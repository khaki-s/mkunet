#!/usr/bin/env python
"""Bind exported WT/TC/ET probabilities and available MRI to one native NIfTI grid."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes

from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for modality in ("t1", "t1ce", "t2", "flair"):
        parser.add_argument("--" + modality, type=Path)
    parser.add_argument("--probabilities", type=Path, required=True,
                        help="nnUNet EXPORTED full-volume .npz with probabilities [3,...]")
    parser.add_argument("--probability-axis-order", choices=("xyz", "zyx"), required=True,
                        help="SimpleITKIO nnUNet export is zyx; nibabel native grid is xyz")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "unassigned"), required=True)
    parser.add_argument("--training-labels", type=Path,
                        help="Optional internal-label GT for training simulation ONLY")
    parser.add_argument("--splits-file", type=Path, help="nnUNet splits_final.json; required with training GT")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    split_verification = None
    if args.split == "train":
        if args.splits_file is None:
            parser.error("--split train always requires --splits-file, including sparse-feedback cases")
        split_bytes = args.splits_file.read_bytes()
        splits = json.loads(split_bytes)
        if args.fold < 0 or args.fold >= len(splits) or args.case_id not in splits[args.fold]["train"]:
            parser.error("Case is not in the declared fold's original training set")
        split_verification = {"path": str(args.splits_file.resolve()), "fold": args.fold,
                              "sha256": hashlib.sha256(split_bytes).hexdigest()}
    paths = [getattr(args, x) for x in ("t1", "t1ce", "t2", "flair")]
    present = np.array([p is not None for p in paths], dtype=bool)
    if not present.any():
        parser.error("At least one MRI is required")
    ref = nib.load(str(next(p for p in paths if p is not None)))
    shape, affine = ref.shape, ref.affine
    if len(shape) != 3:
        parser.error("MRI must be 3D")
    raw = np.zeros((4, *shape), np.float32)
    for i, path in enumerate(paths):
        if path is not None:
            image = nib.load(str(path))
            if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-5, rtol=0):
                parser.error("MRI modalities are not on the same grid")
            raw[i] = image.get_fdata(dtype=np.float32)
    with np.load(args.probabilities, allow_pickle=False) as archive:
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    if probabilities.ndim != 4 or probabilities.shape[0] != 3:
        parser.error("Require three exported REGION probabilities WT,TC,ET, not four class probabilities")
    if args.probability_axis_order == "zyx":
        probabilities = probabilities.transpose(0, 3, 2, 1)
    if probabilities.shape != (3, *shape):
        parser.error("Probabilities must already be restored to the full native image grid")
    if (not np.isfinite(raw).all() or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        parser.error("Nonfinite MRI or invalid probabilities")
    # A fixed corrector-specific normalization, performed once per full case,
    # NOT per click/ROI. Keep the raw modality for the official nnInteractive API.
    foreground = binary_fill_holes(np.any(raw != 0, axis=0))
    if not foreground.any():
        parser.error("MRI foreground is empty")
    norm_seg = np.where(foreground, 0, -1).astype(np.int8)
    mri = np.zeros_like(raw)
    for i in np.flatnonzero(present):
        mri[i] = ZScoreNormalization(use_mask_for_norm=True, intensityproperties={}).run(raw[i].copy(), norm_seg)
    p = np.clip(probabilities, 1e-5, 1 - 1e-5)
    logits = np.log(p) - np.log1p(-p)
    training = {}
    if args.training_labels:
        if args.split != "train" or args.splits_file is None:
            parser.error("Training GT requires --split train and --splits-file")
        label_image = nib.load(str(args.training_labels))
        if label_image.shape != shape or not np.allclose(label_image.affine, affine, atol=1e-5, rtol=0):
            parser.error("Training labels have a different grid")
        labels = label_image.get_fdata()
        if not np.isin(labels, [0,1,2,3]).all():
            parser.error("Expected INTERNAL labels 0=BG,1=ED,2=NCR,3=ET")
        training = {"atomic_labels": labels.astype(np.uint8), "previous_logits": logits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"axis_order": "nibabel_native_xyz", "source_probability_order": args.probability_axis_order,
                "normalization": "case_nonzero_union_hole_filled_zscore", "modalities": ["T1","T1ce","T2","FLAIR"],
                "regions": ["WT","TC","ET"], "source_mri": [str(p.resolve()) if p else None for p in paths],
                "source_probabilities": str(args.probabilities.resolve())}
    if split_verification:
        metadata["training_split_verification"] = split_verification
    if args.training_labels:
        metadata["training_labels"] = str(args.training_labels.resolve())
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, mri=mri, raw_mri=raw, logits=logits, affine=affine,
                            modality_mask=present, case_id=np.array(args.case_id),
                            split=np.array(args.split), metadata=np.array(json.dumps(metadata)), **training)
    print(json.dumps({"output": str(args.output), "shape": shape, "mask": present.tolist()}))


if __name__ == "__main__":
    main()
