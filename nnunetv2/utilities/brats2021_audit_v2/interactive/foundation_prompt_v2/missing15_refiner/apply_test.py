"""Apply an already frozen policy, then score the complete 100x15 test matrix."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from ..common import check_file, file_identity, read_json, read_labels, same_grid, write_json
from .dataset import HEADS, PROJECT, catalog, load_mask, split_manifest
from .infer import load_scores
from .select_val80 import bootstrap, load_frozen, window_from_scores, write_csv

OFFICIAL = Path("/data02/chujingyi/data/nnUNet_results/Dataset137_BraTS2021/"
                "nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300__nnUNetPlans__3d_fullres/"
                "fold_0/test100_15_modalities")
LABELS = (*HEADS, "ED", "NCR")


def verified_official(official, rows, model):
    matrix = read_json(official / "manifest.json")
    contract = matrix["contract"]
    checkpoint = model["fold_0/checkpoint_best.pth"]
    provenance = contract["provenance"]
    if (matrix["status"] != "complete" or contract["protocol"] != "available_files"
            or set(contract["case_ids"]) != set(rows) or set(contract["mask_codes"]) != set(range(1, 16))
            or not contract["mirroring"] or contract["empty_both_dice"] is not None
            or provenance["checkpoint_sha256"] != checkpoint["sha256"]
            or any(provenance["inputs"][cid] != row["modalities"] for cid, row in rows.items())):
        raise ValueError("Official nnU-Net test matrix is not paired to this deployment")
    return file_identity(official / "manifest.json")


def prediction_path(root, patient):
    return root / "test_predictions" / "receipts" / f"{patient}.json"


def save_prediction(path, labels, reference):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.nii.gz")
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labels, reference.affine, header), temporary)
    temporary.replace(path)
    return file_identity(path)


def predict_patient(task):
    root, official, patient, frozen, baseline_catalog, official_binding = task
    path = prediction_path(root, patient)
    binding = {"policy": file_identity(root / "frozen.json"), "official": official_binding, "gt_used": False}
    if path.exists():
        previous = read_json(path)
        if previous["binding"] != binding or set(previous["pairs"]) != {str(c) for c in range(1, 16)}:
            raise ValueError("Found incompatible test prediction")
        for row in previous["pairs"].values():
            check_file(row["baseline"])
            if row["result"] != row["baseline"]:
                check_file(row["result"])
        return patient
    predicted = load_scores(root, "test", patient, frozen["checkpoint"]) if baseline_catalog else {}
    if set(predicted) != set(baseline_catalog):
        raise ValueError(f"{patient}: active refinement jobs are incomplete")
    pairs = {}
    for code, policy in sorted(frozen["policies"].items()):
        official_path = official / "predictions" / f"mask_{code:02d}" / f"{patient}.nii.gz"
        baseline_identity = file_identity(official_path)
        result_identity = baseline_identity
        changes = 0
        if policy.rules:
            input_pair = baseline_catalog[code]
            info = predicted[code]
            if info["baseline"] != input_pair.mask or input_pair.baseline_model != frozen["baseline_model"]:
                raise ValueError("Test score and frozen baseline differ")
            baseline, reference = load_mask(baseline_identity)
            exported, exported_reference = load_mask(input_pair.mask)
            same_grid(reference, exported_reference)
            if not np.array_equal(baseline, exported):
                raise ValueError(f"{patient}/mask{code}: export differs from the official baseline")
            window = window_from_scores(baseline, info)
            if window is not None:
                corrected = window.apply_volume(baseline, policy)
                changes = int(np.count_nonzero(baseline != corrected))
                if changes:
                    target = root / "test_predictions" / f"mask_{code:02d}" / f"{patient}.nii.gz"
                    result_identity = save_prediction(target, corrected, reference)
        pairs[str(code)] = {"baseline": baseline_identity, "result": result_identity, "changed_voxels": changes}
    write_json(path, {"binding": binding, "pairs": pairs})
    return patient


def predict(args, frozen):
    root = args.output
    rows = {r["case_id"]: r for r in split_manifest(args.project, "test")}
    if len(rows) != 100 or set(rows) & set(frozen["validation_ids"]):
        raise ValueError("Expected 100 held-out test patients")
    official_binding = verified_official(args.official, rows, frozen["baseline_model"])
    codes = tuple(c for c, policy in frozen["policies"].items() if policy.rules)
    active = catalog(args.project, "test", codes=codes) if codes else {}
    jobs = [(root, args.official, cid, frozen, {p.code: p for p in active.get(cid, [])}, official_binding)
            for cid in sorted(rows)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, cid in enumerate(pool.map(predict_patient, jobs, chunksize=1), 1):
            if i % 10 == 0:
                print(f"test deployment {i}/100 {cid}", flush=True)


def bin_region(labels, index):
    return labels > index if index < 3 else labels == index - 2


def dice(prediction, reference):
    den = int(prediction.sum()) + int(reference.sum())
    return 200. * np.count_nonzero(prediction & reference) / den if den else 100.


def score_patient(task):
    root, patient, truth_identity, frozen, policy_identity, official_binding = task
    truth, ref = load_mask(truth_identity)
    receipt = read_json(prediction_path(root, patient))
    if (receipt["binding"] != {"policy": policy_identity, "official": official_binding, "gt_used": False}
            or set(receipt["pairs"]) != {str(c) for c in range(1, 16)}):
        raise ValueError(f"{patient}: missing or incompatible deployment receipt")
    truth_regions = [bin_region(truth, i) for i in range(5)]
    rows = []
    for code, files in receipt["pairs"].items():
        code = int(code)
        base, base_ref = load_mask(files["baseline"])
        same_grid(base_ref, ref)
        if files["result"] == files["baseline"]:
            final = base
        else:
            final, final_ref = load_mask(files["result"])
            same_grid(final_ref, ref)
        if int(np.count_nonzero(base != final)) != files["changed_voxels"]:
            raise ValueError(f"{patient} mask {code}: edit count changed")
        for i, region in enumerate(LABELS):
            b, g = bin_region(base, i), truth_regions[i]
            a = b if final is base else bin_region(final, i)
            if i < 3 and i not in frozen["policies"][code].heads and not np.array_equal(a, b):
                raise ValueError(f"{patient} mask {code}: inactive region changed")
            d0, d1 = dice(b, g), dice(a, g)
            rows.append({"case_id": patient, "mask_code": code, "region": region, "baseline_pct": d0,
                         "final_pct": d1, "gain_pp": d1 - d0,
                         "both_empty_baseline": not bool(b.any() or g.any()),
                         "added_voxels": int(np.count_nonzero(a & ~b)),
                         "removed_voxels": int(np.count_nonzero(b & ~a)), "empty_gt": not bool(g.any()),
                         "fp_empty_gt_voxels": int(a.sum()) if not g.any() else 0})
    return rows


def official_metrics(official):
    result = {}
    with (official / "per_case_metrics.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            value = row["dice"]
            result[row["case_id"], int(row["mask_code"]), row["region"]] = float(value) * 100. if value else None
    return result


def score(args, frozen):
    root = args.output
    complete = root / "test100_summary.json"
    if complete.exists():
        previous = read_json(complete)
        if previous["frozen_policy"] != file_identity(root / "frozen.json"):
            raise ValueError("Existing test metrics belong to a different frozen policy")
        for name in ("test100_summary.csv", "test100_per_case.csv"):
            check_file(previous["artifacts"][name])
        return
    deployment = {r["case_id"]: r for r in split_manifest(args.project, "test")}
    truths = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "test", truth=True)}
    if len(truths) != 100 or set(deployment) != set(truths):
        raise ValueError("Test score requires the complete 100-case whitelist")
    official_binding = verified_official(args.official, deployment, frozen["baseline_model"])
    ground_truth_reference = read_json(official_binding["path"])["contract"]["gt_sha256"]
    if any(truths[c]["sha256"] != ground_truth_reference[c] for c in truths):
        raise ValueError("Test GT differs from the official evaluation")
    policy_identity = file_identity(root / "frozen.json")
    results = []
    jobs = [(root, cid, truths[cid], frozen, policy_identity, official_binding) for cid in sorted(truths)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scores in pool.map(score_patient, jobs, chunksize=1):
            results.extend(scores)
    recorded = official_metrics(args.official)
    if len(results) != 7500 or len(recorded) != len(results):
        raise ValueError("Incomplete test metric matrix")
    for row in results:
        reference = recorded[row["case_id"], row["mask_code"], row["region"]]
        if (reference is None) != row["both_empty_baseline"] or (
            reference is not None and abs(row["baseline_pct"] - reference) > 5e-5
        ):
            raise ValueError("Recomputed baseline Dice differs from the official paired records")
        row["official_baseline_pct"] = reference
    summary = []
    for code in range(1, 16):
        for region in LABELS:
            group = [r for r in results if r["mask_code"] == code and r["region"] == region]
            stats = bootstrap([r["gain_pp"] for r in group])
            verified_baseline = [r["official_baseline_pct"] for r in group if r["official_baseline_pct"] is not None]
            summary.append({"mask_code": code, "region": region,
                            "enabled": region in [HEADS[h] for h in frozen["policies"][code].heads],
                            "baseline_pct": float(np.mean([r["baseline_pct"] for r in group])),
                            "final_pct": float(np.mean([r["final_pct"] for r in group])),
                            "official_baseline_pct": float(np.mean(verified_baseline)) if verified_baseline else None,
                            "added_voxels": sum(r["added_voxels"] for r in group),
                            "removed_voxels": sum(r["removed_voxels"] for r in group), **stats})
    write_csv(root / "test100_summary.csv", summary)
    write_csv(root / "test100_per_case.csv", results)
    changes = Counter(row["mask_code"] for row in results if row["region"] in HEADS and
                      (row["added_voxels"] or row["removed_voxels"]))
    write_json(complete, {"patients": 100, "combinations": 15, "case_combinations": 1500,
                          "region_rows": len(results), "frozen_policy": policy_identity,
                          "paired_dice_both_empty": 100, "official_dice_both_empty": None,
                          "official_baseline": official_binding,
                          "official_summary": file_identity(args.official / "summary.json"),
                          "number_of_changed_patient_region_units_by_mask": dict(changes),
                          "rows": summary, "artifacts": {name: file_identity(root / name)
                          for name in ("test100_summary.csv", "test100_per_case.csv")}})
    print(f"Scored 100 patients × 15 combinations: {complete}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("codes", "predict", "score"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--official", type=Path, default=OFFICIAL)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    args.output = args.output.resolve()
    frozen = load_frozen(args.output)
    if args.stage == "codes":
        print(" ".join(str(c) for c, policy in sorted(frozen["policies"].items()) if policy.rules))
    elif args.stage == "predict":
        predict(args, frozen)
    else:
        score(args, frozen)


if __name__ == "__main__":
    main()
