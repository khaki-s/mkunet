#!/usr/bin/env python
"""Frozen ET prompt evaluation, using a trained corrector or official nnInteractive.

Without --replay-dir, GT is used ONLY by the existing error-click simulator.
With --replay-dir, replay the first run's fixed points/text (also for Qwen).
GT is never passed to a corrector or a language model. Validation is the
default. Explicit test evaluation additionally requires a disjoint test list
and a completed validation protocol. Training cases are always rejected.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import (
    file_sha256, score_case, write_csv, write_json,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import RuleController, LLMController
from nnunetv2.utilities.brats2021_audit_v2.interactive.model import load_corrector
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import PointPrompt, SpatialROI
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.training import centered_roi, simulate_error_click


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Available-file manifest selecting evaluation IDs")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-manifest", type=Path, help="Authoritative held-out test case manifest")
    parser.add_argument("--protocol-from", type=Path, help="Completed validation manifest; required for test")
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--corrector-checkpoint", type=Path)
    backend.add_argument("--nninteractive-model", type=Path)
    parser.add_argument("--nninteractive-modality", type=int, choices=range(4))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--click-budgets", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--roi-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", choices=("rule", "qwen-text"), default="rule")
    parser.add_argument("--llm-path", type=Path)
    parser.add_argument("--llm-device", default="cpu")
    parser.add_argument("--replay-dir", type=Path, help="Previous evaluation output; fixes points/text across methods")
    args = parser.parse_args()
    budgets = sorted(set(args.click_budgets))
    if not budgets or min(budgets) < 1 or any(x < 1 for x in args.roi_size):
        parser.error("Click budgets and ROI dimensions must be positive")
    if args.output.exists():
        parser.error("Use a new output directory; existing results are never overwritten")
    if args.controller != "rule" and (args.llm_path is None or args.replay_dir is None):
        parser.error("Qwen evaluation requires --llm-path and --replay-dir to fix the comparison prompts")
    if args.nninteractive_model and args.nninteractive_modality is None:
        parser.error("nnInteractive requires --nninteractive-modality (0=T1,1=T1ce,2=T2,3=FLAIR)")
    folds = json.loads(args.splits_file.read_text())
    if args.fold < 0 or args.fold >= len(folds):
        parser.error("Invalid fold")
    fold = folds[args.fold]
    allowed = set(fold["val"]) - set(fold["train"])
    if args.split == "test":
        if args.test_manifest is None or args.protocol_from is None:
            parser.error("Test requires --test-manifest and --protocol-from a completed validation run")
        test_rows = json.loads(args.test_manifest.read_text())["cases"]
        test_ids = [r["case_id"] for r in test_rows]
        if (not test_ids or len(set(test_ids)) != len(test_ids)
                or set(test_ids) & (set(fold["train"]) | set(fold["val"]))):
            parser.error("Test manifest must be unique and disjoint from original train/validation")
        allowed = set(test_ids)
    rows = json.loads(args.manifest.read_text())["cases"]
    ids = [r["case_id"] for r in rows]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= allowed:
        parser.error(f"Every unique manifest case must belong to the declared {args.split.upper()} set")
    for case_id in ids:
        if (not isinstance(case_id, str) or Path(case_id).name != case_id
                or case_id in (".", "..")):
            parser.error("Invalid case ID")
        for path in (args.prepared / (case_id + ".npz"), args.gt / (case_id + ".nii.gz")):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.replay_dir:
            path = args.replay_dir / "cases" / case_id / "prompt_sequence.json"
            if not path.is_file():
                raise FileNotFoundError(path)
    torch.set_num_threads(4)
    model, shared_engine = None, None
    if args.corrector_checkpoint:
        model = load_corrector(args.corrector_checkpoint, args.device)
        if not model.is_trained_checkpoint:
            parser.error("A trained corrector checkpoint is required")
        model.eval()
        model.requires_grad_(False)
        backend_identity = {"kind": "trained_corrector", "sha256": file_sha256(args.corrector_checkpoint)}
    else:
        folder = args.nninteractive_model.resolve()
        checkpoints = sorted(folder.rglob("*.pth"))
        if not checkpoints:
            parser.error("nnInteractive model directory contains no checkpoint")
        backend_identity = {
            "kind": "nnInteractive", "folder": str(folder), "modality": args.nninteractive_modality,
            "checkpoints": {str(p.relative_to(folder)): file_sha256(p) for p in checkpoints},
        }
    protocol = {
        "backend_identity": backend_identity, "controller": args.controller,
        "budgets": budgets, "roi_size": list(args.roi_size), "seed": args.seed,
        "llm_path": str(args.llm_path.resolve()) if args.llm_path else None,
    }
    if args.protocol_from:
        previous = json.loads(args.protocol_from.read_text())
        if previous.get("status") != "complete" or previous.get("split") != "validation":
            parser.error("--protocol-from must refer to a completed validation evaluation")
        mismatches = [key for key, value in protocol.items() if previous.get(key) != value]
        if mismatches:
            parser.error("Frozen validation settings differ: " + ", ".join(mismatches))
    controller = RuleController()
    if args.controller != "rule":
        from nnunetv2.utilities.brats2021_audit_v2.interactive.llm import LocalQwenGenerator
        controller = LLMController(LocalQwenGenerator(args.llm_path, device=args.llm_device))
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        **protocol, "status": "running", "protocol": "ET oracle-click evaluation, fixed weights",
        "split": args.split,
        "controller": args.controller, "case_ids": ids, "budgets": budgets,
        "roi_size": args.roi_size, "seed": args.seed, "fold": args.fold,
        "split_sha256": file_sha256(args.splits_file),
        "corrector_sha256": backend_identity.get("sha256"),
        "test_manifest_sha256": file_sha256(args.test_manifest) if args.test_manifest else None,
        "protocol_from": str(args.protocol_from.resolve()) if args.protocol_from else None,
        "replay_dir": str(args.replay_dir.resolve()) if args.replay_dir else None,
        "llm_path": str(args.llm_path.resolve()) if args.llm_path else None,
        "gt_used_for": "scoring only" if args.replay_dir else "simulated user clicks and scoring only",
        "model_parameters_updated": False,
    }
    write_json(args.output / "manifest.json", provenance)
    metrics = []
    for case_id in ids:
        path = args.prepared / (case_id + ".npz")
        case_hash = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            if (str(archive["case_id"].item()) != case_id
                    or str(archive["split"].item()) != args.split):
                raise ValueError(f"Prepared identity/split mismatch: {case_id}")
            if args.nninteractive_model:
                from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
                modality = args.nninteractive_modality
                if "raw_mri" not in archive or not bool(archive["modality_mask"][modality]):
                    raise ValueError(f"Selected raw modality is unavailable: {case_id}")
                model = NNInteractiveBackend(
                    args.nninteractive_model, archive["raw_mri"][modality:modality + 1],
                    archive["affine"], modality_index=modality, device=args.device,
                    inference_session=shared_engine)
                shared_engine = model.engine
                # The reused engine has already loaded the official weights.
                model.is_trained_checkpoint = True
            # GT fields, if any, are never read from prepared inputs.
            session = CorrectionSession(
                archive["mri"], archive["modality_mask"], archive["logits"],
                archive["affine"], model, case_id=case_id, split=args.split)
        gt_path = args.gt / (case_id + ".nii.gz")
        gt_image = nib.load(gt_path)
        gt = np.asanyarray(gt_image.dataobj)
        if (gt.shape != session.shape or not np.isin(gt, (0, 1, 2, 3)).all()
                or not np.allclose(gt_image.affine, session.affine, atol=1e-5, rtol=0)):
            raise ValueError(f"GT geometry/internal-label mismatch: {case_id}")
        code = sum((1 << i) for i, flag in enumerate(session.modality_mask) if bool(flag))
        case_dir = args.output / "cases" / case_id
        case_dir.mkdir(parents=True)
        sequence = {
            "case_id": case_id, "original_case": str(path.resolve()),
            "original_case_sha256": case_hash, "rois": {}, "points": [], "commands": [],
            "prompt_source": args.split + " GT simulator; not a human or autonomous LLM",
        }
        fixed = None
        if args.replay_dir:
            fixed = json.loads((args.replay_dir / "cases" / case_id / "prompt_sequence.json").read_text())
            if (fixed["case_id"] != case_id or fixed["original_case_sha256"] != case_hash
                    or int(fixed["evaluated_budget"]) < max(budgets)):
                raise ValueError(f"Replay case/initial state/budget mismatch: {case_id}")
        baseline = {}
        rounds = []

        def save_budget(budget):
            labels = np.zeros(session.shape, np.uint8)
            for head, value in enumerate((1, 2, 3)):
                labels[session.logits[head].numpy() > 0] = value
            destination = case_dir / f"clicks_{budget:02d}.nii.gz"
            nib.save(nib.Nifti1Image(labels, gt_image.affine), destination)
            scored = score_case(gt_path, destination, case_id, code)
            for row in scored:
                region = row["region"]
                if budget == 0:
                    baseline[region] = row["dice"]
                before = baseline[region]
                delta = row["dice"] - before if row["dice"] is not None and before is not None else None
                metrics.append({**row, "click_budget": budget, "actual_clicks": len(rounds),
                                "baseline_dice": before, "dice_delta": delta})
            specification = {**sequence, "evaluated_budget": budget, "actual_clicks": len(rounds)}
            write_json(case_dir / f"interactions_{budget:02d}.json", specification)

        save_budget(0)
        stopped = False
        seed_base = int(hashlib.sha256(f"{args.seed}:{case_id}".encode()).hexdigest()[:8], 16)
        for step in range(1, max(budgets) + 1):
            point = None
            if not stopped:
                if fixed is None:
                    generated = simulate_error_click(
                        torch.sigmoid(session.logits[2]).numpy(), gt == 3,
                        "ET", gt_image.affine, seed_base + step)
                    if generated is not None:
                        point = PointPrompt(f"click_{step}", generated.voxel, "ET", generated.positive)
                        roi_id = f"roi_{step}"
                        roi = centered_roi(point.voxel, session.shape, args.roi_size)
                        command = {
                            "text": "ET漏分了，补上这个正点附近的ET" if point.positive
                                    else "ET分多了，保留周围水肿",
                            "target": "ET", "roi_id": roi_id, "point_ids": [point.point_id],
                        }
                elif step <= len(fixed["commands"]):
                    command = fixed["commands"][step - 1]
                    if len(command["point_ids"]) != 1:
                        raise ValueError("Replay must contain one simulated point per click")
                    entry = next(p for p in fixed["points"] if p["point_id"] == command["point_ids"][0])
                    point = PointPrompt(**entry)
                    roi_id = command["roi_id"]
                    roi = SpatialROI(**fixed["rois"][roi_id])
                if point is None:
                    stopped = True
                else:
                    session.register_point(point)
                    session.register_roi(roi_id, roi)
                    action = controller.propose(command["text"], session, roi_id=roi_id,
                                                point_ids=tuple(command["point_ids"]), target="ET")
                    outcome = session.apply_action(action)
                    rounds.append(outcome)
                    sequence["points"].append(asdict(point))
                    sequence["rois"][roi_id] = asdict(roi)
                    sequence["commands"].append(command)
            if step in budgets:
                save_budget(step)
        write_json(case_dir / "prompt_sequence.json", {
            **sequence, "evaluated_budget": max(budgets), "actual_clicks": len(rounds)})
        write_json(case_dir / "rounds.json", rounds)
        write_csv(args.output / "per_case_metrics.csv", metrics)
        print(json.dumps({"case_id": case_id, "actual_clicks": len(rounds),
                          "completed_cases": len({r["case_id"] for r in metrics})}), flush=True)
    summary = []
    for budget in [0, *budgets]:
        for region in ("WT", "TC", "ET", "ED", "NCR"):
            selected = [r for r in metrics if r["click_budget"] == budget and r["region"] == region]
            summary.append({
                "click_budget": budget, "region": region, "n_cases": len(selected),
                "n_valid_dice": sum(r["dice"] is not None for r in selected),
                "mean_dice": average([r["dice"] for r in selected]),
                "mean_paired_delta": average([r["dice_delta"] for r in selected]),
                "n_worse": sum(r["dice_delta"] is not None and r["dice_delta"] < -1e-8 for r in selected),
                "mean_fp_mm3": average([r["fp_mm3"] for r in selected]),
                "mean_fn_mm3": average([r["fn_mm3"] for r in selected]),
                "mean_hd95_finite_mm": average([r["hd95_mm"] for r in selected]),
                "n_hd95_one_empty": sum(r["hd95_status"] == "one_empty" for r in selected),
                "n_hd95_both_empty": sum(r["hd95_status"] == "both_empty" for r in selected),
                "mean_actual_clicks": average([r["actual_clicks"] for r in selected]),
            })
    write_csv(args.output / "summary.csv", summary)
    write_json(args.output / "summary.json", summary)
    provenance["status"] = "complete"
    write_json(args.output / "manifest.json", provenance)


if __name__ == "__main__":
    main()
