#!/usr/bin/env python
"""Offline training only. Never invoked by inference or an LLM controller.

Manifest schema:
{"simulation": [{"path": "/abs/train_case.npz", "split": "train",
                 "case_id": "BraTS2021_x"}],
 "feedback": ["/abs/accepted_train_case_feedback_directory"]}

Simulation NPZ: mri[4,D,H,W] normalized on the SAME grid as previous_logits;
modality_mask[4]; previous_logits[3,D,H,W]; affine[4,4];
region_gt[3,D,H,W] OR atomic_labels[D,H,W] (0 BG,1 ED,2 NCR,3 ET).
Only original training-set GT may be used here. A user-reviewed point produces
partial supervision; it never becomes a dense-label training example.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nnunetv2.utilities.brats2021_audit_v2.interactive.feedback import load_feedback
from nnunetv2.utilities.brats2021_audit_v2.interactive.model import (
    CorrectorConfig, TorchPromptCorrector, load_corrector, save_corrector,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import REGIONS, PointPrompt, StructuredAction
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.training import (
    centered_roi, example_from_session, labels_to_regions,
    offline_training_step, simulate_error_click, validate_simulation_identity, verify_training_split,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True,
                        help="Authoritative nnUNet splits_final.json; every input must be in fold train")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rounds-per-case", type=int, default=3)
    parser.add_argument("--roi-size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--features", type=int, nargs="+", default=(16, 32, 64))
    parser.add_argument("--targets", nargs="+", choices=REGIONS, default=list(REGIONS),
                        help="Regions sampled for error-driven training; use ET for an ET-only corrector")
    parser.add_argument("--max-delta", type=float,
                        help="Optional residual bound for a NEW model; old default is 8.0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialize-from", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.rounds_per_case < 1:
        parser.error("--steps and --rounds-per-case must be positive")
    if len(args.targets) != len(set(args.targets)):
        parser.error("--targets must not contain duplicate regions")
    if args.max_delta is not None and (not np.isfinite(args.max_delta) or args.max_delta <= 0):
        parser.error("--max-delta must be finite and positive")
    if args.initialize_from and args.max_delta is not None:
        parser.error("A resumed model uses its saved residual bound; omit --max-delta")
    if args.output.exists():
        parser.error("Output already exists; choose a new checkpoint path")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads(args.manifest.read_text())
    simulations = manifest.get("simulation", [])
    if any(row.get("split") != "train" for row in simulations):
        raise ValueError("Every simulation record must explicitly belong to the training split")
    feedback = [load_feedback(path, for_training=True) for path in manifest.get("feedback", [])]
    if not simulations and not feedback:
        raise ValueError("Need original training-set simulation cases and/or confirmed train feedback")
    # Complete ALL membership/file-identity checks before creating an optimizer.
    split_provenance = verify_training_split(
        args.splits_file, args.fold, simulations, [meta for _, meta in feedback])
    for row in simulations:
        validate_simulation_identity(row)
    device = torch.device(args.device)
    model = (load_corrector(args.initialize_from, device) if args.initialize_from else
             TorchPromptCorrector(CorrectorConfig(features=tuple(args.features),
                 max_delta=args.max_delta if args.max_delta is not None else 8.0)).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses = []
    no_progress = 0
    while len(losses) < args.steps:
        kind = "feedback" if feedback and (not simulations or rng.random() < 0.5) else "simulation"
        if kind == "feedback":
            example, meta = feedback[int(rng.integers(len(feedback)))]
            loss = offline_training_step(model, optimizer, example, device)
            losses.append(loss)
            print(json.dumps({"step": len(losses), "source": kind, "loss": loss}), flush=True)
            continue
        row = simulations[int(rng.integers(len(simulations)))]
        with np.load(row["path"], allow_pickle=False) as data:
            if "split" in data and str(data["split"].item()) != row["split"]:
                raise ValueError("NPZ split conflicts with manifest; evaluation samples cannot enter training")
            if "case_id" in data and str(data["case_id"].item()) != row.get("case_id"):
                raise ValueError("NPZ case_id conflicts with the training manifest")
            gt = (torch.from_numpy(data["region_gt"].copy()).float() if "region_gt" in data else
                  labels_to_regions(data["atomic_labels"]))
            session = CorrectionSession(
                data["mri"], data["modality_mask"], data["previous_logits"],
                data["affine"], model, case_id=row.get("case_id", Path(row["path"]).stem), split="train",
                source_provenance={"training_split_verification": split_provenance},
            )
        if gt.shape != session.logits.shape or not ((gt == 0) | (gt == 1)).all():
            raise ValueError("Simulation GT must match the prediction grid and contain binary regions")
        if (gt[1] > gt[0]).any() or (gt[2] > gt[1]).any():
            raise ValueError("Simulation GT must obey ET <= TC <= WT")
        completed_rounds = 0
        for round_index in range(args.rounds_per_case):
            point = None
            selected_heads = [REGIONS.index(target) for target in args.targets]
            for head in rng.permutation(selected_heads):
                target = REGIONS[int(head)]
                point = simulate_error_click(
                    torch.sigmoid(session.logits[head]).numpy(), gt[head].numpy(), target,
                    session.affine, int(rng.integers(2 ** 31)),
                )
                if point is not None:
                    break
            if point is None:
                break
            point = PointPrompt(f"round_{round_index}", point.voxel, point.target, point.positive)
            roi_id = f"roi_{round_index}"
            session.register_point(point)
            session.register_roi(roi_id, centered_roi(point.voxel, session.shape, args.roi_size))
            action = StructuredAction(
                target=point.target, operation="include" if point.positive else "exclude",
                roi_id=roi_id, positive_point_ids=(point.point_id,) if point.positive else (),
                negative_point_ids=() if point.positive else (point.point_id,),
            )
            example = example_from_session(session, action, dense_region_gt=gt)
            loss = offline_training_step(model, optimizer, example, device)
            losses.append(loss)
            completed_rounds += 1
            # Simulated clicks are training GT. Subsequent rounds see the actual
            # current correction and retained constraints, not the untouched base.
            outcome = session.apply_action(action)
            print(json.dumps({
                "step": len(losses), "source": kind, "case_id": session.case_id,
                "round": round_index + 1, "target": point.target, "loss": loss,
                "learned_prompt_satisfaction": outcome["prompt_satisfaction_before_constraints"],
            }), flush=True)
            if len(losses) >= args.steps:
                break
        no_progress = no_progress + 1 if completed_rounds == 0 else 0
        if no_progress >= max(20, len(simulations) * 2):
            raise RuntimeError("Repeatedly found no simulation errors; no optimization steps were taken")
    save_corrector(model, args.output, training_metadata={
        "manifest": str(args.manifest.resolve()), "seed": args.seed, "steps": len(losses),
        "rounds_per_case": args.rounds_per_case, "roi_size": list(args.roi_size),
        "simulation_targets": args.targets, "simulation_prompts": "error-driven points only",
        "max_delta": model.config.max_delta,
        "last_loss": losses[-1], "loss": "PyTorch BCEWithLogits only over known entries",
        "hard_click_constraints_used_in_training_loss": False,
        "training_split_verification": split_provenance,
    })
    print(f"Saved trained corrector: {args.output}")


if __name__ == "__main__":
    main()
