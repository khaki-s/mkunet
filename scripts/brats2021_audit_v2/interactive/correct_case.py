#!/usr/bin/env python
"""Execute bounded multi-round human/text corrections, saving a new case version."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import BoxPrompt, PointPrompt, SpatialROI, StructuredAction
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.model import TorchPromptCorrector, load_corrector
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import RuleController, LLMController, action_from_json
from nnunetv2.utilities.brats2021_audit_v2.interactive.atomic import atomic_point_transaction


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="prepare_case.py NPZ, no GT required")
    parser.add_argument("--interactions", type=Path, required=True, help="Registered ROIs/points and ordered commands JSON")
    parser.add_argument("--output", type=Path, required=True, help="New directory; original predictions are never overwritten")
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument("--corrector-checkpoint", type=Path)
    backend.add_argument("--nninteractive-model", type=Path)
    backend.add_argument("--demo-untrained", action="store_true",
                         help="Interface test only: zero-init corrector + explicit click constraints, not learned correction")
    parser.add_argument("--nninteractive-modality", type=int, choices=range(4))
    parser.add_argument("--controller", choices=("rule", "qwen-text", "qwen-vl"), default="rule")
    parser.add_argument("--llm-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--llm-device", default="cpu")
    parser.add_argument("--accept-feedback", action="store_true",
                        help="Explicitly confirm marked sparse labels for offline feedback, not the expanded prediction")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    specification = json.loads(args.interactions.read_text())
    case_path, case_sha = str(args.case.resolve()), sha256_file(args.case)
    if "original_case" in specification and specification["original_case"] != case_path:
        parser.error("Replay must use its ORIGINAL case NPZ, not another case or corrected state")
    if "original_case_sha256" in specification and specification["original_case_sha256"] != case_sha:
        parser.error("Original case contents changed since the saved replay")
    torch.set_num_threads(4)
    with np.load(args.case, allow_pickle=False) as data:
        tensors = {key: np.array(data[key]) for key in
                   ("mri", "logits", "affine", "modality_mask", "case_id", "split")}
        raw = np.array(data["raw_mri"]) if "raw_mri" in data else None
        provenance = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
    if "case_id" in specification and specification["case_id"] != str(tensors["case_id"].item()):
        parser.error("Replay belongs to a different case_id")
    if args.corrector_checkpoint:
        backend_identity = {"kind":"trained_corrector", "sha256":sha256_file(args.corrector_checkpoint)}
    elif args.nninteractive_model:
        folder = args.nninteractive_model.resolve()
        backend_identity = {"kind":"nnInteractive", "folder":str(folder),
                            "modality":args.nninteractive_modality,
                            "checkpoints":{str(p.relative_to(folder)):sha256_file(p)
                                           for p in sorted(folder.rglob("*.pth"))}}
    else:
        backend_identity = {"kind":"zero_initialized_demo", "config":"default_v1"}
    if "backend_identity" in specification and specification["backend_identity"] != backend_identity:
        parser.error("Replay backend differs; preserve the original backend/checkpoint for exact undo/replay")
    if args.nninteractive_model:
        from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
        if args.nninteractive_modality is None or raw is None:
            parser.error("nnInteractive requires --nninteractive-modality and raw_mri in case NPZ")
        corrector = NNInteractiveBackend(
            args.nninteractive_model, raw[args.nninteractive_modality:args.nninteractive_modality+1],
            tensors["affine"], modality_index=args.nninteractive_modality, device=args.device)
    elif args.corrector_checkpoint:
        corrector = load_corrector(args.corrector_checkpoint, args.device)
        if not corrector.is_trained_checkpoint:
            parser.error("Checkpoint is marked untrained; use --demo-untrained only for interface checks")
    else:
        corrector = TorchPromptCorrector().to(args.device).eval()
    session = CorrectionSession(
        tensors["mri"], tensors["modality_mask"], tensors["logits"], tensors["affine"],
        corrector, case_id=str(tensors["case_id"].item()), split=str(tensors["split"].item()),
        source_provenance={**provenance, "original_case":case_path, "original_case_sha256":case_sha,
                           "backend_identity":backend_identity})
    for roi_id, value in specification.get("rois", {}).items():
        session.register_roi(roi_id, SpatialROI(tuple(value["start"]), tuple(value["stop"])))
    for value in specification.get("points", []):
        if "world_mm" in value:
            session.register_world_point(value["point_id"], value["world_mm"], value["target"], value["positive"])
        else:
            session.register_point(PointPrompt(value["point_id"], tuple(value["voxel"]),
                                               value["target"], value["positive"]))
    for value in specification.get("boxes", []):
        session.register_box(BoxPrompt(**value))
    if args.controller != "rule":
        if args.llm_path is None:
            parser.error("--llm-path must point to locally downloaded weights")
        from nnunetv2.utilities.brats2021_audit_v2.interactive.llm import LocalQwenGenerator
        controller = LLMController(LocalQwenGenerator(args.llm_path,
                                   vision=args.controller == "qwen-vl", device=args.llm_device))
    else:
        controller = RuleController()
    commands = specification.get("commands", [])
    if not commands:
        parser.error("Provide at least one correction or undo command")
    operations, replay_commands = [], []
    for command in commands:
        if command.get("undo") is True:
            operations.append({"undo": session.undo()})
            replay_commands.append({"undo": True})
            continue
        if "atomic_assignment" in command:
            assignment = command["atomic_assignment"]
            normalized = []
            with atomic_point_transaction(session, assignment["voxel"], assignment["class_name"]):
                for raw_action in command["actions"]:
                    action = action_from_json(json.dumps(raw_action), session)
                    operations.append(session.apply_action(action))
                    normalized.append(action.to_dict())
            replay_commands.append({"atomic_assignment": assignment, "actions": normalized,
                                    "human_prompt_count": 1, "visual_tool_calls": 2})
            continue
        if "action" in command:
            action = action_from_json(json.dumps(command["action"]), session)
        else:
            kwargs = dict(roi_id=command["roi_id"], point_ids=tuple(command.get("point_ids", ())),
                          target=command.get("target"), box_prompt=command.get("box_prompt", False),
                          box_id=command.get("box_id"))
            if isinstance(controller, LLMController):
                kwargs["images"] = tuple(command.get("images", ()))
            action = controller.propose(command["text"], session, **kwargs)
        operations.append(session.apply_action(action))
        replay_commands.append({"action": action.to_dict()})
    if args.accept_feedback and not session.history:
        parser.error("No remaining correction round to confirm after undo")
    args.output.mkdir(parents=True, exist_ok=False)
    probabilities = torch.sigmoid(session.logits).numpy()
    labels = np.zeros(session.shape, np.uint8)
    for head, label in enumerate((1, 2, 3)):
        labels[probabilities[head] > .5] = label
    nib.save(nib.Nifti1Image(labels, session.affine), str(args.output / "corrected_internal_labels.nii.gz"))
    np.savez_compressed(args.output / "state.npz", mri=session.mri.numpy(), raw_mri=raw if raw is not None else np.empty(0),
                        logits=session.logits.numpy(), probabilities=probabilities, affine=session.affine,
                        modality_mask=session.modality_mask.numpy(), case_id=np.array(session.case_id),
                        split=np.array(session.split))
    report = {"case_id": session.case_id, "split": session.split, "demo_untrained": args.demo_untrained,
              "controller": args.controller, "original_case": str(args.case.resolve()),
              "original_case_sha256":case_sha, "backend_identity":backend_identity,
              "operations": operations, "remaining_history": session.history,
              "feedback_confirmed_by_user": args.accept_feedback,
              "scope": "Current-case inference only; no model parameters updated"}
    (args.output / "session.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Replay the same normalized actions against the ORIGINAL case to continue
    # across processes, retaining clicks and undo without repeating LLM decisions.
    replay = {"rois": specification.get("rois", {}), "points": specification.get("points", []),
              "boxes": specification.get("boxes", []),
              "commands": replay_commands, "original_case": case_path, "original_case_sha256":case_sha,
              "case_id":session.case_id, "backend_identity":backend_identity}
    (args.output / "replay.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n")
    if args.accept_feedback:
        from nnunetv2.utilities.brats2021_audit_v2.interactive.feedback import save_feedback
        save_feedback(session, args.output / "feedback", confirmed_by_user=True)
    print(json.dumps({"output": str(args.output), "rounds": len(session.history),
                      "demo_untrained": args.demo_untrained}, ensure_ascii=False))


if __name__ == "__main__":
    main()
