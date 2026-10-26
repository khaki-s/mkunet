"""Export sparse accepted feedback with provenance; predicted expansion is never GT."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from .schemas import StructuredAction
from .training import example_from_session


def save_feedback(session, directory, confirmed_by_user=False):
    """Save the latest round plus its pre-correction probabilities for offline replay.

    `confirmed_by_user` confirms these marked points, NOT every voxel of the model
    output. Test/validation feedback retains its split and the training loader
    rejects it. Reassigning data to training is an explicit dataset-design change.
    """
    if not session.history:
        raise ValueError("No correction round to export")
    if type(confirmed_by_user) is not bool:
        raise ValueError("confirmed_by_user must be explicit boolean")
    directory = Path(directory)
    if (directory / "feedback.npz").exists() or (directory / "feedback.json").exists():
        raise FileExistsError("Feedback files already exist; use a new directory for each accepted round")
    directory.mkdir(parents=True, exist_ok=True)
    action = StructuredAction.from_dict(session.history[-1]["action"])
    example = example_from_session(session, action)
    # Session now contains corrected logits. Restore the actual *input* state in
    # this exported training example, without mutating the live session.
    previous = session._undo_stack[-1]["logits"]
    example["previous_logits"] = previous.clone()
    example["model_input"][8:11] = torch.sigmoid(previous)
    roi = session.rois[action.roi_id]
    crop_affine = session.affine.copy()
    crop_affine[:3, 3] = (session.affine @ np.r_[roi.start, 1])[:3]
    arrays = {key: value.detach().cpu().numpy() for key, value in example.items()}
    arrays["affine"] = crop_affine
    arrays["corrected_logits_for_review_only"] = session.logits[(slice(None), *roi.slices)].numpy()
    np.savez_compressed(directory / "feedback.npz", **arrays)
    metadata = {
        "format": "brats_sparse_feedback_v1",
        "case_id": session.case_id, "split": session.split,
        "confirmed_by_user": confirmed_by_user,
        "supervision": "Only labels[known] are confirmed; predictions are review-only",
        "action": action.to_dict(), "roi": asdict(roi),
        "modality_mask": session.modality_mask.int().tolist(),
        "history": session.history, "source_grid_affine": session.affine.tolist(),
        "source_provenance": session.source_provenance,
        "region_order": ["WT", "TC", "ET"],
        "known_entries": int(example["known"].sum()),
        "all_region_voxel_entries": int(example["known"].numel()),
    }
    (directory / "feedback.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def load_feedback(directory, for_training=True):
    directory = Path(directory)
    metadata = json.loads((directory / "feedback.json").read_text())
    if metadata.get("format") != "brats_sparse_feedback_v1":
        raise ValueError("Unsupported feedback format")
    if for_training:
        if metadata.get("split") != "train":
            raise ValueError("Feedback from validation/test/unassigned cases cannot enter training")
        if metadata.get("confirmed_by_user") is not True:
            raise ValueError("Feedback must be explicitly confirmed by the human user")
    keys = ("model_input", "previous_logits", "labels", "known", "update_heads", "taper")
    with np.load(directory / "feedback.npz", allow_pickle=False) as data:
        example = {key: torch.from_numpy(data[key].copy()) for key in keys}
    if example["known"].dtype != torch.bool or not example["known"].any():
        raise ValueError("Feedback has no sparse binary confirmation mask; boxes alone are not dense GT")
    return example, metadata
