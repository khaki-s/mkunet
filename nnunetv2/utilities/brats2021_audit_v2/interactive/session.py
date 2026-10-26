"""Transactional, reversible inference on one already-preprocessed image grid."""
from dataclasses import asdict
import json
from time import perf_counter
import numpy as np
import torch
from torch import nn

from .prompts import (
    constraint_bounds, gaussian_from_points, project_nested_logits,
    roi_taper, validate_affine, world_to_voxel,
)
from .schemas import ATOMIC_REGIONS, REGIONS, BoxPrompt, PointPrompt, SpatialROI, StructuredAction


class CorrectionSession:
    def __init__(
        self, mri, modality_mask, logits, affine, corrector, case_id="case",
        split="unassigned", enforce_clicks=True, sigma_mm=3.0, taper_voxels=3,
        source_provenance=None,
    ):
        self.mri = torch.as_tensor(mri, dtype=torch.float32).detach().cpu().clone()
        self.logits = torch.as_tensor(logits, dtype=torch.float32).detach().cpu().clone()
        mask = torch.as_tensor(modality_mask)
        if mask.shape != (4,) or not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError("modality_mask must contain four binary flags, with at least one present")
        self.modality_mask = mask.bool().cpu()
        if self.mri.ndim != 4 or self.mri.shape[0] != 4:
            raise ValueError("mri must have shape [4,D,H,W] in T1,T1ce,T2,FLAIR order")
        if self.logits.shape != (3, *self.mri.shape[1:]):
            raise ValueError("logits must have shape [3,D,H,W] in WT,TC,ET order")
        if not torch.isfinite(self.mri).all() or not torch.isfinite(self.logits).all():
            raise ValueError("MRI and logits must be finite")
        if (self.mri[~self.modality_mask] != 0).any():
            raise ValueError("Missing MRI channels must be zero, using the explicit availability mask")
        self.affine = validate_affine(affine)
        self.corrector = corrector
        self.case_id = str(case_id)
        if split not in ("train", "validation", "test", "unassigned"):
            raise ValueError("split must be train, validation, test, or unassigned")
        self.split = split
        if source_provenance is not None and not isinstance(source_provenance, dict):
            raise ValueError("source_provenance must be a JSON-compatible dictionary")
        self.source_provenance = json.loads(json.dumps(source_provenance or {}, allow_nan=False))
        self.enforce_clicks = bool(enforce_clicks)
        if sigma_mm <= 0 or not np.isfinite(sigma_mm):
            raise ValueError("sigma_mm must be positive")
        if type(taper_voxels) is not int or taper_voxels < 0:
            raise ValueError("taper_voxels must be a nonnegative integer")
        self.sigma_mm, self.taper_voxels = float(sigma_mm), taper_voxels
        self.points = {}
        self.boxes = {}
        self.rois = {}
        self.confirmed_labels = torch.zeros_like(self.logits)
        self.confirmed_mask = torch.zeros_like(self.logits, dtype=torch.bool)
        self.history = []
        self._undo_stack = []
        self.initial_logits = self.logits.clone()

    @property
    def shape(self):
        return tuple(self.mri.shape[1:])

    def register_roi(self, roi_id, roi):
        if not isinstance(roi_id, str) or not roi_id or roi_id in self.rois:
            raise ValueError("ROI ID must be a new, nonempty string")
        if not isinstance(roi, SpatialROI):
            raise ValueError("roi must be a SpatialROI")
        roi.validate(self.shape)
        self.rois[roi_id] = roi
        return roi

    def register_box(self, box):
        if not isinstance(box, BoxPrompt):
            raise ValueError("box must be a BoxPrompt")
        if box.box_id in self.boxes:
            raise ValueError("Box ID already exists")
        box.bounds.validate(self.shape)
        self.boxes[box.box_id] = box
        return box

    def register_point(self, point):
        if not isinstance(point, PointPrompt):
            raise ValueError("point must be a PointPrompt")
        if point.point_id in self.points:
            raise ValueError("Point ID already exists")
        if any(p >= n for p, n in zip(point.voxel, self.shape)):
            raise ValueError("Point is outside current image grid")
        self.points[point.point_id] = point
        return point

    def register_world_point(self, point_id, world_mm, target, positive):
        voxel = world_to_voxel(world_mm, self.affine, self.shape)
        return self.register_point(PointPrompt(point_id, voxel, target, positive))

    def validate_action(self, action):
        if not isinstance(action, StructuredAction):
            raise ValueError("action must be a StructuredAction")
        if action.roi_id not in self.rois:
            raise ValueError("Action references an unknown ROI")
        roi = self.rois[action.roi_id]
        if action.box_id is not None:
            if action.box_id not in self.boxes:
                raise ValueError("Action references an unknown registered box")
            box = self.boxes[action.box_id]
            if box.target != action.target or box.positive != (action.operation == "include"):
                raise ValueError("Action target/polarity does not match the registered box")
            if any(a < r or b > s for a, b, r, s in zip(box.start, box.stop, roi.start, roi.stop)):
                raise ValueError("Annotated box must lie inside its permitted 3D editing ROI")
        if not (action.positive_point_ids or action.negative_point_ids or action.box_prompt):
            raise ValueError("Action must reference points or an explicitly positive/negative box prompt")
        if (action.operation == "include" and action.negative_point_ids
                and not action.positive_point_ids and not action.box_prompt):
            raise ValueError("include requires at least one positive point or a box")
        if (action.operation == "exclude" and action.positive_point_ids
                and not action.negative_point_ids and not action.box_prompt):
            raise ValueError("exclude requires at least one negative point or a box")
        positions = {}
        for positive, ids in ((True, action.positive_point_ids), (False, action.negative_point_ids)):
            for point_id in ids:
                if point_id not in self.points:
                    raise ValueError(f"Unknown registered point: {point_id}")
                point = self.points[point_id]
                if point.target != action.target or point.positive != positive:
                    raise ValueError("Action target/polarity does not match the registered human point")
                if action.operation == "replace":
                    replacement_bit = ATOMIC_REGIONS[action.replacement_class][REGIONS.index(action.target)]
                    if replacement_bit != int(point.positive):
                        raise ValueError("replacement_class conflicts with the registered point's target/polarity")
                if not roi.contains(point.voxel):
                    raise ValueError("Action point is outside its permitted ROI")
                if (action.operation != "replace" and point.voxel in positions
                        and positions[point.voxel] != positive):
                    raise ValueError("The same voxel cannot be both positive and negative in one action")
                positions[point.voxel] = positive
        # Validate on copies, so bad actions never modify the session.
        self._action_constraints(action)
        return action

    def _action_constraints(self, action):
        labels, known = self.confirmed_labels.clone(), self.confirmed_mask.clone()
        head = REGIONS.index(action.target)
        for positive, ids in ((True, action.positive_point_ids), (False, action.negative_point_ids)):
            for point_id in ids:
                point = self.points[point_id]
                if action.operation == "replace":
                    labels[(slice(None), *point.voxel)] = torch.tensor(
                        ATOMIC_REGIONS[action.replacement_class], dtype=labels.dtype)
                    known[(slice(None), *point.voxel)] = True
                else:
                    labels[(head, *point.voxel)] = float(positive)
                    known[(head, *point.voxel)] = True
        constraint_bounds(known, labels)
        return labels, known

    def build_model_input(self, action):
        """Return [1,21,*ROI] on CPU. Coordinates refer to the current array grid."""
        self.validate_action(action)
        roi = self.rois[action.roi_id]
        slices = (slice(None), *roi.slices)
        labels, known = self._action_constraints(action)
        local_known = known[slices]
        local_labels = labels[slices]
        positive, negative = [], []
        # Include confirmed earlier clicks within this ROI for multi-turn consistency.
        for head in range(3):
            pos = torch.nonzero(local_known[head] & (local_labels[head] == 1)).tolist()
            neg = torch.nonzero(local_known[head] & (local_labels[head] == 0)).tolist()
            positive.append(gaussian_from_points(roi.shape, pos, self.affine, self.sigma_mm))
            negative.append(gaussian_from_points(roi.shape, neg, self.affine, self.sigma_mm))
        target = torch.zeros((3, *roi.shape), dtype=torch.float32)
        target[REGIONS.index(action.target)] = 1
        box = torch.zeros((1, *roi.shape), dtype=torch.float32)
        if action.box_prompt:
            # Signed box conditions operation; it is NOT dense voxel supervision.
            sign = 1 if action.operation == "include" else -1
            if action.box_id is None:
                box.fill_(sign)  # Explicit legacy 3D box for custom backends.
            else:
                annotation = self.boxes[action.box_id]
                local_slices = tuple(slice(a - r, b - r) for a, b, r in
                                     zip(annotation.start, annotation.stop, roi.start))
                box[(0, *local_slices)] = sign
        availability = self.modality_mask.float().view(4, 1, 1, 1).expand(4, *roi.shape)
        return torch.cat((
            self.mri[slices], availability, torch.sigmoid(self.logits[slices]),
            target, torch.stack(positive), torch.stack(negative), box,
        ), dim=0).unsqueeze(0)

    def apply_action(self, action):
        self.validate_action(action)
        start_time = perf_counter()
        roi = self.rois[action.roi_id]
        slices = (slice(None), *roi.slices)
        model_input = self.build_model_input(action)
        was_training = None
        if isinstance(self.corrector, nn.Module):
            was_training = self.corrector.training
            self.corrector.eval()
            parameter = next(self.corrector.parameters(), None)
            if parameter is not None:
                model_input = model_input.to(parameter.device)
        try:
            with torch.no_grad():
                if hasattr(self.corrector, "predict_delta"):
                    delta = self.corrector.predict_delta(
                        model_input, session=self, action=action, roi=roi)
                else:
                    delta = self.corrector(model_input)
        finally:
            if was_training is not None:
                self.corrector.train(was_training)
        delta = torch.as_tensor(delta).detach().cpu()
        if delta.shape != (1, 3, *roi.shape) or not torch.isfinite(delta).all():
            raise ValueError("Corrector must return finite [1,3,*ROI] delta logits")
        delta = delta[0].float()
        if action.operation != "replace":
            head = REGIONS.index(action.target)
            # The task is target-specific, even when the corrector predicts all heads.
            keep = torch.zeros_like(delta)
            keep[head] = delta[head]
            delta = keep
        previous = self.logits[slices].clone()
        candidate = previous + roi_taper(roi.shape, self.taper_voxels) * delta
        labels, known = self._action_constraints(action)
        learned_labels = candidate > 0
        click_mask = known[slices]
        desired = labels[slices].bool()
        count = int(click_mask.sum())
        before_clamp = (float((learned_labels[click_mask] == desired[click_mask]).float().mean())
                        if count else None)
        candidate = project_nested_logits(
            candidate, action.target,
            known[slices] if self.enforce_clicks else None,
            labels[slices] if self.enforce_clicks else None,
            operation=action.operation,
        )
        if not torch.isfinite(candidate).all():
            raise ValueError("Correction produced nonfinite values")
        # Commit only after all inference, shape, constraints and finite checks succeed.
        self._undo_stack.append({
            "roi": roi, "logits": previous,
            "labels": self.confirmed_labels[slices].clone(),
            "known": self.confirmed_mask[slices].clone(),
        })
        self.logits[slices] = candidate
        self.confirmed_labels[slices] = labels[slices]
        self.confirmed_mask[slices] = known[slices]
        record = {
            "round": len(self.history) + 1, "action": action.to_dict(),
            "roi": asdict(roi), "elapsed_seconds": perf_counter() - start_time,
            "changed_voxels": int((candidate != previous).any(0).sum()),
            "confirmed_region_voxels": count,
            "prompt_satisfaction_before_constraints": before_clamp,
            "prompt_satisfaction_after_constraints": (
                float(((candidate > 0)[click_mask] == desired[click_mask]).float().mean())
                if count else None),
            "hard_click_constraints": self.enforce_clicks,
            "backend": type(self.corrector).__name__,
            "learned_corrector_checkpoint": bool(getattr(self.corrector, "is_trained_checkpoint", False)),
        }
        self.history.append(record)
        return dict(record)

    def undo(self):
        if not self._undo_stack:
            return False
        state = self._undo_stack.pop()
        slices = (slice(None), *state["roi"].slices)
        self.logits[slices] = state["logits"]
        self.confirmed_labels[slices] = state["labels"]
        self.confirmed_mask[slices] = state["known"]
        self.history.pop()
        return True
