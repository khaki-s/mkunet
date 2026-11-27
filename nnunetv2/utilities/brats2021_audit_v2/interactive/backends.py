"""Adapters that call an installed upstream library, without vendoring its model."""
from pathlib import Path

import numpy as np
import torch

from .schemas import SpatialROI
from .inference_runtime import new_session, configure_session, prediction_context


class NNInteractiveBackend:
    """One raw, available MRI channel -> one selected binary region at a time.

    The official checkpoint is NOT a four-modal/three-region model. Keep its
    native one-channel API and merge only changes to the selected target.
    raw_image and its affine must describe exactly the CorrectionSession grid.
    """

    def __init__(self, model_folder, raw_image, affine, *, modality_index,
                 device="cuda:0", inference_session=None, replay_box_history=False,
                 full_image_context=False, cudnn_benchmark=False):
        self.raw_image = np.asarray(raw_image, dtype=np.float32)
        self.affine = np.asarray(affine, dtype=np.float64)
        if self.raw_image.ndim != 4 or self.raw_image.shape[0] != 1:
            raise ValueError("Official nnInteractive backend requires ONE raw MRI [1,D,H,W]")
        if not np.isfinite(self.raw_image).all() or self.affine.shape != (4, 4):
            raise ValueError("Invalid image/affine")
        if type(modality_index) is not int or modality_index not in range(4):
            raise ValueError("Select modality_index 0=T1,1=T1ce,2=T2,3=FLAIR")
        self.modality_index = modality_index
        self.replay_box_history = bool(replay_box_history)
        self.full_image_context = bool(full_image_context)
        self.is_trained_checkpoint = inference_session is None
        if inference_session is None:
            folder = Path(model_folder).expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError("Supply a downloaded nnInteractive trained model folder")
            inference_session = new_session(folder, device, cudnn_benchmark=cudnn_benchmark)
        else:
            configure_session(inference_session, cudnn_benchmark=cudnn_benchmark)
        self.engine = inference_session
        # Inference mode disables autograd, but is independent of module.eval().
        # The installed official loader leaves this flag True; make the frozen
        # correction protocol explicit (also when reusing a loaded session).
        network = getattr(self.engine, "network", None)
        if isinstance(network, torch.nn.Module):
            network.eval()

    @torch.inference_mode()
    def predict_delta(self, model_input, *, session, action, roi):
        if action.operation == "replace":
            raise ValueError("nnInteractive baseline handles one binary target; use the trained "
                             "multi-region corrector for an explicit atomic replacement")
        if not bool(session.modality_mask[self.modality_index]):
            raise ValueError("Selected nnInteractive MRI modality is absent")
        if tuple(self.raw_image.shape[1:]) != tuple(session.logits.shape[1:]):
            raise ValueError("Raw image and correction state must be on the same grid")
        affine = session.affine.detach().cpu().numpy() if torch.is_tensor(session.affine) else session.affine
        if not np.allclose(self.affine, affine, atol=1e-5, rtol=0):
            raise ValueError("Raw image and correction state have different affine coordinates")
        # A model's image context and the region we allow it to edit are distinct.
        # Full-context mode lets the official backend perform its own cropping
        # and normalization on the raw volume; we still merge only the edit ROI.
        context = (SpatialROI((0, 0, 0), tuple(session.shape))
                   if self.full_image_context else roi)
        # Validate box geometry/capabilities before resetting the upstream engine.
        box_call = None
        if action.box_prompt:
            if action.box_id is None:
                raise ValueError("nnInteractive requires a registered box_id distinct from the 3D editing ROI")
            box = session.boxes[action.box_id]
            coordinates = [[a - r, b - r] for a, b, r in zip(box.start, box.stop, context.start)]
            if (not any(b - a == 1 for a, b in coordinates)
                    and not getattr(self.engine, "supported_interactions", {}).get("bbox3d", False)):
                raise ValueError("This nnInteractive checkpoint supports 2D slice boxes only; "
                                 "one box axis must have thickness 1 inside a 3D editing ROI")
            box_call = ("box", coordinates, box.positive)
        target = ("WT", "TC", "ET").index(action.target)
        crop = np.ascontiguousarray(self.raw_image[(slice(None), *context.slices)])
        old_logits = session.logits[(target, *context.slices)].detach().cpu()
        initial = (old_logits > 0).numpy().astype(np.uint8)
        output = torch.zeros(initial.shape, dtype=torch.uint8)
        self.engine.set_image(crop)
        self.engine.set_target_buffer(output)
        # Rebuild from current case state each turn: undo never leaves hidden
        # upstream interactions from a future turn in the next prediction.
        self.engine.add_initial_seg_interaction(initial, run_prediction=False)
        calls = []
        if self.replay_box_history:
            # Only replay previous, unmodified boxes for the same binary head
            # fully contained by this editing crop. Clipping would change the
            # meaning of a box. This opt-in preserves the old demo behavior.
            seen = set()
            for record in session.history:
                previous = record.get("action", {}).get("box_id")
                if previous is None or previous == action.box_id or previous in seen:
                    continue
                seen.add(previous)
                annotation = session.boxes[previous]
                if annotation.target == action.target and all(
                    r <= a < b <= s for a, b, r, s in
                    zip(annotation.start, annotation.stop, context.start, context.stop)
                ):
                    calls.append(("box", [[a-r, b-r] for a, b, r in
                                          zip(annotation.start, annotation.stop, context.start)],
                                  annotation.positive))
        current_calls = [box_call] if box_call is not None else []
        for point_id in action.positive_point_ids + action.negative_point_ids:
            point = session.points[point_id]
            coords = tuple(int(v) - int(s) for v, s in zip(point.voxel, context.start))
            current_calls.append(("point", coords, point.positive))
        current_coordinates = {entry[1] for entry in current_calls if entry[0] == "point"}
        prior_known = session.confirmed_mask[(target, *context.slices)]
        prior_labels = session.confirmed_labels[(target, *context.slices)]
        for coordinates in torch.nonzero(prior_known).tolist():
            coordinates = tuple(coordinates)
            if coordinates not in current_coordinates:
                calls.append(("point", coordinates, bool(prior_labels[coordinates])))
        # nnInteractive 2.6 centers prediction on the LAST added interaction.
        # Replay old annotations first, then submit the new prompt; otherwise
        # a stale click silently becomes the center of the next correction.
        calls.extend(current_calls)
        if not calls:
            raise ValueError("No interaction to send to nnInteractive")
        with prediction_context(self.engine):
            for i, (kind, coordinates, positive) in enumerate(calls):
                method = (self.engine.add_point_interaction if kind == "point"
                          else self.engine.add_bbox_interaction)
                method(coordinates, include_interaction=positive, run_prediction=i == len(calls) - 1)
        selected = output.bool()
        changed = selected != torch.from_numpy(initial).bool()
        proposed = torch.where(selected, torch.tensor(4.59512), torch.tensor(-4.59512))
        edit = tuple(slice(a-r, b-r) for a, b, r in zip(roi.start, roi.stop, context.start))
        delta = torch.zeros((1, 3, *roi.shape), device=model_input.device, dtype=model_input.dtype)
        delta[0, target] = torch.where(changed[edit], proposed[edit] - old_logits[edit], 0.).to(delta)
        return delta
