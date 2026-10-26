"""Adapters that call an installed upstream library, without vendoring its model."""
from pathlib import Path

import numpy as np
import torch


class NNInteractiveBackend:
    """One raw, available MRI channel -> one selected binary region at a time.

    The official checkpoint is NOT a four-modal/three-region model. Keep its
    native one-channel API and merge only changes to the selected target.
    raw_image and its affine must describe exactly the CorrectionSession grid.
    """

    def __init__(self, model_folder, raw_image, affine, *, modality_index,
                 device="cuda:0", inference_session=None):
        self.raw_image = np.asarray(raw_image, dtype=np.float32)
        self.affine = np.asarray(affine, dtype=np.float64)
        if self.raw_image.ndim != 4 or self.raw_image.shape[0] != 1:
            raise ValueError("Official nnInteractive backend requires ONE raw MRI [1,D,H,W]")
        if not np.isfinite(self.raw_image).all() or self.affine.shape != (4, 4):
            raise ValueError("Invalid image/affine")
        if type(modality_index) is not int or modality_index not in range(4):
            raise ValueError("Select modality_index 0=T1,1=T1ce,2=T2,3=FLAIR")
        self.modality_index = modality_index
        self.is_trained_checkpoint = inference_session is None
        if inference_session is None:
            folder = Path(model_folder).expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError("Supply a downloaded nnInteractive trained model folder")
            try:
                from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            except ImportError as exc:
                raise ImportError("nnInteractive is optional; install the official package in a "
                                  "compatible separate environment and supply local weights") from exc
            inference_session = nnInteractiveInferenceSession(
                device=torch.device(device), use_torch_compile=False,
                verbose=False, torch_n_threads=4, do_autozoom=True)
            inference_session.initialize_from_trained_model_folder(str(folder))
        self.engine = inference_session

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
        # Validate box geometry/capabilities before resetting the upstream engine.
        box_call = None
        if action.box_prompt:
            if action.box_id is None:
                raise ValueError("nnInteractive requires a registered box_id distinct from the 3D editing ROI")
            box = session.boxes[action.box_id]
            coordinates = [[a - r, b - r] for a, b, r in zip(box.start, box.stop, roi.start)]
            if (not any(b - a == 1 for a, b in coordinates)
                    and not getattr(self.engine, "supported_interactions", {}).get("bbox3d", False)):
                raise ValueError("This nnInteractive checkpoint supports 2D slice boxes only; "
                                 "one box axis must have thickness 1 inside a 3D editing ROI")
            box_call = ("box", coordinates, box.positive)
        target = ("WT", "TC", "ET").index(action.target)
        crop = np.ascontiguousarray(self.raw_image[(slice(None), *roi.slices)])
        old_logits = session.logits[(target, *roi.slices)].detach().cpu()
        initial = (old_logits > 0).numpy().astype(np.uint8)
        output = torch.zeros(initial.shape, dtype=torch.uint8)
        self.engine.set_image(crop)
        self.engine.set_target_buffer(output)
        # Rebuild from current case state each turn: undo never leaves hidden
        # upstream interactions from a future turn in the next prediction.
        self.engine.add_initial_seg_interaction(initial, run_prediction=False)
        calls = [box_call] if box_call is not None else []
        for point_id in action.positive_point_ids + action.negative_point_ids:
            point = session.points[point_id]
            coords = tuple(int(v) - int(s) for v, s in zip(point.voxel, roi.start))
            calls.append(("point", coords, point.positive))
        current_coordinates = {entry[1] for entry in calls if entry[0] == "point"}
        prior_known = session.confirmed_mask[(target, *roi.slices)]
        prior_labels = session.confirmed_labels[(target, *roi.slices)]
        for coordinates in torch.nonzero(prior_known).tolist():
            coordinates = tuple(coordinates)
            if coordinates not in current_coordinates:
                calls.append(("point", coordinates, bool(prior_labels[coordinates])))
        if not calls:
            raise ValueError("No interaction to send to nnInteractive")
        for i, (kind, coordinates, positive) in enumerate(calls):
            method = (self.engine.add_point_interaction if kind == "point"
                      else self.engine.add_bbox_interaction)
            method(coordinates, include_interaction=positive, run_prediction=i == len(calls) - 1)
        selected = output.bool()
        changed = selected != torch.from_numpy(initial).bool()
        proposed = torch.where(selected, torch.tensor(4.59512), torch.tensor(-4.59512))
        delta = torch.zeros((1, 3, *initial.shape), device=model_input.device, dtype=model_input.dtype)
        delta[0, target] = torch.where(changed, proposed - old_logits, 0.).to(delta)
        return delta
