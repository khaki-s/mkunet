"""Offline prompt simulation and partial-label loss; no optimizer runs at inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, label

from .model import TorchPromptCorrector
from .prompts import roi_taper
from .schemas import REGIONS, PointPrompt, SpatialROI, StructuredAction
from .session import CorrectionSession


def verify_training_split(splits_file, fold, simulation_records=(), feedback_metadata=()):
    """Validate every input case against the authoritative fold before optimization.

    Metadata's self-declared ``split=train`` is insufficient. This function checks
    actual case membership in the supplied nnUNet splits_final.json train list.
    """
    split_path = Path(splits_file)
    encoded = split_path.read_bytes()
    splits = json.loads(encoded)
    if (not isinstance(splits, list) or type(fold) is not int
            or fold < 0 or fold >= len(splits)):
        raise ValueError("Invalid fold for nnUNet splits file")
    train = splits[fold].get("train")
    validation = splits[fold].get("val", [])
    if not isinstance(train, list) or not train or any(not isinstance(case, str) for case in train):
        raise ValueError("Fold must contain a nonempty training-case list")
    if not isinstance(validation, list) or set(train) & set(validation):
        raise ValueError("Fold train and validation cases must be disjoint")
    whitelist = set(train)
    for source, records in (("simulation", simulation_records), ("feedback", feedback_metadata)):
        for record in records:
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or case_id not in whitelist:
                raise ValueError(f"{source} case {case_id!r} is not in original fold {fold} train set")
            if record.get("split") != "train":
                raise ValueError(f"{source} case {case_id!r} is not declared as training data")
    return {
        "path": str(split_path.resolve()), "sha256": hashlib.sha256(encoded).hexdigest(),
        "fold": fold, "training_case_count": len(whitelist),
        "verified_simulation_cases": sorted({record["case_id"] for record in simulation_records}),
        "verified_feedback_cases": sorted({record["case_id"] for record in feedback_metadata}),
    }


def validate_simulation_identity(record):
    """Preflight every file's embedded identity before any randomized training step."""
    with np.load(record["path"], allow_pickle=False) as data:
        required = {"mri", "modality_mask", "previous_logits", "affine"}
        if not required.issubset(data.files) or not ({"region_gt", "atomic_labels"} & set(data.files)):
            raise ValueError(f"Simulation file lacks training fields: {record['path']}")
        if "split" in data and str(data["split"].item()) != record["split"]:
            raise ValueError("NPZ split conflicts with manifest; evaluation samples cannot enter training")
        if "case_id" in data and str(data["case_id"].item()) != record.get("case_id"):
            raise ValueError("NPZ case_id conflicts with the training manifest")


def sparse_region_bce(logits, labels, known):
    """Standard BCEWithLogits reduced ONLY over human/GT-confirmed region entries."""
    if logits.shape != labels.shape or logits.shape != known.shape:
        raise ValueError("logits, labels, and known must have the same shape")
    if known.dtype != torch.bool or not torch.isfinite(logits).all():
        raise ValueError("Known mask must be boolean and logits finite")
    if ((labels[known] != 0) & (labels[known] != 1)).any():
        raise ValueError("Known region labels must be binary")
    if not known.any():
        return logits.sum() * 0
    # Index BEFORE computing BCE; unlabeled NaNs/sentinels cannot contaminate loss.
    return F.binary_cross_entropy_with_logits(logits[known], labels[known], reduction="mean")


def labels_to_regions(atomic_labels):
    """Current Dataset137 internal convention: 0 BG, 1 ED, 2 NCR, 3 ET."""
    values = torch.as_tensor(atomic_labels)
    if not ((values == 0) | (values == 1) | (values == 2) | (values == 3)).all():
        raise ValueError("Expected internal labels 0=BG,1=ED,2=NCR,3=ET; convert raw BraTS first")
    return torch.stack((values > 0, values >= 2, values == 3)).float()


def simulate_error_click(probability, region_gt, target="ET", affine=None, seed=0):
    """Pick an interior FN positive or FP negative from the largest error component.

    This is an error-driven click simulator, not a claim of exact replication of
    nnInteractive's full training curriculum.
    """
    if target not in REGIONS:
        raise ValueError("target must be WT/TC/ET")
    pred = np.asarray(probability) > 0.5
    gt = np.asarray(region_gt)
    if pred.ndim != 3 or pred.shape != gt.shape or not np.isin(gt, [0, 1]).all():
        raise ValueError("Expected matching 3D probability and binary region GT")
    gt = gt.astype(bool)
    spacing = np.ones(3) if affine is None else np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    candidates = []
    for positive, errors in ((True, gt & ~pred), (False, pred & ~gt)):
        components, n = label(errors)
        if n:
            volumes = np.bincount(components.ravel())
            volumes[0] = 0
            component = int(volumes.argmax())
            candidates.append((int(volumes[component]), positive, components == component))
    if not candidates:
        return None
    _, positive, component = max(candidates, key=lambda item: item[0])
    # Pad so a component touching every crop edge still has a finite external boundary.
    distance = distance_transform_edt(np.pad(component, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    coordinates = np.argwhere(distance == distance.max())
    rng = np.random.default_rng(seed)
    voxel = tuple(int(v) for v in coordinates[rng.integers(len(coordinates))])
    return PointPrompt("simulated_click", voxel, target, positive)


def centered_roi(voxel, image_shape, roi_shape):
    roi_shape = tuple(min(int(r), int(n)) for r, n in zip(roi_shape, image_shape))
    if len(roi_shape) != 3 or any(r < 1 for r in roi_shape):
        raise ValueError("roi_shape must contain three positive lengths")
    start = tuple(max(0, min(int(v) - r // 2, int(n) - r))
                  for v, r, n in zip(voxel, roi_shape, image_shape))
    return SpatialROI(start, tuple(s + r for s, r in zip(start, roi_shape)))


def example_from_session(session, action, dense_region_gt=None):
    """Build an offline example before apply_action; dense GT is training-only."""
    model_input = session.build_model_input(action)
    roi = session.rois[action.roi_id]
    slices = (slice(None), *roi.slices)
    labels, known = session._action_constraints(action)
    labels, known = labels[slices], known[slices]
    update_heads = torch.zeros(3)
    if action.operation == "replace":
        update_heads[:] = 1
    else:
        update_heads[REGIONS.index(action.target)] = 1
    if dense_region_gt is not None:
        gt = torch.as_tensor(dense_region_gt, dtype=torch.float32)
        if gt.shape != session.logits.shape or not ((gt == 0) | (gt == 1)).all():
            raise ValueError("Dense region GT must match the image and contain 0/1")
        if (gt[1] > gt[0]).any() or (gt[2] > gt[1]).any():
            raise ValueError("Dense region GT must obey ET <= TC <= WT")
        labels = gt[slices].clone()
        # Supervise the selected output on its whole ROI from ACTUAL training GT.
        known = update_heads.bool().view(3, 1, 1, 1).expand_as(labels).clone()
    return {
        "model_input": model_input[0].cpu(),
        "previous_logits": session.logits[slices].clone(),
        "labels": labels.clone(),
        "known": known.clone(),
        "update_heads": update_heads,
        "taper": roi_taper(roi.shape, session.taper_voxels),
    }


def simulate_training_example(
    mri, modality_mask, previous_logits, region_gt, affine, target="ET", seed=0,
    roi_shape=(64, 64, 64), sigma_mm=3.0, taper_voxels=3,
):
    """Generate one dense-GT training crop. Never call this on test GT for tuning."""
    previous_logits = torch.as_tensor(previous_logits, dtype=torch.float32)
    region_gt = torch.as_tensor(region_gt, dtype=torch.float32)
    head = REGIONS.index(target)
    point = simulate_error_click(torch.sigmoid(previous_logits[head]).numpy(),
                                 region_gt[head].numpy(), target, affine, seed)
    if point is None:
        return None
    session = CorrectionSession(
        mri, modality_mask, previous_logits, affine, corrector=None, split="train",
        sigma_mm=sigma_mm, taper_voxels=taper_voxels,
    )
    roi = centered_roi(point.voxel, session.shape, roi_shape)
    session.register_roi("simulated_roi", roi)
    session.register_point(point)
    action = StructuredAction(
        target=target, operation="include" if point.positive else "exclude", roi_id="simulated_roi",
        positive_point_ids=(point.point_id,) if point.positive else (),
        negative_point_ids=() if point.positive else (point.point_id,),
    )
    return example_from_session(session, action, dense_region_gt=region_gt)


def offline_training_step(model, optimizer, example, device="cpu"):
    """One optimizer step on an explicit offline example; no hard click clamping."""
    if not example["known"].any():
        raise ValueError("No confirmed supervision in this example")
    model.train()
    model_input = example["model_input"].unsqueeze(0).to(device)
    previous = example["previous_logits"].unsqueeze(0).to(device)
    heads = example["update_heads"].to(device).view(1, 3, 1, 1, 1)
    taper = example["taper"].to(device).view(1, 1, *previous.shape[2:])
    prediction = previous + taper * heads * model(model_input)
    loss = sparse_region_bce(
        prediction, example["labels"].unsqueeze(0).to(device),
        example["known"].unsqueeze(0).to(device),
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite corrector loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0, error_if_nonfinite=True)
    optimizer.step()
    model.is_trained_checkpoint = True
    return float(loss.detach())
