"""Physical-space prompts and nested-region constraints; no prediction is treated as GT."""
import numpy as np
import torch
from .schemas import REGIONS


def validate_affine(affine):
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be finite 4x4 array-voxel to world-mm transform")
    if not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("Invalid/singular affine")
    return affine.copy()


def world_to_voxel(world_mm, affine, shape):
    """Nearest voxel, rejecting out-of-grid world positions rather than clipping."""
    point = np.asarray(world_mm, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("world_mm must contain three finite coordinates")
    coord = (np.linalg.inv(validate_affine(affine)) @ np.r_[point, 1])[:3]
    if np.any(coord < 0) or np.any(coord > np.asarray(shape) - 1):
        raise ValueError("World point lies outside this image grid")
    return tuple(int(v) for v in np.floor(coord + 0.5))


def gaussian_from_points(shape, voxels, affine, sigma_mm=3.0, device=None):
    """Gaussian heatmap using the full affine metric, including oblique/sheared axes."""
    if not np.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("sigma_mm must be positive")
    affine = validate_affine(affine)
    result = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    if not voxels:
        return result
    coordinates = torch.stack(torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32, device=device) for n in shape),
        indexing="ij",
    ), dim=0)
    transform = torch.as_tensor(affine[:3, :3], dtype=torch.float32, device=device)
    for point in voxels:
        delta = coordinates - torch.tensor(point, dtype=torch.float32, device=device).view(3, 1, 1, 1)
        world_delta = torch.einsum("ij,jxyz->ixyz", transform, delta)
        gaussian = torch.exp(-world_delta.square().sum(0) / (2 * sigma_mm ** 2))
        # Finite support avoids declaring every voxel a prompt through tiny tails.
        gaussian = torch.where(gaussian >= np.exp(-4.5), gaussian, torch.zeros_like(gaussian))
        result = torch.maximum(result, gaussian)
    return result


def roi_taper(shape, width=3, device=None):
    """Smooth transition entirely *inside* ROI; external voxels are never written."""
    if type(width) is not int or width < 0:
        raise ValueError("taper width must be a nonnegative integer")
    output = torch.ones(tuple(shape), dtype=torch.float32, device=device)
    if width:
        for axis, length in enumerate(shape):
            coords = torch.arange(length, dtype=torch.float32, device=device)
            distance = torch.minimum(coords + 1, length - coords)
            fraction = (distance / (width + 1)).clamp(max=1)
            weight = 0.5 - 0.5 * torch.cos(torch.pi * fraction)
            view = [1, 1, 1]
            view[axis] = length
            output = output * weight.view(view)
    return output


def constraint_bounds(known, labels, margin=4.59511985013459):
    """Known labels propagate only necessary logical bounds ET <= TC <= WT."""
    if known.shape != labels.shape or known.shape[0] != 3 or known.dtype != torch.bool:
        raise ValueError("known/labels must have matching [3,D,H,W] shapes")
    if not torch.isfinite(labels).all() or ((labels[known] != 0) & (labels[known] != 1)).any():
        raise ValueError("Confirmed labels must be binary")
    lower = torch.full_like(labels, -torch.inf)
    upper = torch.full_like(labels, torch.inf)
    for head in range(3):
        positive = known[head] & (labels[head] == 1)
        negative = known[head] & (labels[head] == 0)
        for parent in range(head + 1):
            lower[parent][positive] = margin
        for child in range(head, 3):
            upper[child][negative] = -margin
    if (lower > upper).any():
        raise ValueError("Conflicting confirmed labels violate ET <= TC <= WT")
    return lower, upper


def project_nested_logits(logits, target, known=None, labels=None, operation=None):
    """Preserve requested region preferentially; constrain only the supplied crop.

    ET negative does not assign BG: WT/TC remain free. An explicit ET positive
    necessarily implies TC/WT positive. Confirmed sparse labels supply bounds.
    """
    if logits.ndim != 4 or logits.shape[0] != 3 or target not in REGIONS:
        raise ValueError("Expected [3,D,H,W] logits and WT/TC/ET target")
    z = logits.clone()
    if operation == "exclude" and target == "ET":
        # Preserve WT/TC exactly: "not ET" does not mean "not tumor".
        z[2] = torch.minimum(z[2], torch.minimum(z[1], z[0]))
    elif operation == "include" and target == "WT":
        z[0] = torch.maximum(z[0], torch.maximum(z[1], z[2]))
    elif target == "WT":
        z[1] = torch.minimum(z[1], z[0])
        z[2] = torch.minimum(z[2], z[1])
    elif target == "TC":
        z[0] = torch.maximum(z[0], z[1])
        z[2] = torch.minimum(z[2], z[1])
    else:
        z[1] = torch.maximum(z[1], z[2])
        z[0] = torch.maximum(z[0], z[1])
    if known is not None:
        lower, upper = constraint_bounds(known, labels)
        z = torch.maximum(lower, torch.minimum(upper, z))
    return z
