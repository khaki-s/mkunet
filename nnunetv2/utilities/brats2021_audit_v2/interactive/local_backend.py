"""Optional geometric safeguards around finite nnInteractive annotations.

This wrapper is not trained and does not guarantee Dice improvement. It limits
the upstream proposal using only the current prediction, MRI availability,
annotation coordinates, and image affine. Ground truth/error maps are not inputs.
"""
from __future__ import annotations

from numbers import Real

import numpy as np
import torch

from .backends import NNInteractiveBackend
from .schemas import REGIONS


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _affine_metric(affine):
    matrix = np.asarray(affine, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0)
            or abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12):
        raise ValueError('Local correction requires a finite invertible voxel-to-world affine')
    # G includes shear cross terms as well as anisotropic voxel sizes/rotation.
    return matrix[:3, :3].T @ matrix[:3, :3]


def _relative_grid(roi, origin):
    axes = np.ogrid[tuple(slice(start, stop) for start, stop in zip(roi.start, roi.stop))]
    return [axis.astype(np.float64) - float(value) for axis, value in zip(axes, origin)]


def _quadratic_distance(offsets, metric, shape):
    distance = np.zeros(shape, dtype=np.float64)
    for i in range(3):
        distance += metric[i, i] * offsets[i] ** 2
        for j in range(i + 1, 3):
            distance += 2 * metric[i, j] * offsets[i] * offsets[j]
    return np.maximum(distance, 0.0)


def _point_distance_squared(roi, point, metric):
    return _quadratic_distance(_relative_grid(roi, point), metric, roi.shape)


def _box_distance_squared(roi, box, metric):
    """Exact physical distance to a 2D rectangle of annotated voxel centres.

    Half-open annotation bounds [start, stop) represent a closed continuous
    rectangle from start to stop-1 at the annotated slice centre. Degenerate
    one-pixel widths are supported. For sheared affines, coordinate-wise clipping
    is not the closest physical point, so solve the bounded two-variable least
    squares problem using its interior and four edge candidates.
    """
    widths = np.array(box.stop) - np.array(box.start)
    slice_axes = np.flatnonzero(widths == 1)
    if not len(slice_axes):
        raise ValueError('Local box support requires a 2D slice rectangle, not a 3D editing ROI')
    normal = int(slice_axes[0])
    axis_a, axis_b = [axis for axis in range(3) if axis != normal]
    length_a, length_b = float(widths[axis_a] - 1), float(widths[axis_b] - 1)
    offsets = _relative_grid(roi, box.start)
    base = _quadratic_distance(offsets, metric, roi.shape)
    dot_a = sum(metric[axis_a, axis] * offsets[axis] for axis in range(3))
    dot_b = sum(metric[axis_b, axis] * offsets[axis] for axis in range(3))
    aa, ab, bb = metric[axis_a, axis_a], metric[axis_a, axis_b], metric[axis_b, axis_b]

    def objective(a, b):
        return np.maximum(base - 2 * a * dot_a - 2 * b * dot_b
                          + aa * a * a + 2 * ab * a * b + bb * b * b, 0.0)

    closest = np.full(roi.shape, np.inf, dtype=np.float64)
    for fixed_a in (0.0, length_a):
        b = np.clip((dot_b - ab * fixed_a) / bb, 0.0, length_b)
        closest = np.minimum(closest, objective(fixed_a, b))
    for fixed_b in (0.0, length_b):
        a = np.clip((dot_a - ab * fixed_b) / aa, 0.0, length_a)
        closest = np.minimum(closest, objective(a, fixed_b))
    determinant = aa * bb - ab * ab
    a = (bb * dot_a - ab * dot_b) / determinant
    b = (aa * dot_b - ab * dot_a) / determinant
    inside = (a >= 0) & (a <= length_a) & (b >= 0) & (b <= length_b)
    return np.where(inside, np.minimum(closest, objective(a, b)), closest)


class SpatiallyConstrainedNNInteractiveBackend(NNInteractiveBackend):
    """Filter proposals by direction, finite-prompt distance and source class.

    include permits positive delta only; exclude permits negative delta only.
    Source-class protection applies to the sphere around each current point,
    using that point's current predicted atomic label (BG/ED/NCR/ET). Supports
    from multiple points are united. A box has no known source class: its
    rectangle neighbourhood receives only spatial/directional filtering.

    Prior annotations can still be replayed by the parent and protected by the
    existing CorrectionSession constraints; only NEW annotations define this
    wrapper's allowed update support. The parent can read full image context,
    while returning an update cropped to the original editing ROI.
    """

    def __init__(self, model_folder, raw_image, affine, *, modality_index,
                 radius_mm=8.0, directional=True, same_source_class=True,
                 modality_by_head=None, full_image_context=True, **parent_kwargs):
        if (isinstance(radius_mm, bool) or not isinstance(radius_mm, Real)
                or not np.isfinite(float(radius_mm)) or radius_mm <= 0):
            raise ValueError('radius_mm must be finite and positive')
        if type(directional) is not bool or type(same_source_class) is not bool:
            raise ValueError('directional and same_source_class must be booleans')
        if type(full_image_context) is not bool:
            raise ValueError('full_image_context must be boolean')
        if modality_by_head is not None:
            if (not isinstance(modality_by_head, dict) or set(modality_by_head) != set(REGIONS)
                    or any(type(value) is not int or value not in range(4)
                           for value in modality_by_head.values())):
                raise ValueError('modality_by_head must map exactly WT/TC/ET to integer modality indices 0..3')
            modality_by_head = dict(modality_by_head)
        self.radius_mm = float(radius_mm)
        self.directional = directional
        self.same_source_class = same_source_class
        self.modality_by_head = modality_by_head
        self.default_modality_index = modality_index
        self.metric = _affine_metric(affine)
        self.last_constraints = None
        super().__init__(model_folder, raw_image, affine, modality_index=modality_index,
                         full_image_context=full_image_context, **parent_kwargs)

    def _allowed_support(self, session, action, roi):
        support = np.zeros(roi.shape, dtype=bool)
        radius_squared = self.radius_mm ** 2
        threshold = radius_squared + max(1e-8, radius_squared * 1e-8)
        current = _numpy(session.logits[(slice(None), *roi.slices)]) > 0
        labels = np.zeros(roi.shape, dtype=np.uint8)
        for head in range(3):
            labels[current[head]] = head + 1
        source_classes = []
        point_ids = action.positive_point_ids + action.negative_point_ids
        for point_id in point_ids:
            point = session.points[point_id]
            if not roi.contains(point.voxel):
                raise ValueError('Current point must be inside the editing ROI')
            local = tuple(value - start for value, start in zip(point.voxel, roi.start))
            source_class = int(labels[local])
            source_classes.append(source_class)
            nearby = _point_distance_squared(roi, point.voxel, self.metric) <= threshold
            if self.same_source_class:
                nearby &= labels == source_class
            support |= nearby
        if action.box_prompt:
            if action.box_id is None or action.box_id not in session.boxes:
                raise ValueError('Local box support requires an explicit registered box_id')
            box = session.boxes[action.box_id]
            if any(a < r or b > s for a, b, r, s in zip(box.start, box.stop, roi.start, roi.stop)):
                raise ValueError('Current box must be contained by the editing ROI')
            support |= _box_distance_squared(roi, box, self.metric) <= threshold
        if not point_ids and not action.box_prompt:
            raise ValueError('Local correction requires a current point or registered box')
        return support, source_classes

    @torch.inference_mode()
    def predict_delta(self, model_input, *, session, action, roi):
        if action.target not in REGIONS or action.operation not in ('include', 'exclude'):
            raise ValueError('Local nnInteractive supports include/exclude on WT/TC/ET')
        selected = (self.default_modality_index if self.modality_by_head is None
                    else self.modality_by_head[action.target])
        available = _numpy(session.modality_mask)
        if available.shape != (4,) or not bool(available[selected]):
            raise ValueError(f'Selected MRI modality {selected} for {action.target} is absent')
        raw = np.ascontiguousarray(_numpy(session.mri[selected:selected + 1]), dtype=np.float32)
        if raw.shape != (1, *session.logits.shape[1:]) or not np.isfinite(raw).all():
            raise ValueError('Selected session MRI must be finite and on the current prediction grid')
        # Apply modality choice before the upstream forward. No GT or error map
        # is read; only the explicit MRI availability flags allow this channel.
        self.modality_index = selected
        self.raw_image = raw
        support, source_classes = self._allowed_support(session, action, roi)
        proposal = super().predict_delta(model_input, session=session, action=action, roi=roi)
        proposal = torch.as_tensor(proposal)
        if proposal.shape != (1, 3, *roi.shape) or not torch.isfinite(proposal).all():
            raise ValueError('Parent backend must return finite ROI delta of shape [1,3,*ROI]')
        head = REGIONS.index(action.target)
        target_delta = proposal[0, head]
        accepted = target_delta
        if self.directional:
            accepted = accepted.clamp_min(0) if action.operation == 'include' else accepted.clamp_max(0)
        accepted = torch.where(torch.as_tensor(support, device=proposal.device), accepted, 0.0)
        filtered = torch.zeros_like(proposal)
        filtered[0, head] = accepted
        self.last_constraints = {
            'kind': 'untrained_geometric_proposal_filter',
            'radius_mm': self.radius_mm,
            'directional': self.directional,
            'same_source_class': self.same_source_class,
            'point_source_classes': source_classes,
            'box_source_class_filter': False,
            'box_distance_definition': 'continuous 2D rectangle between annotated voxel centres',
            'selected_modality_index': selected,
            'full_image_context': self.full_image_context,
            'target': action.target,
            'allowed_voxels': int(support.sum()),
            'proposed_nonzero_voxels': int(torch.count_nonzero(target_delta)),
            'accepted_nonzero_voxels': int(torch.count_nonzero(accepted)),
            'gt_used': False,
        }
        return filtered
