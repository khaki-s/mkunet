"""Finite GT-oracle prompts for nested BraTS labels, never dense model inputs.

Internal labels are 0=BG, 1=ED, 2=NCR/NET (combined), 3=ET. A prompt is
selected from an atomic-class error, then translated into one WT/TC/ET
boundary correction. This gives the simulator the correct class transition;
it is more informative than an untyped binary ED-negative click and must be
reported as a hierarchical transition protocol.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

CLASS_NAMES = ('BG', 'ED', 'NCR', 'ET')
HEADS = ('WT', 'TC', 'ET')
TARGETS = {'ED': 1, 'ET': 3, 'NCR': 2}


def _validated_inputs(pred_labels, gt_labels, affine):
    pred, gt = np.asarray(pred_labels), np.asarray(gt_labels)
    if (pred.ndim != 3 or pred.shape != gt.shape or any(n < 1 for n in pred.shape)
            or not np.issubdtype(pred.dtype, np.integer)
            or not np.issubdtype(gt.dtype, np.integer)):
        raise ValueError('Prediction and GT must be matching nonempty 3D integer labels')
    if not np.isin(pred, (0, 1, 2, 3)).all() or not np.isin(gt, (0, 1, 2, 3)).all():
        raise ValueError('Expected internal labels 0=BG,1=ED,2=NCR,3=ET')
    affine = np.asarray(affine, dtype=np.float64)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], (0, 0, 0, 1), atol=1e-8, rtol=0)
            or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12):
        raise ValueError('Require a finite, invertible 4x4 voxel-to-world affine')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid voxel spacing')
    return pred, gt, spacing


def _largest_error(pred, gt, target, rng):
    """Reservoir-sample tied largest components without storing every mask."""
    best_size, tied_count, selected = 0, 0, None
    targets = tuple(TARGETS) if target == 'joint' else (target,)
    for source_target in targets:
        value = TARGETS[source_target]
        for error_kind in ('FN', 'FP'):
            errors = ((gt == value) & (pred != value) if error_kind == 'FN'
                      else (pred == value) & (gt != value))
            components, count = label(errors, structure=generate_binary_structure(3, 1))
            if count == 0:
                continue
            volumes = np.bincount(components.ravel())
            volumes[0] = 0
            local_max = int(volumes.max())
            if local_max < best_size:
                continue
            candidates = np.flatnonzero(volumes == local_max)
            if local_max > best_size:
                best_size, tied_count = local_max, 0
            chosen = int(rng.integers(tied_count + len(candidates)))
            if chosen >= tied_count:
                component_id = int(candidates[chosen - tied_count])
                selected = (source_target, error_kind, components == component_id)
            tied_count += len(candidates)
    return None if selected is None else (*selected, best_size)


def _cap_interval(start, stop, anchor, maximum):
    if stop - start <= maximum:
        return int(start), int(stop)
    new_start = max(start, min(anchor - maximum // 2, stop - maximum))
    return int(new_start), int(new_start + maximum)


def _negative_rectangle(rectangle, anchor, gt_positive):
    """Shrink a box to exclude GT-positive head pixels while retaining its anchor.

    No rectangle mask is returned to the predictor: only the finite bounds are
    an oracle annotation. The rule is deliberately conservative, not a claim
    of a globally maximum-area rectangle.
    """
    r0, r1, c0, c1 = rectangle
    ar, ac = anchor
    while gt_positive[r0:r1, c0:c1].any():
        bad = np.argwhere(gt_positive[r0:r1, c0:c1])[0]
        row, col = int(bad[0] + r0), int(bad[1] + c0)
        candidates = []
        if row < ar:
            candidates.append((row + 1, r1, c0, c1))
        if row > ar:
            candidates.append((r0, row, c0, c1))
        if col < ac:
            candidates.append((r0, r1, col + 1, c1))
        if col > ac:
            candidates.append((r0, r1, c0, col))
        if not candidates:
            raise ValueError('Negative box anchor is GT-positive for its selected head')
        # Largest remaining area, then lexicographic bounds: deterministic.
        r0, r1, c0, c1 = min(candidates, key=lambda b: (-(b[1]-b[0])*(b[3]-b[2]), b))
    return int(r0), int(r1), int(c0), int(c1)


def select_prompt(pred_labels, gt_labels, affine, *, target, kind, seed,
                  box_axis=2, box_padding=2, box_max_extent=64):
    """Return a JSON-compatible finite prompt, or None when target errors vanish.

    Point: one interior voxel with a nested-region head and +/- polarity.
    Box: one 2D slice rectangle, half-open native-array bounds. No point is
    implied or sent by a box request; ``anchor`` is simulator provenance only.
    Point_box: the same selected error produces both that interior point and
    the box below, sharing one head and polarity; selection runs only once.
    Boxes use GT geometry and are a different information budget from points.
    Negative boxes are shrunk until they contain no positive GT pixel for the
    requested nested head; positive boxes are localization, not dense labels.
    """
    if target not in (*TARGETS, 'joint'):
        raise ValueError('target must be ED, ET, NCR, or joint')
    if kind not in ('point', 'box', 'point_box'):
        raise ValueError('kind must be point, box, or point_box')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(box_axis) is not int or box_axis not in range(3):
        raise ValueError('box_axis must be 0,1,2 in native voxel-array coordinates')
    if type(box_padding) is not int or box_padding < 0:
        raise ValueError('box_padding must be a nonnegative integer')
    if type(box_max_extent) is not int or box_max_extent < 1:
        raise ValueError('box_max_extent must be a positive integer')
    pred, gt, spacing = _validated_inputs(pred_labels, gt_labels, affine)
    rng = np.random.default_rng(seed)
    selection = _largest_error(pred, gt, target, rng)
    if selection is None:
        return None
    source_target, error_kind, component, size = selection
    distance = distance_transform_edt(np.pad(component, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    centers = np.argwhere(distance == distance.max())
    voxel = tuple(int(v) for v in centers[int(rng.integers(len(centers)))])
    old_class, new_class = int(pred[voxel]), int(gt[voxel])
    if old_class == new_class:
        raise AssertionError('The chosen prompt must refer to a current error')
    positive = new_class > old_class
    head_index = new_class - 1 if positive else new_class
    result = {
        'kind': kind, 'requested_target': target, 'source_target': source_target,
        'error_kind': error_kind, 'error_component_voxels': int(size),
        'anchor': list(voxel), 'pred_class': old_class, 'gt_class': new_class,
        'head': HEADS[head_index], 'positive': bool(positive),
        'operation': 'include' if positive else 'exclude',
        'transition': f'{CLASS_NAMES[old_class]}->{CLASS_NAMES[new_class]}',
        'prompt_protocol': 'gt_oracle_hierarchical_transition_v1',
    }
    if kind in ('point', 'point_box'):
        result['voxel'] = list(voxel)
    if kind == 'point':
        return result

    plane_pred = np.take(pred, voxel[box_axis], axis=box_axis)
    plane_gt = np.take(gt, voxel[box_axis], axis=box_axis)
    plane_component = np.take(component, voxel[box_axis], axis=box_axis)
    plane_axes = [axis for axis in range(3) if axis != box_axis]
    anchor2d = tuple(voxel[axis] for axis in plane_axes)
    transition = plane_component & (plane_pred == old_class) & (plane_gt == new_class)
    objects, _ = label(transition, structure=generate_binary_structure(2, 1))
    obj = int(objects[anchor2d])
    if obj == 0:
        raise AssertionError('Box anchor must belong to the selected transition')
    coordinates = np.argwhere(objects == obj)
    bounds = []
    clipped = False
    for dim in range(2):
        lower = max(0, int(coordinates[:, dim].min()) - box_padding)
        upper = min(plane_pred.shape[dim], int(coordinates[:, dim].max()) + 1 + box_padding)
        lo, hi = _cap_interval(lower, upper, anchor2d[dim], box_max_extent)
        clipped |= lo != lower or hi != upper
        bounds.append((lo, hi))
    rectangle = (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
    before_shrink = rectangle
    if not positive:
        rectangle = _negative_rectangle(rectangle, anchor2d, plane_gt > head_index)
    r0, r1, c0, c1 = rectangle
    start, stop = list(voxel), [v + 1 for v in voxel]
    for axis, lo, hi in zip(plane_axes, (r0, c0), (r1, c1)):
        start[axis], stop[axis] = int(lo), int(hi)
    result.update({
        'box_start': start, 'box_stop': stop, 'box_axis': int(box_axis),
        'box_padding': int(box_padding), 'box_max_extent': int(box_max_extent),
        'box_area_voxels': int((r1-r0)*(c1-c0)),
        'box_degenerate': bool(r1-r0 == 1 or c1-c0 == 1),
        'box_extent_clipped': bool(clipped),
        'box_negative_shrunk': bool(rectangle != before_shrink),
        'negative_box_gt_pure': None if positive else True,
        'gt_geometry_used': True,
        'box_semantics': 'slice_localization_not_dense_supervision',
    })
    if kind == 'point_box':
        result.update({'pairing_protocol': 'same_error_same_anchor_same_head_same_polarity_v1',
                       'annotation_count': 2})
    return result
