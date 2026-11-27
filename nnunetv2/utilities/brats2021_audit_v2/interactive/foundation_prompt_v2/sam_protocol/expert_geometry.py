"""Select native interaction geometry from multi-slice SAM masks."""
import numpy as np
from scipy import ndimage

from ..sam_dense import prompt_geometry


def components(base, votes, head, spacing):
    """Consider both majority-supported changes and distinct WT discrepancies."""
    connectivity = ndimage.generate_binary_structure(3, 2)
    for positive, strength in ((True, votes), (False, 3 - votes)):
        eligible = ~base if positive else base
        strata = [(strength >= 2, 2 if head == 0 else 1)]
        if head == 0:
            strata.append((strength == 1, 1))
        for support, limit in strata:
            labels, count = ndimage.label(eligible & support, connectivity)
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            windows = ndimage.find_objects(labels)
            ranked = np.argsort(sizes[1:], kind='stable')[::-1][:limit] + 1
            for k in ranked:
                if sizes[k] < 24:
                    break
                window = windows[k-1]
                local = labels[window] == k
                depth = ndimage.distance_transform_edt(np.pad(local, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
                priority = depth * (.6 + .2 * strength[window])
                span = np.flatnonzero(local.any(axis=(0, 1)))
                central = int(np.argmax(priority.max(axis=(0, 1))))
                offset = min(span, key=lambda z: abs(abs(z-central)*spacing[2]-3.))
                planes = []
                component = labels == k
                origin = np.array([s.start for s in window])
                for z in sorted({central, int(offset)}):
                    xy = np.asarray(np.unravel_index(np.argmax(priority[..., z]), local.shape[:2]))
                    point = np.append(xy, z) + origin
                    planes.append(plane_prompt(base[..., point[2]], component[..., point[2]],
                                               point, positive, spacing[:2]))
                yield component, planes, positive, bool(limit == 1 and head == 0)


def plane_prompt(base, component, point, positive, spacing):
    coords = np.argwhere(component)
    pad = np.ceil(8. / spacing).astype(int)
    low = np.maximum(0, coords.min(axis=0) - pad)
    high = np.minimum(component.shape, coords.max(axis=0) + pad + 1)
    window = tuple(slice(a, b) for a, b in zip(low, high))
    seed = np.zeros_like(base)
    edited = base | component if positive else base & ~component
    seed[window] = edited[window]
    point_xy = point[[1, 0]].astype(np.float32)
    if seed.any():
        points, box = prompt_geometry(seed, padding=4, max_points=2)
        points = np.concatenate((points, point_xy[None]), axis=0)
        signs = np.asarray([1] * (len(points) - 1) + [int(positive)], np.int32)
    else:
        box = np.asarray([low[1], low[0], high[1] - 1, high[0] - 1], np.float32)
        points = point_xy[None]
        signs = np.asarray([int(positive)], np.int32)
    box = np.concatenate((np.minimum(box[:2], point_xy), np.maximum(box[2:], point_xy)))
    return int(point[2]), seed, points, signs, box


def native_prompt(mask, component, z, positive, spacing):
    """Choose the center and bounded box of SAM's signed evidence on one plane."""
    agreed = (mask == positive) & component
    if not agreed.any():
        return None, 0.
    depth = ndimage.distance_transform_edt(np.pad(agreed, 1), sampling=spacing)[1:-1, 1:-1]
    center = np.unravel_index(np.argmax(depth), agreed.shape)
    connected, _ = ndimage.label(agreed)
    coordinates = np.argwhere(connected == connected[center])
    low = np.maximum(np.maximum(coordinates.min(axis=0) - 1, 0), np.asarray(center) - 24)
    high = np.minimum(np.minimum(coordinates.max(axis=0) + 2, agreed.shape), np.asarray(center) + 25)
    prompt = {'point': [int(center[0]), int(center[1]), z], 'positive': positive,
              'box': [[int(a), int(b)] for a, b in zip(low, high)] + [[z, z + 1]]}
    return prompt, float(depth[center])
