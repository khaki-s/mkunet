"""Direction-supervised three-dimensional SAM disagreement prompts."""
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HEADS = ('WT', 'TC', 'ET')
GROUPS = tuple((h, positive) for h in range(3) for positive in (True, False))


def _select(mask):
    return tuple(slice(max(0, axis.min() - 10), min(n, axis.max() + 11))
                 for axis, n in zip(np.nonzero(mask), mask.shape))


def component_features(mri, baseline, masks, affine, *, connected=False):
    """Features use native voxels; local arrays only bound CPU feature work."""
    if masks.shape != (2, 3, *baseline.shape) or mri.shape != (4, *baseline.shape):
        raise ValueError('Expected two SAM variants and four MRI modalities on the original grid')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    heads = baseline[None] > np.arange(3)[:, None, None, None]
    difference = np.logical_xor(masks[0], heads)
    if connected:
        difference |= np.logical_xor(masks[1], heads)
    brain = np.any(mri != 0, axis=0)
    disagreement = difference.any(0) & brain
    output = {'shape': np.array(baseline.shape), 'affine': np.asarray(affine), 'connected': np.array(connected),
              'sizes': np.array([np.count_nonzero(baseline > h) for h in range(3)])}
    if not disagreement.any():
        return output

    local = _select(disagreement)
    origin = np.array([s.start for s in local])
    brain_local = brain[local]
    base = baseline[local]
    evidence = []
    for image in mri:
        sample = image[::2, ::2, ::2]
        sample = sample[sample != 0]
        if not len(sample):
            raise ValueError('Expected nonempty MRI modality')
        q10, q50, q90 = np.percentile(sample, (10, 50, 90))
        normalized = np.clip((image[local] - q50) / max(1e-6, q90 - q10), -4, 4)
        smoothed = ndimage.gaussian_filter(normalized, sigma=1.25 / spacing)
        neighborhood = ndimage.gaussian_filter(normalized, sigma=3.0 / spacing)
        evidence.append((normalized, smoothed, neighborhood))

    for group, (head, positive) in enumerate(GROUPS):
        prior = base > head
        sam = masks[(0, head, *local)]
        other = masks[(1, head, *local)]
        primary_seed = (sam == positive) & (prior != positive) & brain_local
        seed = primary_seed.copy()
        if connected:
            seed |= (other == positive) & (prior != positive) & brain_local
        # Keep the neighboring heads intact for each single-head edit.
        editable = (base == head + int(not positive)) & brain_local
        seed &= editable
        if not seed.any():
            continue
        if connected:
            seed_distance = ndimage.distance_transform_edt(~seed, sampling=spacing)
            change = (seed_distance <= 2.) & editable
        else:
            change = seed
        labels, _ = ndimage.label(change)  # Six-connectivity in the native 3D grid.
        coordinates = np.nonzero(change)
        ids = labels[coordinates]
        sizes = np.bincount(ids)
        objects = ndimage.find_objects(labels)
        span = np.array([0, *(box[2].stop - box[2].start for box in objects)], dtype=np.float32)
        depth = ndimage.distance_transform_edt(np.pad(change, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        base_side = np.pad(~prior if positive else prior, 1)
        if connected:
            distance, nearest = ndimage.distance_transform_edt(base_side, sampling=spacing, return_indices=True)
            at = tuple(axis + 1 for axis in coordinates)
            boundary = tuple(np.clip(nearest[(i, *at)] - 1, 0, base.shape[i] - 1) for i in range(3))
        else:
            distance = ndimage.distance_transform_edt(base_side, sampling=spacing)
        baseline_depth = distance[1:-1, 1:-1, 1:-1]
        local_prior = ndimage.uniform_filter(prior.astype(np.float32), size=5)
        local_sam = ndimage.uniform_filter(sam.astype(np.float32), size=5)
        native = tuple(c + offset for c, offset in zip(coordinates, origin))
        z = coordinates[2]
        preceding = change[coordinates[0], coordinates[1], np.maximum(0, z-1)] & (z > 0)
        following = change[coordinates[0], coordinates[1], np.minimum(change.shape[2]-1, z+1)] & (z+1 < change.shape[2])
        fields = [depth[coordinates], np.minimum(baseline_depth[coordinates], 25),
                  np.log1p(sizes[ids]), np.log1p(span[ids]), (other[coordinates] == positive),
                  preceding.astype(float) + following, local_prior[coordinates], local_sam[coordinates],
                  np.full(len(ids), float(positive)), np.full(len(ids), np.log1p(output['sizes'][head]))]
        if connected:
            fields.extend((primary_seed[coordinates], seed_distance[coordinates]))
        # Compare local intensity context with the subject's own predicted tissues.
        inner = prior & brain_local
        background = ~prior & brain_local
        for raw, smooth, wide in evidence:
            value = raw[coordinates]
            inner_center = np.median(raw[inner]) if inner.any() else 0.
            outer_center = np.median(raw[background]) if background.any() else 0.
            fields.extend((value, smooth[coordinates], wide[coordinates], value - smooth[coordinates],
                           value - inner_center, value - outer_center))
            if connected:
                fields.extend((value - raw[boundary], smooth[coordinates] - smooth[boundary]))
        flat = np.ravel_multi_index(native, baseline.shape).astype(np.int32)
        output[f'index_{group}'] = flat
        output[f'component_{group}'] = ids.astype(np.int32)
        output[f'features_{group}'] = np.column_stack(fields).astype(np.float32)
        if connected:
            output[f'seed_{group}'] = seed[coordinates]
    return output


def train_direction(samples, keys):
    """Fit one supervised point-direction and coverage estimator per region."""
    models = []
    for head in range(3):
        vectors, labels, weights = [], [], []
        for key in keys:
            with np.load(samples[key], allow_pickle=False) as data:
                for group in (head * 2, head * 2 + 1):
                    name = f'features_{group}'
                    if name not in data:
                        continue
                    x, y = data[name], data[f'target_{group}']
                    rng = np.random.default_rng(1976 + group)
                    chosen = rng.choice(len(y), min(6500, len(y)), replace=False)
                    vectors.append(x[chosen])
                    labels.append(y[chosen])
                    weights.append(np.full(len(chosen), np.sqrt(len(y)) / len(chosen)))
        if not vectors:
            models.append(0.)
            continue
        x, y, w = np.concatenate(vectors), np.concatenate(labels), np.concatenate(weights)
        if np.unique(y).size == 1:
            models.append(float(y[0]))
            continue
        model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.075, max_leaf_nodes=15,
                                               min_samples_leaf=45, l2_regularization=3., random_state=42)
        with threadpool_limits(limits=4):
            model.fit(x, y, sample_weight=w / w.mean())
        models.append(model)
    return models


def directional_probability(model, x):
    if isinstance(model, float):
        return np.full(len(x), model, dtype=np.float32)
    with threadpool_limits(limits=2):
        return model.predict_proba(x)[:, 1].astype(np.float32)


def _half_open_box(voxels, anchor, shape):
    # Bound the finite prompt around the anchor. The network still sees its
    # original 192³ patch and the backend still receives complete MRI context.
    lo = np.maximum(voxels.min(0) - 1, np.maximum(0, anchor - 11))
    hi = np.minimum(voxels.max(0) + 2, np.minimum(shape, anchor + 12))
    lo[2], hi[2] = anchor[2], anchor[2] + 1
    return lo.tolist(), hi.tolist()


def _local_probability(coords, probabilities):
    lo = coords.min(0)
    shape = tuple(coords.max(0) - lo + 1)
    centered = tuple((coords - lo).T)
    scores = np.zeros(shape, np.float32)
    support = np.zeros(shape, np.float32)
    support[centered] = 1
    scores[centered] = probabilities
    mean = ndimage.uniform_filter(scores, 5)[centered]
    count = ndimage.uniform_filter(support, 5)[centered]
    return mean / count.clip(1e-6)


def propose(data, models, *, per_group=None):
    """Rank 3D components and place distinct finite points in physical space."""
    affine, shape = data['affine'], tuple(data['shape'])
    connected = bool(data.get('connected', False))
    per_group = per_group if per_group is not None else (4 if connected else 3)
    for group, (head, positive) in enumerate(GROUPS):
        index_name = f'index_{group}'
        if index_name not in data:
            continue
        features = data[f'features_{group}']
        ids, flat = data[f'component_{group}'], data[index_name]
        probability = directional_probability(models[head], features)
        seeds = data[f'seed_{group}'] if connected else None
        order = np.argsort(ids, kind='stable')
        borders = np.flatnonzero(np.diff(ids[order])) + 1
        ranked = []
        for chunk in np.split(order, borders):
            if len(chunk) < 5:
                continue
            p = probability[chunk]
            if connected and not seeds[chunk].any():
                continue
            hot = np.maximum(p - .45, 0.)
            ranked.append((float(hot.sum() ** .5 + p.max()), int(ids[chunk[0]]), chunk))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        candidates = []
        for i, (_, part, chunk) in enumerate(ranked[:per_group]):
            coords = np.column_stack(np.unravel_index(flat[chunk], shape))
            p = probability[chunk]
            depth = features[chunk, 0]
            priority = p + .025 * np.minimum(depth, 4)
            if connected:
                priority = .45 * p + .55 * _local_probability(coords, p) + .02 * np.minimum(depth, 4)
                priority[~seeds[chunk]] = -np.inf
            # Place a second point in a different area of the strongest
            # component when it spans at least 8mm, without spending both
            # slots there if other large components are available.
            n = 2 if i == 0 and (connected or len(ranked) < per_group) else 1
            for _ in range(n):
                j = int(np.argmax(priority))
                if not np.isfinite(priority[j]):
                    break
                anchor = coords[j]
                include = (coords[:, 2] == anchor[2]) & (p >= min(p[j], max(.38, p[j] - .18)))
                if connected:
                    include &= seeds[chunk]
                footprint = coords[include]
                lo, hi = _half_open_box(footprint, anchor, shape)
                prompt = {'kind': 'point_box', 'head': HEADS[head], 'positive': positive,
                          'operation': 'include' if positive else 'exclude', 'voxel': anchor.tolist(),
                          'anchor': anchor.tolist(), 'box_start': lo, 'box_stop': hi, 'box_axis': 2,
                          'source': 'SAM3_3d_disagreement', 'component': part, 'group': group,
                          'component_voxels': len(chunk), 'component_slices': len(np.unique(coords[:, 2])),
                          'anchor_probability': float(p[j])}
                candidates.append((prompt, flat[chunk], p))
                mm = (coords - anchor) @ affine[:3, :3].T
                priority[np.sum(mm * mm, axis=1) < 64.] = -np.inf
        yield from candidates[:per_group]


def dump_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    joblib.dump(obj, tmp)
    tmp.replace(path)
