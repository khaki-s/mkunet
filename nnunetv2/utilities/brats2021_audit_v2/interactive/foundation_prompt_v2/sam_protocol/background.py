"""Paired SAM3 prompts with multimodal image evidence and local negative points."""
from contextlib import nullcontext
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torch

from ..common import file_identity, read_json, write_json
from ..sam_dense import prompt_geometry
from . import dense
from .cohorts import rows

PROFILE = 'multimodal_background_v5'
PREFERENCES = ((3, 2, 1, 0), (1, 0, 2, 3), (1, 0, 2, 3))


def negative_points(seed, channels, brain, spacing, limit=3):
    """Find nearby visible points that are consistently unlike the seed tissue."""
    outside = distance_transform_edt(~seed, sampling=spacing)
    inside = distance_transform_edt(seed, sampling=spacing)
    foreground = seed & (inside > 2.) & brain
    if np.count_nonzero(foreground) < 15:
        foreground = seed & brain
    ring = (outside >= 3.) & (outside <= 16.) & brain
    if np.count_nonzero(foreground) < 8 or np.count_nonzero(ring) < 8:
        return np.empty((0, 2), dtype=np.float32)
    evidence = []
    for image in channels:
        positive = image[foreground & (image > 0)]
        background = image[ring & (image > 0)]
        if len(positive) < 8 or len(background) < 8:
            continue
        lo, mid, hi = np.quantile(positive, (.20, .50, .80))
        normal_mid = np.median(background)
        scale = max(float(hi - lo), .07)
        contrast = abs(float(mid - normal_mid)) / scale
        if contrast < .30:
            continue
        sign = np.sign(mid - normal_mid)
        different = sign * (mid - image) / scale
        evidence.append(np.clip(different, -3, 3))
    if not evidence:
        return np.empty((0, 2), dtype=np.float32)
    scores = np.stack(evidence)
    confidence = gaussian_filter(scores.mean(0), sigma=1.)
    reliable = ring & (confidence >= .85) & (scores.min(axis=0) > .1)
    priority = np.where(reliable, confidence - .025 * np.abs(outside - 6.), -np.inf)
    yr, xr = np.ogrid[:seed.shape[0], :seed.shape[1]]
    selected = []
    for _ in range(limit):
        pos = np.unravel_index(np.argmax(priority), seed.shape)
        if not np.isfinite(priority[pos]):
            break
        y, x = pos
        selected.append((x, y))
        suppress = np.square((yr - y) * spacing[0]) + np.square((xr - x) * spacing[1]) < 12.**2
        priority[suppress] = -np.inf
    return np.asarray(selected, dtype=np.float32).reshape(-1, 2)


def _normalize(mri, available):
    scaled = {}
    for i in available:
        volume = mri[i]
        samples = volume[::2, ::2, ::2]
        samples = samples[samples != 0]
        if not len(samples):
            raise ValueError('An available MRI modality contains no visible tissue')
        low, high = np.percentile(samples, (1, 99))
        value = np.clip((volume - low) / max(float(high - low), 1.e-6), 0., 1.)
        # Keep visible low-signal tissue distinct from image padding.
        scaled[i] = np.where(volume != 0, np.maximum(value, 1. / 255), 0.)
    return scaled


def _projected_seed(original, present, index, images, spacing, brain):
    """A conservative cue for blank axial slices close to an observed lesion."""
    if not len(present):
        return None
    near = int(present[np.argmin(np.abs(present - index))])
    distance = abs(near - index) * spacing[2]
    if distance > 6.:
        return None
    previous = original[..., near]
    if np.count_nonzero(previous) < 30:
        return None
    # A continuation has to be supported in at least half of the visible
    # overlapping territory in both of the available leading modalities.
    overlap = previous & brain
    if np.count_nonzero(overlap) < .7 * np.count_nonzero(previous):
        return None
    for volume in images[:2]:
        past = volume[..., near][previous]
        current = volume[..., index][overlap]
        q20, q50, q80 = np.quantile(past, (.20, .50, .80))
        if abs(float(np.median(current) - q50)) > max(float(q80 - q20), .08) * .65:
            return None
    return overlap


class BackgroundPairedRefiner(dense.SamDenseRefiner):
    @torch.inference_mode()
    def generate_pair(self, mri, baseline, availability, spacing):
        available = [i for i, on in enumerate(availability) if on]
        fields = _normalize(mri, available)
        brain = np.any(mri[available] != 0, axis=0)
        source = [tuple(i for i in preferences if i in fields) for preferences in PREFERENCES]
        references = np.stack([baseline > h for h in range(len(source))])
        wt_slices = np.flatnonzero(references[0].any(axis=(0, 1)))
        masks = np.stack((references.copy(), references.copy()))
        scores = np.zeros((2, len(source), baseline.shape[-1]), dtype=np.float32)
        counters = dict(calls=0, encodes=0, background_points=0, projected_wt_slices=0)
        digest = hashlib.sha256()

        for order in dict.fromkeys(source):
            heads = [head for head, item in enumerate(source) if item == order]
            views = [fields[i] for i in order]
            rgb_views = [views[i % len(views)] for i in range(3)]
            for z in range(baseline.shape[-1]):
                requests = []
                for h in heads:
                    seed = references[h, ..., z]
                    if not seed.any():
                        if h != 0:
                            continue
                        seed = _projected_seed(references[h], wt_slices, z, views, spacing, brain[..., z])
                        if seed is None:
                            continue
                        counters['projected_wt_slices'] += 1
                    positives, box = prompt_geometry(seed, padding=8 if h == 0 else 4)
                    background = negative_points(seed, [view[..., z] for view in views[:2]], brain[..., z], spacing[:2])
                    points = np.concatenate((positives, background), axis=0)
                    labels = np.concatenate((np.ones(len(positives), np.int32), np.zeros(len(background), np.int32)))
                    counters['background_points'] += len(background)
                    for payload in (np.array((h, z, len(points)), np.int32), points, labels, box):
                        digest.update(payload.tobytes())
                    requests.append((h, seed, points, labels, box))
                if not requests:
                    continue
                rgb = (np.stack([view[..., z] for view in rgb_views], axis=-1) * 255).astype(np.uint8)
                amp = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
                with amp:
                    state = self.processor.set_image(Image.fromarray(rgb))
                    counters['encodes'] += 1
                    for h, seed, points, labels, box in requests:
                        priors = dense.soft_prompts(seed, self.mask_size, spacing[:2])
                        for variant, prior in enumerate(priors):
                            proposals, quality, _ = self.model.predict_inst(
                                state, point_coords=points, point_labels=labels, box=box, mask_input=prior,
                                multimask_output=True, return_logits=False, normalize_coords=True)
                            proposals = np.asarray(proposals) > 0
                            quality = np.asarray(quality).reshape(-1)
                            if proposals.shape != (len(quality), *seed.shape) or not np.isfinite(quality).any():
                                raise ValueError('Unexpected paired SAM3 output')
                            xy = np.rint(points).astype(int)
                            click_ok = np.mean(proposals[:, xy[:, 1], xy[:, 0]] == labels[None, :], axis=1)
                            best = int(np.argmax(np.where(np.isfinite(quality), quality + .25 * click_ok, -np.inf)))
                            masks[variant, h, ..., z] = proposals[best] & brain[..., z]
                            scores[variant, h, z] = quality[best]
                            counters['calls'] += 1
                del state
        return masks, scores, {**counters, 'encoder_resolution': 1008,
                               'modalities': [list(order) for order in source],
                               'native_shape': list(baseline.shape), 'shared_point_box_digest': digest.hexdigest()}


def contract(train_manifest):
    return {'sam': dense.sam_contract(train_manifest),
            'paired_prompt_code': file_identity(dense.__file__)['sha256'],
            'background_prompt_code': file_identity(__file__)['sha256'],
            'prompt_profile': PROFILE, 'variants': list(dense.VARIANTS)}


def cache_background(args):
    train = rows(args.cohort, 'train')
    validation = rows(args.cohort, 'validation')
    selected = {'train': train, 'validation': validation, 'both': train + validation}[args.split]
    selected = selected[args.shard_index::args.num_shards]
    binding = read_json(args.cohort)
    config = contract(read_json(binding['train_manifest']['path']))
    source = Path(args.output)
    stats = dense.build_cache(selected, args, source, config, factory=BackgroundPairedRefiner)
    path = source / f'{args.split}_shard{args.shard_index}.json'
    if stats['new_cases'] or not path.exists():
        write_json(path, stats)
    return path
