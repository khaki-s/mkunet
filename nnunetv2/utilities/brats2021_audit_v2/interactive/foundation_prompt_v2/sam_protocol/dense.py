"""A paired soft-mask ablation with the same SAM weights, image, points and box."""
from contextlib import nullcontext
from pathlib import Path
from time import monotonic

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from ..common import MODALITIES, check_file, file_identity, read_json, read_labels, same_grid, write_json
from ..sam_dense import HEADS, SamDenseRefiner, prompt_geometry
from ..sam_fusion_pipeline import contract as sam_contract
from .cohorts import rows

VARIANTS = ('sdf', 'weak_boundary')
PREFERRED_MODALITIES = ((3, 2, 1, 0), (1, 0, 2, 3), (1, 0, 2, 3))


def load_native(row):
    """Load only available MRI channels. Reuse the original prediction grid."""
    check_file(row['prediction'])
    reference = nib.load(row['prediction']['path'])
    baseline = read_labels(reference)
    mri = np.zeros((4, *baseline.shape), dtype=np.float32)
    for index, name in enumerate(MODALITIES):
        if row['availability'][index]:
            identity = row['modalities'][name]
            check_file(identity)
            image = nib.load(identity['path'])
            same_grid(image, reference)
            mri[index] = image.get_fdata(dtype=np.float32)
    if not np.isfinite(mri).all() or not any(row['availability']):
        raise ValueError('Require finite images and at least one MRI modality')
    return mri, baseline, reference


def soft_prompts(mask, mask_size, spacing_mm):
    """SDF baseline in voxel units; attenuate the same encoding near the edge."""
    d = distance_transform_edt(mask) - distance_transform_edt(~mask)
    physical = distance_transform_edt(mask, sampling=spacing_mm) - distance_transform_edt(~mask, sampling=spacing_mm)
    sdf = np.clip(d / 2., -6., 6.)
    # At 1 mm the prior contributes about one third of its original strength.
    # The outer 3x scale quickly restores the original signed evidence.
    strength = 1.0 - .75 * np.exp(-np.square(physical / 3.0))
    maps = np.stack((sdf, sdf * strength)).astype(np.float32)
    low = F.interpolate(torch.from_numpy(maps[:, None]), size=mask_size, mode='bilinear', align_corners=False)
    return low.numpy()


class PairedSamRefiner(SamDenseRefiner):
    @torch.inference_mode()
    def generate_pair(self, mri, baseline, availability, spacing):
        if mri.shape != (4, *baseline.shape) or baseline.ndim != 3:
            raise ValueError('Expected four native volume slots and one label map')
        sources = [next(i for i in prefs if availability[i]) for prefs in PREFERRED_MODALITIES]
        masks = np.stack([baseline > h for h in range(3)])
        candidates = np.stack((masks.copy(), masks.copy()))
        scores = np.zeros((2, 3, baseline.shape[2]), dtype=np.float32)
        calls, encodes = 0, 0
        for channel in sorted(set(sources)):
            selected = [h for h in range(3) if sources[h] == channel]
            slices = np.flatnonzero(masks[selected].any(axis=(0, 1, 2)))
            # Match the original normalization, including its full-volume statistics.
            values = mri[channel][mri[channel] != 0]
            low, high = np.percentile(values, (1, 99)) if len(values) else (0., 1.)
            for z in slices:
                gray = (np.clip((mri[channel, :, :, z] - low) / max(float(high-low), 1.e-6), 0, 1) * 255).astype(np.uint8)
                rgb = Image.fromarray(np.repeat(gray[..., None], 3, axis=-1))
                amp = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
                with amp:
                    state = self.processor.set_image(rgb)
                    encodes += 1
                    for h in selected:
                        base_slice = masks[h, :, :, z]
                        if not base_slice.any():
                            continue
                        points, box = prompt_geometry(base_slice)
                        labels = np.ones(len(points), dtype=np.int32)
                        priors = soft_prompts(base_slice, self.mask_size, spacing[:2])
                        for variant, prior in enumerate(priors):
                            proposals, iou, _ = self.model.predict_inst(
                                state, point_coords=points, point_labels=labels, box=box, mask_input=prior,
                                multimask_output=True, return_logits=False, normalize_coords=True)
                            proposals = np.asarray(proposals)
                            iou = np.asarray(iou).reshape(-1)
                            if (proposals.shape[1:] != base_slice.shape or len(proposals) != len(iou)
                                    or not np.isfinite(iou).any()):
                                raise ValueError('Unexpected SAM mask candidates')
                            best = int(np.argmax(np.where(np.isfinite(iou), iou, -np.inf)))
                            candidates[variant, h, :, :, z] = proposals[best] > 0
                            scores[variant, h, z] = iou[best]
                            calls += 1
                del state
        return candidates, scores, {'calls': calls, 'encodes': encodes, 'encoder_resolution': 1008,
                                    'modalities': sources, 'native_shape': list(baseline.shape)}


def read_masks(root, row):
    folder = Path(root) / row['case_id']
    receipt = read_json(folder / 'sam.json')
    if receipt['row'] != row or receipt['variants'] != list(VARIANTS) or receipt.get('gt_used'):
        raise ValueError(f'Cache does not match input for {row["case_id"]}')
    check_file(receipt['arrays'])
    with np.load(receipt['arrays']['path'], allow_pickle=False) as data:
        return data['masks'], data['scores'], receipt


def _atomic_npz(path, **arrays):
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def build_cache(selected, args, root, contract, *, factory=PairedSamRefiner):
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model = None
    started = monotonic()
    written = 0
    for row in selected:
        key = row['case_id']
        folder = root / key
        if (folder / 'sam.json').exists():
            _, _, receipt = read_masks(root, row)
            if receipt['contract'] != contract:
                raise ValueError(f'{key}: cached SAM configuration differs')
            continue
        mri, baseline, ref = load_native(row)
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        if model is None:
            model = factory(str(device))
        masks, scores, details = model.generate_pair(mri, baseline, row['availability'], spacing)
        folder.mkdir(exist_ok=True)
        path = folder / 'sam.npz'
        _atomic_npz(path, masks=masks, scores=scores)
        write_json(folder / 'sam.json', {'row': row, 'variants': list(VARIANTS), 'arrays': file_identity(path),
                                         'contract': contract, 'gt_used': False, **details})
        written += 1
        print(f'SAM3 {key} mask={row["mask_code"]} calls={details["calls"]} elapsed={monotonic()-started:.1f}s', flush=True)
    return {'completed': len(selected), 'new_cases': written, 'seconds': monotonic() - started,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else 0}


def cache(args):
    train = rows(args.cohort, 'train')
    validation = rows(args.cohort, 'validation')
    selected = {'train': train, 'validation': validation, 'both': train + validation}[args.split]
    selected = selected[args.shard_index::args.num_shards]
    binding = read_json(args.cohort)
    source = read_json(binding['train_manifest']['path'])
    method = {'sam': sam_contract(source), 'paired_prompt_code': file_identity(__file__)['sha256'],
              'variants': list(VARIANTS)}
    path = Path(args.output) / f'{args.split}_shard{args.shard_index}.json'
    stats = build_cache(selected, args, args.output, method)
    if stats['new_cases'] or not path.exists():
        write_json(path, stats)
    return path
