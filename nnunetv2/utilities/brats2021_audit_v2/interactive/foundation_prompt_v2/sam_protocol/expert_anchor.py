"""Check additions with a nearby independently supported background point."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..common import check_file, file_identity, read_json, read_labels, same_grid, write_json
from . import candidates, dense, expert_prompts, residual
from .cohorts import rows


def background_point(image, outside, xyz, spacing):
    """Choose a visible, close background example with similar MRI intensity."""
    x, y, z = xyz
    center = np.array([x, y])
    width = np.ceil(14. / spacing[:2]).astype(int)
    low = np.maximum(center - width, 0)
    high = np.minimum(center + width + 1, image.shape[:2])
    window = (slice(low[0], high[0]), slice(low[1], high[1]), z)
    choices = np.argwhere(outside[window] & (image[window] != 0)) + low
    distance = np.linalg.norm((choices - center) * spacing[:2], axis=1)
    use = (distance >= 3.) & (distance <= 14.)
    if not use.any():
        return None
    choices, distance = choices[use], distance[use]
    values = image[choices[:, 0], choices[:, 1], z]
    low_value, high_value = np.percentile(values, (10,90))
    difference = np.abs(values - image[x, y, z]) / max(1.e-6, float(high_value - low_value))
    index = int(np.argmin(distance / 7. + np.minimum(difference, 4.)))
    return [int(choices[index,0]), int(choices[index,1]), int(z)]


@torch.inference_mode()
def expert_anchor_responses(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backend = None
    for row in rows(args.cohort, args.split)[args.shard_index::args.num_shards]:
        case = row['case_id']
        mri, labels, ref, _, _, responses, sam, native = residual._evidence(row,args.dense,args.responses)
        if sam['contract']['prompt_profile'] != expert_prompts.EXPANDED or 'native_margin' not in native['executor']:
            raise ValueError('Background anchors require the paired native-margin candidates')
        recipe = {**native['executor'], 'background_anchor_code': file_identity(__file__)['sha256']}
        folder = Path(args.output) / case
        receipt_path = folder / 'responses.json'
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if (previous['row'] != row or previous['sam'] != sam['arrays'] or previous['executor'] != recipe
                    or previous['preceding_responses'] != native['arrays']):
                raise ValueError(f'{case}: existing contrastive responses use other inputs')
            check_file(previous['arrays'])
            continue
        tolerance = native['executor']['native_margin']['tolerance']
        with np.load(sam['arrays']['path'], allow_pickle=False) as values:
            fields = values['components'][0]
        before = labels > 0
        unanimously_background = ~before
        for identity in sam['expert_inputs']:
            check_file(identity)
            mask = nib.load(identity['path'])
            same_grid(mask, ref)
            unanimously_background &= read_labels(mask) == 0
        channel = native['modalities'][0]
        image = mri[channel]
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        anchors = []
        for action in sam['actions']:
            if action['head'] != 0 or not action['positive']:
                continue
            coordinates = np.argwhere(fields == action['id'])
            results = {}
            for v, signal in enumerate(action['variants']):
                if not signal['accept']:
                    continue
                prompt = signal['prompt']
                point = background_point(image, unanimously_background, prompt['point'], spacing)
                if point is None:
                    continue
                key = (tuple(prompt['point']), tuple(tuple(p) for p in prompt['box']), tuple(point))
                if key not in results:
                    if backend is None:
                        if torch.cuda.mem_get_info(device)[0] < args.min_free_mib * 2**20:
                            raise RuntimeError('Insufficient free CUDA memory for the 192³ point control')
                        backend = candidates.NativeExecutor(str(device))
                    if not anchors:
                        backend.engine.set_image(np.ascontiguousarray(image[None], dtype=np.float32))
                    results[key] = expert_prompts.native_margins(backend, before, coordinates, prompt,
                                                                 negative_point=point)
                kept, margin = results[key]
                responses[v,0,1][tuple(coordinates[kept].T)] = margin > -tolerance
                anchors.append({'action': action['id'], 'variant': dense.VARIANTS[v], 'negative_point': point})
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / 'responses.npz'
        packed = np.packbits(responses.reshape((*responses.shape[:3], -1)), axis=-1)
        dense._atomic_npz(arrays, packed=packed, shape=np.asarray(labels.shape))
        write_json(receipt_path, {**native, 'executor':recipe, 'arrays':file_identity(arrays),
                                  'preceding_responses': native['arrays'], 'background_anchors':anchors,
                                  'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30})
        print(f'{case}: {len(anchors)} WT positive cues checked with an independent background point', flush=True)
    return args.output
