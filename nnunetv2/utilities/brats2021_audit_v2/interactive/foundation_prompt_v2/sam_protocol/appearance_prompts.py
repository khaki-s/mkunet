"""Use an MRI proposal as label-free evidence for shared SAM3 point and box prompts."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..common import check_file, file_identity, read_json, write_json
from . import background, dense, expert_geometry, expert_prompts, image_memory
from .cohorts import calibration_rows

PRE_CALIBRATION_CODE = '8abfbfeadd6db42f8a094c49dbd30918053cd55b152ebe471b73b9015d904826'


def score_bands(initial, probability):
    # Encode signed confidence in the three bands used by the existing
    # component selector. Uncertain voxels support their original label.
    foreground = np.select((probability < .2, probability < .4, probability < .5),
                           (0, 1, 2), default=3)
    background = np.select((probability > .8, probability > .6, probability > .5),
                           (3, 2, 1), default=0)
    return np.where(initial, foreground, background).astype(np.uint8)


@torch.inference_mode()
def appearance_sam(args):
    spec = read_json(args.cohort)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source = file_identity(args.proposal_model)
    contract = {**background.contract(read_json(spec['train_manifest']['path'])),
                'prompt_profile': expert_prompts.APPEARANCE, 'proposal_model': source,
                'profile_code': file_identity(__file__)['sha256'],
                'native_geometry_code': file_identity(expert_geometry.__file__)['sha256'],
                'envelope_code': file_identity(expert_prompts.__file__)['sha256']}
    model = None
    chosen = calibration_rows(args.cohort, args.split, getattr(args, 'case_list', None))
    selected = chosen[args.shard_index::args.num_shards]
    for row in selected:
        key = row['case_id']
        folder = Path(args.output) / key
        path = folder / 'sam.json'
        proposal = read_json(Path(args.proposals) / key / 'proposal.json')
        if (proposal['gt_used'] is not False or proposal['binding']['row'] != row
                or proposal['binding']['model'] != source):
            raise ValueError(f'{key}: MRI proposal came from another input or checkpoint')
        if path.is_file():
            _, _, previous = dense.read_masks(args.output, row)
            equivalent = {**contract, 'profile_code': PRE_CALIBRATION_CODE,
                          'envelope_code': expert_prompts.PRE_CALIBRATION_CODE}
            if previous['contract'] not in (contract, equivalent) or previous['proposal'] != proposal['arrays']:
                raise ValueError(f'{key}: saved paired masks use other MRI evidence')
            continue
        check_file(proposal['arrays'])
        with np.load(proposal['arrays']['path'], allow_pickle=False) as data:
            probability = data['probabilities']
        mri, labels, reference = dense.load_native(row)
        before = np.stack([labels > h for h in range(3)])
        if probability.shape != before.shape or not np.isfinite(probability).all():
            raise ValueError('MRI proposal is not on the original native voxel grid')
        votes = score_bands(before, probability)
        if model is None:
            model, processor = image_memory.load_image_only(device, stream_blocks=args.stream_blocks)
            mask_size = model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        masks = np.stack((before.copy(), before.copy()))
        scores = np.zeros((2, 3, labels.shape[-1]), np.float32)
        indices = np.zeros(before.shape, np.uint8)
        digest = hashlib.sha256()
        actions = []
        sources = [next(i for i in prefs if row['availability'][i]) for prefs in dense.PREFERRED_MODALITIES]
        scaled = background._normalize(mri, set(sources))
        for h, channel in enumerate(sources):
            for comp, action, geometry in expert_prompts._actions(model, processor, scaled[channel], before[h],
                                                                 votes[h], h, spacing, mask_size, True):
                ident = len(actions) + 1
                indices[h, comp] = ident
                for v, outcome in enumerate(action['variants']):
                    if outcome['accept']:
                        masks[v, h, comp] = action['positive']
                    for z, *_ in geometry:
                        scores[v, h, z] = max(scores[v, h, z], outcome['quality'])
                for z, points, signs, box in geometry:
                    for array in (np.asarray([h, z, ident], np.int32), points, signs, box):
                        digest.update(array.tobytes())
                actions.append({'head': h, 'id': ident, **action})
        fields = expert_prompts._extend_disagreement(before, votes, indices, actions, spacing,
                                                     expert_prompts.EXPERT_RADIUS_MM)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / 'sam.npz'
        dense._atomic_npz(arrays, masks=masks, scores=scores, components=fields,
                          core_components=indices, majority=votes >= 2)
        write_json(path, {'row': row, 'contract': contract, 'variants': list(dense.VARIANTS),
                          'arrays': file_identity(arrays), 'gt_used': False, 'proposal': proposal['arrays'],
                          'native_shape': list(labels.shape), 'encoder_resolution': processor.resolution,
                          'shared_point_box_digest': digest.hexdigest(), 'actions': actions})
        print(f'{key}: {len(actions)} shared SAM3 prompts from native MRI proposal', flush=True)
    return args.output
