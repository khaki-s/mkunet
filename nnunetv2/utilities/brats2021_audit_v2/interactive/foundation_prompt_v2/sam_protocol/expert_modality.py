"""Use leave-one-modality-out disagreement near existing paired SAM3 cues."""
from pathlib import Path

import nibabel as nib
import numpy as np

from ..common import check_file, file_identity, read_json, read_labels, same_grid, write_json
from . import dense, expert_prompts
from .cohorts import rows

MODALITY_CODES = (7, 11, 13, 14)


def expert_modality_expand(args):
    """Extend the candidate search from a completed, held-out modality archive."""
    archive = Path(args.archive)
    source = file_identity(archive / 'manifest.json')
    provenance = read_json(source['path'])
    fresh = provenance.get('schema') == 'frozen_modality_expert_v1'
    split = getattr(args, 'split', 'validation')
    if (split == 'train' or fresh) and (not fresh or provenance.get('gt_used') is not False):
        raise ValueError('This comparison requires verified label-free modality predictions')
    selected = rows(args.cohort, split)[args.shard_index::args.num_shards]
    for row in selected:
        key = row['case_id']
        masks, scores, sam = dense.read_masks(args.dense, row)
        if sam['contract']['prompt_profile'] != expert_prompts.EXPANDED or row['mask_code'] != 15:
            raise ValueError('This comparison requires four-modality receipts and expanded SAM3 evidence')
        contract = {**sam['contract'], 'envelope_code': file_identity(__file__)['sha256'],
                    'modality_archive': source, 'modality_codes': list(MODALITY_CODES)}
        folder = Path(args.output) / key
        receipt = folder / 'sam.json'
        if receipt.exists():
            old = read_json(receipt)
            if old['row'] != row or old['contract'] != contract or old['modality_expanded_from'] != sam['arrays']:
                raise ValueError(f'{key}: existing modality evidence belongs to another comparison')
            check_file(old['arrays'])
            for dependency in old['modality_sources']:
                check_file(dependency)
            continue
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        labels = read_labels(reference)
        base = np.stack([labels > h for h in range(3)])
        disagrees = np.zeros(base.shape, dtype=bool)
        identities = []
        for code in MODALITY_CODES:
            path = archive / 'predictions' / f'mask_{code:02d}' / f'{key}.nii.gz'
            identity = file_identity(path)
            if fresh:
                record = read_json(path.parent / f'{key}.json')
                expected = {'row': row, 'weights': provenance['weights'], 'mask_code': code,
                            'controlled_preprocessing': True, 'mirroring': True}
                if record['binding'] != expected or record['gt_used'] is not False or record['mask'] != identity:
                    raise ValueError(f'{key} mask_{code:02d}: modality input or checkpoint disagrees')
            image = nib.load(path)
            same_grid(image, reference)
            partial = read_labels(image)
            for h in range(3):
                disagrees[h] |= (partial > h) != base[h]
            identities.append(identity)
        signed_votes = np.where(base, 3 - disagrees, disagrees).astype(np.uint8)
        with np.load(sam['arrays']['path'], allow_pickle=False) as arrays:
            previous = arrays['components']
            core = arrays['core_components']
            majority = arrays['majority']
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        expanded = expert_prompts._extend_disagreement(base, signed_votes, previous, sam['actions'],
                                                        spacing, sam['contract']['radius_mm'])
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / 'sam.npz'
        dense._atomic_npz(output, masks=masks, scores=scores, components=expanded,
                          core_components=core, majority=majority)
        write_json(receipt, {**sam, 'contract': contract, 'arrays': file_identity(output),
                             'modality_expanded_from': sam['arrays'], 'modality_sources': identities})
        print(f'{key}: added {np.count_nonzero(expanded) - np.count_nonzero(previous)} modality disagreement voxels',
              flush=True)
    return args.output
