"""Collect continuous SAM3 and 192³ nnInteractive evidence for supervised use."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..common import check_file, file_identity, read_json, read_labels, same_grid, write_json
from . import candidates, dense, expert_prompts, residual
from .cohorts import rows

SCHEMA = 'paired_expert_continuous_v1'
MODALITY_SCHEMA = 'paired_expert_modality_continuous_v2'
EXTRA_COLUMNS = 8


def read_features(root, row, sam, native):
    receipt = read_json(Path(root) / row['case_id'] / 'features.json')
    schema = MODALITY_SCHEMA if sam.get('modality_sources') else SCHEMA
    expected = {'row': row, 'sam': sam['arrays'], 'responses': native['arrays'], 'schema': schema}
    if receipt['binding'] != expected or receipt['gt_used'] is not False:
        raise ValueError(f'{row["case_id"]}: continuous features originated from different candidates')
    check_file(receipt['arrays'])
    with np.load(receipt['arrays']['path'], allow_pickle=False) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    data['schema'] = schema
    return data, receipt['arrays']


def _head_features(backend, before, coordinates, fields, sam, experts, modalities, response, spacing, h, v):
    n = len(coordinates)
    columns = EXTRA_COLUMNS + bool(modalities)
    if not n:
        return np.empty((0, columns), np.float32), 0
    base = before > h
    original = base.ravel()[coordinates]
    signed_ids = fields[h].ravel()[coordinates]
    xyz = np.stack(np.unravel_index(coordinates, before.shape), axis=1)
    votes = np.sum([(labels.ravel()[coordinates] > h) for labels in experts], axis=0)
    extra = np.zeros((n, columns), np.float32)
    extra[:, 2] = votes / 3
    if modalities:
        differences = [(labels.ravel()[coordinates] > h) != original for labels in modalities]
        extra[:, EXTRA_COLUMNS] = np.mean(differences, axis=0)
    mismatches = 0
    for action in sam['actions']:
        if action['head'] != h or not action['variants'][v]['accept']:
            continue
        action_limit = signed_ids == action['id']
        chosen = np.flatnonzero(action_limit)
        if not len(chosen):
            continue
        cue = action['variants'][v]
        prompt = cue['prompt']
        inside, margins = expert_prompts.native_margins(backend, base, xyz[chosen], prompt)
        slots = chosen[inside]
        extra[slots, 0] = margins
        extra[slots, 1] = 1
        extra[chosen, 3] = cue['agreement']
        extra[chosen, 4] = cue['quality']
        extra[chosen, 5] = np.ptp(cue['plane_agreements'])
        extra[chosen, 6] = np.linalg.norm((xyz[chosen] - prompt['point']) * spacing, axis=1)
        other = action['variants'][1-v]
        extra[chosen, 7] = other['agreement']
        tolerance = response['tolerance']
        threshold = -tolerance if prompt['positive'] else tolerance
        masks = response['masks'][1 if prompt['positive'] else 2].ravel()
        mismatches += int(np.count_nonzero((margins > threshold) != masks[coordinates[slots]]))
        mismatches += int(np.count_nonzero(original[chosen[~inside]] != masks[coordinates[chosen[~inside]]]))
    return extra, mismatches


@torch.inference_mode()
def collect_expert_features(args):
    chosen = rows(args.cohort, args.split)[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    backend = None
    for row in chosen:
        key = row['case_id']
        if 'gt' in row:
            raise ValueError('Feature acquisition uses label-free rows')
        mri, before, ref, sam_masks, confidence, responses, sam, native = residual._evidence(
            row, args.dense, args.responses)
        if sam['contract']['prompt_profile'] != expert_prompts.EXPANDED or 'native_margin' not in native['executor']:
            raise ValueError('Continuous evidence requires expanded paired prompts and native margins')
        directory = Path(args.output) / key
        path = directory / 'features.json'
        if path.exists():
            read_features(args.output, row, sam, native)
            continue
        tolerance = native['executor']['native_margin']['tolerance']
        with np.load(sam['arrays']['path'], allow_pickle=False) as arrays:
            fields = arrays['components']
        votes = []
        for identities in (sam['expert_inputs'], sam.get('modality_sources', ())):
            loaded = []
            for identity in identities:
                check_file(identity)
                mask = nib.load(identity['path'])
                same_grid(mask, ref)
                loaded.append(read_labels(mask))
            votes.append(loaded)
        experts, modalities = votes
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        arrays, mismatch, total = {}, 0, 0
        channels = native['modalities']
        for h, channel in enumerate(channels):
            if backend is None:
                if torch.cuda.mem_get_info(device)[0] < args.min_free_mib * 2**20:
                    raise RuntimeError('Insufficient free CUDA memory for native 192³ confidence extraction')
                backend = candidates.NativeExecutor(str(device))
            backend.engine.set_image(np.ascontiguousarray(mri[channel][None], dtype=np.float32))
            for v in range(len(dense.VARIANTS)):
                selected, x = residual.observations(mri, before, sam_masks[v, h], responses[v, h],
                                                   confidence[v, h], ref.affine, h)
                native_view = {'tolerance': tolerance, 'masks': responses[v, h]}
                extra, disagreements = _head_features(backend, before, selected, fields, sam, experts, modalities,
                                                       native_view, spacing, h, v)
                arrays[f'indices_{v}_{h}'] = selected
                arrays[f'x_{v}_{h}'] = np.column_stack((x, extra)) if len(selected) else x
                total += int(extra[:, 1].sum())
                mismatch += disagreements
        if mismatch > max(3, int(.002 * total)):
            raise ValueError(f'{key}: continuous and original native outputs differ at {mismatch} voxels')
        directory.mkdir(parents=True, exist_ok=True)
        values = directory / 'features.npz'
        dense._atomic_npz(values, **arrays)
        schema = MODALITY_SCHEMA if modalities else SCHEMA
        binding = {'row': row, 'sam': sam['arrays'], 'responses': native['arrays'], 'schema': schema}
        write_json(path, {'binding': binding, 'arrays': file_identity(values), 'gt_used': False,
                          'native_margin_parity_mismatch': mismatch, 'native_margin_comparisons': total,
                          'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30,
                          'feature_code': file_identity(__file__)['sha256']})
        print(f'{key}: native continuous response, {mismatch}/{total} threshold differences', flush=True)
    return args.output
