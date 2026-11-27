"""Calibrate agreement between the two fixed-geometry SAM3 prompt variants."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from ..common import check_file, file_identity, manifest, read_json, read_labels, save_mask, same_grid, write_json
from . import gpu_fusion, residual, scoring
from .cohorts import rows

MODES = ('mean', 'agreement', 'downweight_single')
EXPERIMENT = 'sam_paired_consensus_wt_v1'


def _paired(row, args, models):
    mri, labels, ref, masks, iou, response, sam, native = residual._evidence(row, args.dense, args.responses)
    points, estimates = [], []
    for i, model in enumerate(models):
        indices, features = residual.observations(mri, labels, masks[i, 0], response[i, 0], iou[i, 0], ref.affine, 0)
        keep = labels.ravel()[indices] <= 1
        points.append(indices[keep])
        estimates.append(residual.posterior(model, features[keep]))
    union = np.union1d(*points)
    available = np.zeros(len(union), dtype=np.uint8)
    sums = np.zeros(len(union), dtype=np.float32)
    for coords, p in zip(points, estimates):
        positions = np.searchsorted(union, coords)
        available[positions] += 1
        sums[positions] += p
    p = sums / np.maximum(available, 1)
    paired = available == 2
    # Shrink single-arm agreement toward the baseline decision, keeping the
    # posterior unchanged where both routes support the proposed correction.
    retained_prior = labels.ravel()[union] > 0
    consensus = {'mean': (union, p), 'agreement': (union[paired], p[paired]),
                 'downweight_single': (union, np.where(paired, p, .65 * p + .35 * retained_prior))}
    return labels, ref, consensus, sam, native


def consensus_predict(args):
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Train calibration readouts on CUDA')
    torch.cuda.set_device(device)
    all_train = rows(args.cohort, 'train', truth=True)
    fitting = [row['case_id'] for i, row in enumerate(all_train) if i % 5 != 4]
    calibration = all_train[4::5]
    samples = residual._sample_ids(args.samples, all_train)
    provisional = [gpu_fusion._train(samples, fitting, i, 0, device) for i in range(2)]
    train_scores = {name: [] for name in MODES}
    for row in calibration:
        clean = {k: v for k, v in row.items() if k != 'gt'}
        baseline, ref, variants, _, _ = _paired(clean, args, provisional)
        truth = residual._truth(row, ref) > 0
        for name, (coords, p) in variants.items():
            train_scores[name].append(residual._gains(p, coords, baseline > 0, truth))
        print('Consensus calibration', row['case_id'], flush=True)
    options = (*residual.THRESHOLDS, None)
    settings = {}
    for name, values in train_scores.items():
        gains = np.stack(values)
        objective = gains.mean(0) - .25 * np.maximum(0, -gains).mean(0)
        i, j = np.unravel_index(np.argmax(objective), objective.shape) if objective.max() > 0 else (-1, -1)
        settings[name] = {'add': options[i], 'remove': options[j], 'calibration_gain_pp': float(gains[:, i, j].mean())}
    chosen = max(MODES, key=lambda name: settings[name]['calibration_gain_pp'])
    models, _, identities = residual.load_models(args.model, args.cohort)
    models = [row['model'] for row in models[0]['variants']]
    root = Path(args.output)
    contract = {'schema': EXPERIMENT, 'cohort': file_identity(args.cohort), 'models': identities,
                'code': file_identity(__file__), 'calibration': settings, 'chosen_on_train': chosen,
                'calibration_ids': [row['case_id'] for row in calibration]}
    write_json(root / 'calibration.json', contract)
    for row in rows(args.cohort, 'validation'):
        baseline, ref, variants, sam, native = _paired(row, args, models)
        folder = root / row['case_id']
        outputs = {}
        for name, (coords, p) in variants.items():
            config = settings[name]
            original = baseline > 0
            plus, minus = residual._changed(p, coords, original, config['add'], config['remove'])
            updated = np.array(original, order='C')
            updated.ravel()[coords[plus]] = True
            updated.ravel()[coords[minus]] = False
            labels = np.where(baseline > 1, baseline, updated).astype(np.uint8)
            outputs[name] = save_mask(folder / f'{name}.nii.gz', labels, ref)
        write_json(folder / 'prediction.json', {'row': row, 'gt_used': False, 'sam': sam['arrays'],
                   'response': native['arrays'], 'outputs': outputs, 'experiment': file_identity(root / 'calibration.json')})
        print('Consensus prediction', row['case_id'], flush=True)
    return root


def consensus_score(args):
    root = Path(args.predictions)
    binding = file_identity(root / 'calibration.json')
    calibration = read_json(binding['path'])
    if (calibration['schema'] != EXPERIMENT or calibration['cohort'] != file_identity(args.cohort)
            or calibration['code'] != file_identity(__file__)):
        raise ValueError('Mismatched frozen consensus study')
    labels = {row['case_id']: row for row in manifest(args.truth, allowed=('validation',))['cases']}
    cohort = rows(args.cohort, 'validation')
    records = {name: [] for name in MODES}
    for row in cohort:
        key = row['case_id']
        actual = read_json(root / key / 'prediction.json')
        if actual['row'] != row or actual['gt_used'] or actual['experiment'] != binding:
            raise ValueError('Consensus prediction does not match calibration or patient')
        before, truth, ref, volume = scoring._image_pair(row, labels[key])
        for name, entry in actual['outputs'].items():
            check_file(entry)
            image = nib.load(entry['path'])
            same_grid(ref, image)
            records[name].extend({'case_id': key, **r} for r in scoring._measure(before, read_labels(image), truth, volume))
    write_json(args.output, {'schema': EXPERIMENT, 'n_cases': len(cohort), 'calibration': calibration,
               'by_mode': {name: scoring.summary(value) for name, value in records.items()}, 'per_case': records})
    return args.output
