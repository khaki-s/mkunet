"""Train-only ablation of learned 3D error maps for SAM-derived corrections."""
from pathlib import Path

import joblib
import numpy as np
from scipy import ndimage
import torch

from ..common import file_identity, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import candidates, dense, residual, targeted_background
from .cohorts import rows

SCHEMA = 'sam_signed_error_critic_train_ablation_v1'
POINT_CONFIDENCE = .7
EDIT_CONFIDENCE = .55
MIN_POINT_RISK = .35


def fit_critic(args):
    assessment = {row['case_id'] for row in targeted_background._selected(args.cohort)}
    training = [row for row in rows(args.cohort, 'train', truth=True) if row['case_id'] not in assessment]
    samples = residual._sample_ids(args.samples, training)
    keys = [row['case_id'] for row in training]
    models = []
    for variant, name in enumerate(dense.VARIANTS):
        model = residual._fit(samples, keys, variant, 0, args.threads)
        models.append(model)
        print(f'Fitted {name} on {len(keys)} training patients', flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_atomic({'schema': SCHEMA, 'cohort': file_identity(args.cohort),
                 'trained_on': keys, 'assessment': sorted(assessment), 'models': models}, path)
    return path


def _error_probability(mri, baseline, raw, outputs, quality, affine, models, threads):
    base = baseline > 0
    weight = np.zeros(base.size, dtype=np.uint8)
    risk = np.zeros(base.size, dtype=np.float32)
    for model, proposal, result, score in zip(models, raw, outputs, quality):
        indices, features = residual.observations(mri, baseline, proposal[0], result[0], score[0], affine, 0)
        foreground = residual.posterior(model, features, threads)
        risk[indices] += np.where(base.ravel()[indices], 1. - foreground, foreground)
        weight[indices] += 1
    np.divide(risk, weight, out=risk, where=weight > 0)
    return risk.reshape(base.shape)


def _proposal(base, risk, image, spacing, positive):
    # Choose a point where the 3D error map contains stable local evidence.
    # Normalize the smoothing by how much of the local band the critic saw.
    side = (~base if positive else base) & (image != 0) & (risk > 0)
    if not side.any():
        return None
    weights = ndimage.gaussian_filter(side.astype(np.float32), 1.2 / spacing)
    confidence = ndimage.gaussian_filter(np.where(side, risk, 0.), 1.2 / spacing)
    confidence /= np.maximum(weights, 1.e-6)
    cutoff = max(MIN_POINT_RISK, float(np.quantile(confidence[side], .95)))
    eligible = side & (risk >= MIN_POINT_RISK) & (confidence >= cutoff) & (weights >= .12)
    roi_mask = base | eligible if positive else base & ~eligible
    return candidates.locate_component(roi_mask, base, image, spacing, positive)


def assess_critic(args):
    record = joblib.load(args.model)
    selected = targeted_background._selected(args.cohort)
    keys = {row['case_id'] for row in selected}
    if (record['schema'] != SCHEMA or record['cohort'] != file_identity(args.cohort)
            or keys != set(record['assessment']) or keys.intersection(record['trained_on'])):
        raise ValueError('Critic assessment must remain disjoint from critic fitting')
    selected = selected[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError('Critic assessment partition is empty')
    engine = None
    device = torch.device(args.device)
    if args.execute:
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            if torch.cuda.mem_get_info(device)[0] < candidates.MIN_FREE_MIB * 2**20:
                raise RuntimeError('Insufficient free CUDA memory for native 192³ critic assessment')
        engine = candidates.NativeExecutor(str(device))
    results = []
    for row in selected:
        clean = {k: v for k, v in row.items() if k != 'gt'}
        mri, initial, reference, sam, quality, outputs, _, _ = residual._evidence(clean, args.dense, args.responses)
        base = initial > 0
        risk = _error_probability(mri, initial, sam, outputs, quality, reference.affine,
                                  record['models'], args.threads)
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        image = mri[3]
        prompts = [_proposal(base, risk, image, spacing, positive) for positive in (True, False)]
        truth = residual._truth(row, reference) > 0
        item = {'case_id': row['case_id'], 'prompts': prompts, 'point_is_error': [
            None if prompt is None else bool(truth[tuple(prompt['point'])] != base[tuple(prompt['point'])])
            for prompt in prompts]}

        def dice(mask):
            count = int(mask.sum() + truth.sum())
            return 200. * np.count_nonzero(mask & truth) / count if count else 100.

        item['baseline_WT_pct'] = dice(base)
        item['critic_mask_gain_pp'] = dice(np.where(risk >= POINT_CONFIDENCE, ~base, base)) - dice(base)
        if engine is not None:
            engine.engine.set_image(np.ascontiguousarray(image[None]))
            add, remove = [engine.finite(base, prompt) for prompt in prompts]
            combined = (base | add) & remove
            accepted = np.where(risk >= EDIT_CONFIDENCE, combined, base)
            item['actual_192_gain_pp'] = dice(combined) - dice(base)
            item['actual_192_gated_gain_pp'] = dice(accepted) - dice(base)
        results.append(item)
        print(item['case_id'], item['point_is_error'], flush=True)
    metrics = [k for k in results[0] if k.endswith('_pp')]
    details = {key: float(np.mean([record[key] for record in results])) for key in metrics}
    write_json(args.output, {'schema': SCHEMA, 'split': 'training patients excluded from critic fitting',
                             'n_cases': len(results), 'model': file_identity(args.model), 'mean': details,
                             'proposal_only': not args.execute, 'execution_device': device.type if args.execute else None,
                             'per_case': results})
    return args.output
