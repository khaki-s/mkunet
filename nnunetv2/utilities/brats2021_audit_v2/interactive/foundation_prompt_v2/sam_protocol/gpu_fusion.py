"""GPU-trained native voxel readout, calibrated on held-out training patients."""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from ..common import file_identity, write_json
from ..sam_components.sam_prompt_components import dump_atomic
from . import dense, residual
from .cohorts import rows


class PortableReadout:
    """CPU NumPy deployment of the small trained Torch network."""

    def __init__(self, center, scale, layers):
        self.center = center
        self.scale = scale
        self.layers = layers

    def predict_proba(self, x):
        predictions = []
        with threadpool_limits(limits=3):
            for piece in np.array_split(x, max(1, int(np.ceil(len(x)/8192)))):
                value = np.clip((piece - self.center) / self.scale, -8, 8)
                for weight, bias in self.layers[:-1]:
                    value = np.maximum(0, value @ weight.T + bias)
                weight, bias = self.layers[-1]
                probability = 1. / (1. + np.exp(-np.clip((value @ weight.T + bias)[:,0], -30, 30)))
                predictions.append(probability)
        p = np.concatenate(predictions)
        return np.column_stack((1.-p, p))


def _patient_dice_weights(samples, keys, head):
    """Reduce the influence of patient volume on a case-mean Dice objective."""
    volumes = {}
    for key in keys:
        coverage = samples[key]['coverage'][f'sdf/{dense.HEADS[head]}']
        errors = coverage['fn'] + coverage['fp']
        if errors:
            volumes[key] = errors / (1. - coverage['baseline_pct'] / 100.)
    if not volumes:
        return {}
    typical = np.median(list(volumes.values()))
    return {key: float(np.clip(np.sqrt(typical / volume), .5, 2.)) for key, volume in volumes.items()}


def _train(samples, keys, variant, head, device, *, case_weights=None):
    case_weights = case_weights or {}
    features, labels, sample_weights = [], [], []
    for key in keys:
        with np.load(samples[key]['arrays']['path'], allow_pickle=False) as f:
            x, y, care = (f[f'{name}_{variant}_{head}'] for name in ('x','y','weight'))
        if len(y):
            features.append(x)
            labels.append(y)
            sample_weights.append(np.full(len(y), float(care) * case_weights.get(key, 1.) / np.sqrt(len(y)),
                                          np.float32))
    if not labels:
        return .5
    x, y = np.concatenate(features), np.concatenate(labels)
    weights = np.concatenate(sample_weights)
    if np.unique(y).size == 1:
        return float(y[0])
    center = x.mean(0).astype(np.float32)
    scale = np.maximum(x.std(0), 1.e-3).astype(np.float32)
    x = np.clip((x-center)/scale,-8,8)
    x, y, weights = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (x, y, weights/weights.mean())]
    torch.manual_seed(3121 + 3*head + variant)
    generator = torch.Generator(device=device).manual_seed(8313 + head)
    model = nn.Sequential(nn.Linear(x.shape[1],96), nn.ReLU(), nn.Dropout(.12),
                           nn.Linear(96,64), nn.ReLU(), nn.Dropout(.12), nn.Linear(64,1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.01, fused=True)
    epochs = 28
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00015)
    for epoch in range(epochs):
        plan = torch.randperm(len(y), device=device, generator=generator)
        for batch in plan.split(8192):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[batch], weight=weights[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch in (0, 13, epochs-1):
            print(f'CUDA fit head={head} soft={variant} n={len(y)} epoch={epoch+1} loss={loss.item():.5f}', flush=True)
    values = [(part.weight.detach().cpu().numpy(), part.bias.detach().cpu().numpy()) for part in model
              if isinstance(part, nn.Linear)]
    return PortableReadout(center, scale, values)


def fit_gpu(args):
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('The independent regional readouts require CUDA training')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    patient_rows = rows(args.cohort, 'train', truth=True)
    patient_ids = [r['case_id'] for r in patient_rows]
    sources = residual._sample_ids(args.samples, patient_rows)
    features_root = getattr(args, 'features', None)
    if {('features' in r['binding']) for r in sources.values()} != {bool(features_root)}:
        raise ValueError('Training and calibration require the same continuous feature inputs')
    feature_schema = None
    if features_root:
        from .expert_features import SCHEMA

        schemas = {record['binding'].get('feature_schema', SCHEMA) for record in sources.values()}
        if len(schemas) != 1:
            raise ValueError('Cannot combine different continuous feature schemas in the same fit')
        feature_schema = schemas.pop()
    heldout = [r for i,r in enumerate(patient_rows) if i%5==4]
    in_fold = [r['case_id'] for i,r in enumerate(patient_rows) if i%5!=4]
    h = args.head
    path = Path(args.output)/f'{dense.HEADS[h]}.joblib'
    dice_weighting = getattr(args, 'dice_patient_weighting', False)
    binding = {'schema':residual.MODEL_SCHEMA, 'cohort':file_identity(args.cohort), 'head':h,
               'train_ids':patient_ids, 'calibration_ids':[r['case_id'] for r in heldout],
               'samples':{key:r['arrays'] for key,r in sources.items()},
               'code_sha256':file_identity(residual.__file__)['sha256'],
               'trainer':'gpu_readout', 'trainer_sha256':file_identity(__file__)['sha256']}
    if dice_weighting:
        binding['case_weighting'] = 'bounded_sqrt_inverse_dice_volume'
    if feature_schema:
        binding['feature_schema'] = feature_schema
    if path.exists():
        if joblib.load(path)['binding'] != binding:
            raise ValueError('GPU model output is bound to another experiment')
        return path
    calibration_weights = _patient_dice_weights(sources, in_fold, h) if dice_weighting else None
    provisional = [_train(sources, in_fold, v, h, device, case_weights=calibration_weights) for v in range(2)]
    gains = [[], []]
    for row in heldout:
        clean = {k:v for k,v in row.items() if k!='gt'}
        mri, base, ref, sam, iou, replies, sam_receipt, native = residual._evidence(clean,args.dense,args.responses)
        features, identity = residual._precomputed_features(features_root, clean, sam_receipt, native)
        if (identity != sources[row['case_id']]['binding'].get('features')
                or features is not None and features['schema'] != feature_schema):
            raise ValueError('Calibration features differ from the sampled training evidence')
        target = residual._truth(row,ref) > h
        for v in range(2):
            idx, x = residual._case_observations(features,mri,base,sam,replies,iou,ref.affine,v,h)
            p = residual.posterior(provisional[v],x)
            native = residual._native_point_values(replies[v, h], base > h, idx) if features_root else None
            gains[v].append(residual._gains(p, idx, base > h, target, native))
        print(f'CUDA readout calibration {dense.HEADS[h]} {row["case_id"]}', flush=True)
    options = ((*residual.THRESHOLDS, residual.NATIVE_POLICY, None) if features_root
               else (*residual.THRESHOLDS, None))
    final = []
    final_weights = _patient_dice_weights(sources, patient_ids, h) if dice_weighting else None
    for v, scores in enumerate(gains):
        scores = np.stack(scores)
        benefit = scores.mean(axis=0)-.20*np.maximum(0.,-scores).mean(axis=0)
        if benefit.max() > 0:
            i,j = np.unravel_index(np.argmax(benefit),benefit.shape)
        else:
            i,j = len(options)-1,len(options)-1
        final.append({'add':options[i],'remove':options[j],
                      'calibration_gain_pp':float(scores[:,i,j].mean()),
                      'calibration_per_case_pp':scores[:,i,j].tolist(),
                      'model':_train(sources,patient_ids,v,h,device,case_weights=final_weights)})
    dump_atomic({'binding':binding,'variants':final},path)
    report = {'binding':binding,'variants':[{k:v for k,v in f.items() if k!='model'} for f in final],
              'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
              'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30}
    write_json(path.with_suffix('.json'),report)
    return path
