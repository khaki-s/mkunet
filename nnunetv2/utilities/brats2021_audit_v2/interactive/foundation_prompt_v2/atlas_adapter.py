"""Optional AtlasSegFM post-fusion ablation; never part of prompt-only inference.

Loads the upstream FusionAdapter and loss verbatim from a pinned source snapshot.
Only train320 labels can optimize this adapter. Inference accepts no ground truth.
The 4 MRI channels replace upstream img_ch=1 through its existing constructor.
Binary masks become signed-distance pseudo-probabilities, NOT model confidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any
from collections import OrderedDict

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn
from torch.nn import functional as F

DEFAULT_SOURCE = Path('/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113/third_party/AtlasSegFM/step3_fusion_adapter.py')
DEFAULT_SPLITS = Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021/splits_final.json')
SOURCE_SHA256 = 'b7a42de53d906ceef17232408ff70c06b99c251f00d732a435cce19c75431032'
SOURCE_COMMIT = '5bd08794bdc6bc2afc20af3ff7055f9d417428b1'
HEADS = ('WT', 'TC', 'ET')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _upstream(source=DEFAULT_SOURCE):
    source = Path(source)
    if _sha(source) != SOURCE_SHA256:
        raise ValueError('Atlas source snapshot changed; review and repin before using it')
    # Select definitions only: never execute the upstream demo, query-GT reader,
    # allocator environment mutation, or its file I/O imports.
    wanted = {'FusionAdapter', 'soft_prob_from_binary', 'build_oracle_gate', 'fusion_loss'}
    parsed = ast.parse(source.read_text())
    body = [node for node in parsed.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    if {node.name for node in body} != wanted:
        raise ValueError('Missing pinned upstream symbols')
    module = ast.Module(body=body, type_ignores=[])
    ns = {'torch': torch, 'nn': nn, 'F': F, 'np': np,
          'distance_transform_edt': distance_transform_edt}
    exec(compile(module, str(source), 'exec'), ns)
    return ns


def get_adapter(*, source=DEFAULT_SOURCE, img_ch=4, base_ch=16, device='cpu'):
    if img_ch != 4:
        raise ValueError('BraTS adapter requires all four MRI channels')
    return _upstream(source)['FusionAdapter'](img_ch=img_ch, base_ch=base_ch).to(device)


def _head_index(head):
    if isinstance(head, str):
        if head not in HEADS:
            raise ValueError(f'Unknown head {head}; expected {HEADS}')
        return HEADS.index(head)
    h = int(head)
    if h not in (0, 1, 2):
        raise ValueError('head must be 0=WT, 1=TC, or 2=ET')
    return h


def _region(labels, head):
    labels = np.asarray(labels)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError('Expected internal labels 0=BG, 1=ED, 2=NCR, 3=ET')
    h = _head_index(head)
    return labels > 0 if h == 0 else (labels >= 2 if h == 1 else labels == 3)


def make_features(mri, baseline_binary, fm_binary, *, source=DEFAULT_SOURCE):
    """Return (image, fm_soft, baseline_soft), tensors with leading batch axis.

    The upstream class performs its original intensity normalization and builds
    10 channels internally: 4 MRI + 2 logits + difference + disagreement + 2
    entropies. Entropy here measures the SDF surrogate, not calibrated uncertainty.
    """
    image = np.asarray(mri, dtype=np.float32)
    baseline = np.asarray(baseline_binary)
    fm = np.asarray(fm_binary)
    if image.ndim != 4 or image.shape[0] != 4 or image.shape[1:] != baseline.shape or fm.shape != baseline.shape:
        raise ValueError('Expected MRI[4,D,H,W] and matching binary masks[D,H,W]')
    if not np.isfinite(image).all() or not np.isin(baseline, [0, 1]).all() or not np.isin(fm, [0, 1]).all():
        raise ValueError('MRI must be finite and masks must be binary')
    to_mask = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    soft = _upstream(source)['soft_prob_from_binary']
    return (torch.from_numpy(np.ascontiguousarray(image))[None],
            soft(to_mask(fm)), soft(to_mask(baseline)))


def _read_manifest(path, splits_path):
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get('schema') != 'foundation_adapter_pairs_v1' or meta.get('split') != 'train':
        raise ValueError('Adapter training requires foundation_adapter_pairs_v1 with split=train')
    splits = json.loads(Path(splits_path).read_text())
    train_ids, val_ids = set(splits[0]['train']), set(splits[0]['val'])
    if train_ids & val_ids:
        raise ValueError('Training/validation split overlap')
    records = meta.get('cases', [])
    if not records or any(r['case_id'] not in train_ids for r in records):
        raise ValueError('Every adapter pair must belong to fixed train320; validation/test forbidden')
    reference = meta.get('source_manifest')
    if reference is not None:
        source_path = reference.get('path') if isinstance(reference, dict) else reference
        source_path = (path.parent / source_path).resolve()
        if isinstance(reference, dict) and reference.get('sha256') != _sha(source_path):
            raise ValueError('Source case manifest changed')
        source_cases = json.loads(source_path.read_text())
        allowed = {r['case_id'] for r in source_cases.get('cases', [])}
        if source_cases.get('split') != 'train' or not allowed <= train_ids or any(r['case_id'] not in allowed for r in records):
            raise ValueError('Pair source must be a train case manifest with matching patient IDs')
        meta['case_source_identity'] = {'path': str(source_path), 'sha256': _sha(source_path)}
    for record in records:
        record['file'] = str((path.parent / record['file']).resolve())
        if not Path(record['file']).is_file():
            raise FileNotFoundError(record['file'])
    return meta, records


def _load_pair(record):
    with np.load(record['file'], allow_pickle=False) as pair:
        missing = set(('mri', 'baseline', 'proposal', 'gt', 'head')) - set(pair.files)
        if missing:
            raise ValueError(f'Missing NPZ fields: {missing}')
        mri = pair['mri'].astype(np.float32)
        baseline, proposal, gt = (pair[k] for k in ('baseline', 'proposal', 'gt'))
        head = _head_index(pair['head'].item())
    if baseline.shape != gt.shape or proposal.shape != gt.shape:
        raise ValueError('Pair label shapes do not match')
    return mri, _region(baseline, head), _region(proposal, head), _region(gt, head), head


def _crop_tensors(items, patch_size, rng, focus):
    shape = np.array(items[0].shape[-3:])
    patch = np.minimum(np.array(patch_size), shape)
    coords = np.argwhere(focus)
    if len(coords) and rng.random() < .75:
        center = coords[rng.integers(len(coords))]
    else:
        center = np.array([rng.integers(n) for n in shape])
    start = np.clip(center - patch // 2, 0, shape - patch)
    sl = tuple(slice(int(s), int(s + p)) for s, p in zip(start, patch))
    return [item[(...,) + sl] for item in items]


def fit_adapter(samples_metadata, output, *, epochs=30, device='cpu',
                patch_size=(64, 64, 64), steps_per_epoch=None, base_ch=16,
                seed=20261113, splits_path=DEFAULT_SPLITS, source=DEFAULT_SOURCE,
                heads=HEADS, sampling='disagreement', cache_pairs=0):
    """Train one region-specific adapter for every head present in train pairs.

    NPZ schema: mri[4,D,H,W], baseline/proposal/gt[D,H,W] internal labels,
    head scalar int(0=WT,1=TC,2=ET). Patches retain all MRI modalities.
    The original upstream loss and Adam LR/milestones are retained; training
    across train320 and patch sampling are project adaptations, not exact paper
    one-shot test-time adaptation. Output must not already exist.
    """
    if epochs < 1 or min(patch_size) < 4 or (steps_per_epoch is not None and steps_per_epoch < 1):
        raise ValueError('epochs/steps must be positive; patch dimensions >=4')
    if not heads or len(set(heads)) != len(heads) or any(h not in HEADS for h in heads):
        raise ValueError('heads must be a nonempty unique subset of WT/TC/ET')
    if sampling not in ('disagreement', 'legacy'):
        raise ValueError('Unknown patch sampling strategy')
    if type(cache_pairs) is not int or cache_pairs < 0:
        raise ValueError('cache_pairs must be a nonnegative integer')
    meta, records = _read_manifest(samples_metadata, splits_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an adapter run: {output}')
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    upstream = _upstream(source)
    grouped = {h: [] for h in range(3)}
    for record in records:
        with np.load(record['file'], allow_pickle=False) as z:
            grouped[_head_index(z['head'].item())].append(record)
    models, losses = {}, []
    for head, head_records in grouped.items():
        if not head_records or HEADS[head] not in heads:
            continue
        # Independent seeds make serial and parallel head training equivalent.
        torch.manual_seed(seed + head)
        rng = np.random.default_rng(seed + head)
        model = get_adapter(source=source, base_ch=base_ch, device=device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[15, 25], gamma=.3)
        cache = OrderedDict()
        for epoch in range(epochs):
            values = []
            order = rng.permutation(len(head_records))
            steps = steps_per_epoch or len(head_records)
            model.train()
            for step in range(steps):
                record = head_records[int(order[step % len(order)])]
                key = record['file']
                if key in cache:
                    tensors, focus = cache.pop(key)
                    cache[key] = (tensors, focus)
                else:
                    mri, baseline, fm, gt, _ = _load_pair(record)
                    image, fm_soft, baseline_soft = make_features(mri, baseline, fm, source=source)
                    binary = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None]
                    fm_t, base_t, gt_t = binary(fm), binary(baseline), binary(gt)
                    target, valid = upstream['build_oracle_gate'](fm_t, base_t, gt_t)
                    # Only source disagreements can be corrected by a convex gate.
                    focus = (baseline != fm) if sampling == 'disagreement' else ((baseline != fm) | gt)
                    if not focus.any():
                        focus = gt
                    tensors = [image, fm_soft, baseline_soft, gt_t, target, valid]
                    if cache_pairs:
                        cache[key] = (tensors, focus)
                        while len(cache) > cache_pairs:
                            cache.popitem(last=False)
                # Cached features stay on CPU; crops move to the selected GPU.
                # Sampling still runs every step with the same RNG sequence.
                items = _crop_tensors(tensors, patch_size, rng, focus)
                image, fm_soft, baseline_soft, gt_t, target, valid = [x.to(device) for x in items]
                pred, aux = model(image, fm_soft, baseline_soft)
                loss = upstream['fusion_loss'](pred, gt_t, aux, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite Atlas adapter loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                values.append(float(loss.detach()))
            sched.step()
            row = {'head': HEADS[head], 'epoch': epoch + 1, 'loss': float(np.mean(values)), 'steps': steps}
            losses.append(row)
            print(json.dumps(row), flush=True)
        models[str(head)] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    provenance = {'schema': 'atlas_bratspostfusion_v1', 'source_commit': SOURCE_COMMIT,
                  'source_sha256': _sha(source), 'source_manifest': str(Path(samples_metadata).resolve()),
                  'manifest_sha256': _sha(samples_metadata), 'split_sha256': _sha(splits_path),
                  'case_source_identity': meta.get('case_source_identity'),
                  'split': 'train', 'case_ids': sorted({r['case_id'] for r in records}),
                  'pair_files': [{'file': r['file'], 'sha256': _sha(r['file'])} for r in records],
                  'base_ch': base_ch, 'img_ch': 4, 'patch_size': list(patch_size),
                  'epochs': epochs, 'seed': seed, 'heads': sorted(models),
                  'sampling': sampling, 'steps_per_epoch': steps_per_epoch,
                  'cache_pairs': cache_pairs,
                  'head_seed_policy': 'seed+head_index',
                  'pseudo_probability': 'sigmoid(signed_voxel_EDT/2); not calibrated model probability',
                  'adaptation': 'offline train320 region-specific patch training; no registration or query GT'}
    checkpoint = output / 'checkpoint_last.pth'
    torch.save({'metadata': provenance, 'models': models}, checkpoint)
    (output / 'metadata.json').write_text(json.dumps(provenance, indent=2))
    (output / 'losses.json').write_text(json.dumps(losses, indent=2))
    return str(checkpoint)


def merge_adapters(checkpoints, output):
    """Combine independently optimized heads, requiring identical provenance."""
    from .common import write_json, file_identity
    root = Path(output)
    if root.exists():
        raise FileExistsError('Use a new merged adapter output directory')
    models, reference, losses = {}, None, []
    for path in checkpoints:
        state = torch.load(path, map_location='cpu', weights_only=True)
        meta = dict(state['metadata'])
        heads = meta.pop('heads')
        if set(heads) != set(state['models']) or meta.get('schema') != 'atlas_bratspostfusion_v1':
            raise ValueError('Invalid adapter head metadata')
        if reference is not None and reference != meta:
            raise ValueError('Adapter heads have different training data, protocol, or configuration')
        if models.keys() & state['models'].keys():
            raise ValueError('Duplicate adapter head')
        reference = meta
        models.update(state['models'])
        loss_path = Path(path).parent / 'losses.json'
        if loss_path.exists():
            losses.extend(json.loads(loss_path.read_text()))
    if set(models) != {'0', '1', '2'}:
        raise ValueError('Require exactly WT, TC and ET heads')
    reference.update(heads=sorted(models), merged_from=[file_identity(p) for p in checkpoints])
    root.mkdir(parents=True)
    path = root / 'checkpoint_last.pth'
    torch.save({'metadata': reference, 'models': models}, path)
    write_json(root / 'metadata.json', reference)
    write_json(root / 'losses.json', losses)
    return path


def _starts(n, p):
    if n <= p:
        return [0]
    starts = list(range(0, n - p + 1, max(1, p // 2)))
    return starts if starts[-1] == n - p else starts + [n - p]


@torch.inference_mode()
def fuse_adapter(checkpoint, current, fm, mri, head, *, device='cpu', source=DEFAULT_SOURCE):
    """Fuse one head from two multilabel maps, with no GT argument or file read.

    Returns dict probability, binary, gate, metadata. Caller must apply the same
    WT/TC/ET hierarchy policy as the surrounding pipeline. The gate weights the
    original mask (1=original, 0=FM). If sources agree, logit convexity preserves
    their binary label: neither branch can fix a shared error through fusion.
    """
    h = _head_index(head)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta = saved['metadata']
    if meta.get('schema') != 'atlas_bratspostfusion_v1' or meta.get('source_sha256') != _sha(source):
        raise ValueError('Incompatible checkpoint or changed upstream source')
    if str(h) not in saved['models']:
        raise ValueError(f'No trained adapter for {HEADS[h]}')
    baseline, proposed = _region(current, h), _region(fm, h)
    image, fm_soft, baseline_soft = make_features(mri, baseline, proposed, source=source)
    model = get_adapter(source=source, base_ch=meta['base_ch'], device=device)
    model.load_state_dict(saved['models'][str(h)], strict=True)
    model.eval().requires_grad_(False)
    shape = baseline.shape
    probability = baseline_soft[0, 0].numpy().copy()
    gate = np.ones(shape, dtype=np.float32)
    diff = np.argwhere(baseline != proposed)
    if len(diff):
        # Only disagreement can change the binary result under convex logit fusion.
        lower = np.maximum(diff.min(0) - 8, 0)
        upper = np.minimum(diff.max(0) + 9, shape)
        roi_shape = upper - lower
        patch = np.minimum(np.array(meta['patch_size']), np.array(shape))
        lower = np.maximum(np.minimum(lower, np.array(shape) - patch), 0)
        upper = np.minimum(np.maximum(upper, lower + patch), shape)
        roi_shape = upper - lower
        accum = np.zeros(tuple(roi_shape), np.float32)
        gates = np.zeros_like(accum)
        count = np.zeros_like(accum)
        import itertools
        for start in itertools.product(*[_starts(int(n), int(p)) for n, p in zip(roi_shape, patch)]):
            end = np.minimum(np.array(start) + patch, roi_shape)
            local = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
            native = tuple(slice(int(l+s), int(l+e)) for l, s, e in zip(lower, start, end))
            inputs = [x[(...,) + native].to(device) for x in (image, fm_soft, baseline_soft)]
            prob, aux = model(*inputs)
            accum[local] += prob[0, 0].cpu().numpy()
            gates[local] += aux['g'][0, 0].cpu().numpy()
            count[local] += 1
        if (count == 0).any():
            raise RuntimeError('Inference patch coverage failed')
        roi = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        probability[roi] = accum / count
        gate[roi] = gates / count
    binary = probability >= .5
    if np.any(binary[baseline == proposed] != baseline[baseline == proposed]):
        raise AssertionError('Convex logit fusion violated agreement invariance')
    return {'probability': probability, 'binary': binary.astype(np.uint8), 'gate': gate,
            'metadata': {'head': HEADS[h], 'checkpoint_sha256': _sha(checkpoint),
                         'source_sha256': _sha(source), 'gate_semantics': '1=baseline,0=FM',
                         'gt_used': False, 'optional_postfusion_ablation': True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--pairs', required=True)
    train.add_argument('--output', required=True)
    train.add_argument('--epochs', type=int, default=30)
    train.add_argument('--steps-per-epoch', type=int)
    train.add_argument('--patch-size', type=int, nargs=3, default=(64, 64, 64))
    train.add_argument('--splits', default=str(DEFAULT_SPLITS))
    train.add_argument('--device', default='cuda')
    pred = commands.add_parser('predict')
    pred.add_argument('--checkpoint', required=True)
    pred.add_argument('--input', required=True, help='NPZ with mri,baseline,proposal,head; GT prohibited')
    pred.add_argument('--output', required=True)
    pred.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.command == 'train':
        return fit_adapter(args.pairs, args.output, epochs=args.epochs, device=args.device,
                           patch_size=args.patch_size, steps_per_epoch=args.steps_per_epoch,
                           splits_path=args.splits)
    with np.load(args.input, allow_pickle=False) as z:
        if any(k in z.files for k in ('gt', 'ground_truth', 'label', 'labels')):
            raise ValueError('Prediction input must not contain ground truth')
        result = fuse_adapter(args.checkpoint, z['baseline'], z['proposal'], z['mri'],
                              z['head'].item(), device=args.device)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, probability=result['probability'], binary=result['binary'], gate=result['gate'])
    Path(str(args.output) + '.json').write_text(json.dumps(result['metadata'], indent=2))


if __name__ == '__main__':
    main()
