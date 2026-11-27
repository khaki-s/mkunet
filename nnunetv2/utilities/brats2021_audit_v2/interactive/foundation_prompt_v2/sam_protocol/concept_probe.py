"""Test a train-only concept embedding for the frozen SAM3 grounding pathway."""
import argparse
import importlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..common import file_identity, manifest, write_json
from ..sam_dense import SAM_CHECKPOINT, SAM_SOURCE
from . import appearance, dense, image_memory, scoring
from .cohorts import rows

TEXT = 'glioma and surrounding edema in brain MRI'


def load(device):
    if torch.cuda.get_device_capability(device)[0] < 8:
        autocast = torch.autocast

        def compat(device_type, dtype=None, *args, **kwargs):
            if device_type == 'cuda' and dtype == torch.bfloat16:
                dtype = torch.float16
            return autocast(device_type, dtype, *args, **kwargs)

        sys.path.insert(0, SAM_SOURCE)
        try:
            with patch.object(torch, 'autocast', compat):
                importlib.import_module('sam3.model_builder')
                importlib.import_module('sam3.model.sam3_image_processor')
        finally:
            sys.path.pop(0)
    model, processor = image_memory.load_image_only(device, stream_blocks=20)
    model.requires_grad_(False).eval()
    model.inst_interactive_predictor.cpu()
    model.inst_interactive_predictor = None
    for part in (model.geometry_encoder, model.transformer, model.dot_prod_scoring, model.segmentation_head):
        part.to(device)
    with torch.no_grad():
        text = model.backbone.forward_text([TEXT], device='cpu')
    text = {k: v.to(device) for k, v in text.items() if torch.is_tensor(v)}
    return model, processor, text


def regular(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [regular(entry) for entry in value]
    if isinstance(value, dict):
        return {key: regular(entry) for key, entry in value.items()}
    return value


def encode(processor, flair, z):
    valid = flair[::2, ::2, ::2]
    valid = valid[valid != 0]
    lo, hi = np.percentile(valid, (1, 99))
    gray = (255. * np.clip((flair[..., z].astype(np.float32) - lo) / max(float(hi - lo), 1.e-6), 0, 1)).astype(np.uint8)
    image = Image.fromarray(np.repeat(gray[..., None], 3, axis=2))
    with torch.autocast('cuda', dtype=torch.float16):
        result = processor.set_image(image)['backbone_out']
    return regular(result)


def ground(model, processor, visual, text, delta):
    state = {**visual, **text, 'language_features': text['language_features'] + delta}
    prediction = model.forward_grounding(state, processor.find_stage, None, model._get_dummy_prompt())
    return prediction['pred_masks'][0].float(), prediction['pred_logits'][0, :, 0].float()


def objective(masks, logits, target):
    target = F.interpolate(target[None, None], size=masks.shape[-2:], mode='nearest')[0, 0]
    with torch.no_grad():
        submitted = masks > 0
        overlap = (submitted & target.bool()).sum((1, 2))
        union = (submitted | target.bool()).sum((1, 2))
        match = (overlap / union.clamp_min(1)).argmax()
    active = masks[match]
    probability = active.sigmoid()
    true_positive = (probability * target).sum()
    union = probability.sum() + target.sum() - true_positive
    cls = torch.zeros_like(logits)
    cls[match] = 1.
    return (F.binary_cross_entropy_with_logits(active, target) + 1. - (true_positive + 1) / (union + 1)
            + .25 * F.binary_cross_entropy_with_logits(logits, cls))


def dice(mask, target):
    common = np.count_nonzero(mask & target)
    total = np.count_nonzero(mask) + np.count_nonzero(target)
    return float(2. * common / total) if total else 1.


def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    model, processor, text = load(device)
    if processor.resolution != 1008:
        raise ValueError('Keep the official SAM3 image input')
    delta = torch.nn.Parameter(torch.zeros_like(text['language_features'], dtype=torch.float32))
    optimizer = torch.optim.AdamW([delta], lr=.0005)
    train = rows(args.cohort, 'train', truth=True)
    chosen = np.linspace(0, len(train) - 1, args.support, dtype=int)
    targets, support = [], []
    for index in chosen:
        row = train[index]
        volumes, _, _, truth, _ = appearance.read_case(args.cache, row, 'train')
        z = int((truth > 0).sum((0, 1)).argmax())
        visual = encode(processor, volumes[3], z)
        support.append({key: regular(value).cpu() if torch.is_tensor(value) else
                        [item.cpu() for item in value] if isinstance(value, list) else value
                        for key, value in visual.items() if key != 'sam2_backbone_out'})
        targets.append(torch.from_numpy((truth[..., z] > 0).astype(np.float32)))
        print(f'Frozen 1008² train support {row["case_id"]} slice {z}', flush=True)
    losses = []
    for step in range(args.steps):
        at = step % len(support)
        visual = {key: value.to(device) if torch.is_tensor(value) else
                  [item.to(device) for item in value] if isinstance(value, list) else value
                  for key, value in support[at].items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            masks, logits = checkpoint(lambda learned: ground(model, processor, visual, text, learned),
                                       delta, use_reentrant=False)
        value = objective(masks, logits, targets[at].to(device))
        value.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError('Nonfinite concept embedding gradient')
        torch.nn.utils.clip_grad_norm_([delta], 3.)
        optimizer.step()
        losses.append(float(value.detach()))
        print(f'concept step {step + 1} loss={losses[-1]:.5f} peak_gib={torch.cuda.max_memory_reserved(device)/2**30:.3f}', flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights = output / 'embedding.pth'
    torch.save({'embedding': delta.detach().cpu(), 'text': TEXT,
                'train_ids': [train[i]['case_id'] for i in chosen]}, weights)
    pred = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        validation = rows(args.cohort, 'validation')[:args.validation]
        for row in validation:
            volumes, old, _, _, _ = appearance.read_case(args.cache, row, 'validation')
            z = int((old > 0).sum((0, 1)).argmax())
            canvas = encode(processor, volumes[3], z)
            masks = {}
            for name, shift in (('text', torch.zeros_like(delta)), ('adapted', delta)):
                batch, logits = ground(model, processor, canvas, text, shift)
                selected = batch[logits.argmax()]
                converted = F.interpolate(selected[None, None], size=old.shape[:2], mode='bilinear',
                                          align_corners=False)[0, 0] > 0
                masks[name] = converted.cpu().numpy()
            path = output / f'{row["case_id"]}.npz'
            dense._atomic_npz(path, **masks)
            pred.append({'row': row, 'slice': z, 'masks': file_identity(path)})
    write_json(output / 'predictions.json', {'records': pred, 'gt_used': False, 'embedding': file_identity(weights)})
    refs = {row['case_id']: row for row in manifest(args.truth, allowed=('validation',))['cases']}
    result = []
    for item in pred:
        row, z = item['row'], item['slice']
        old, target, _, _ = scoring._image_pair(row, refs[row['case_id']])
        mask = target[..., z] > 0
        with np.load(item['masks']['path'], allow_pickle=False) as predicted:
            result.append({'case_id': row['case_id'], 'slice': z, 'baseline': dice(old[..., z] > 0, mask),
                           **{key: dice(predicted[key], mask) for key in predicted.files}})
    write_json(output / 'score.json', {'scope': 'single axial slice per case, concept smoke only',
               'sam_checkpoint': file_identity(SAM_CHECKPOINT), 'encoder_resolution': processor.resolution,
               'train_losses': losses, 'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30,
               'per_case': result})
    print(result, flush=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description=__doc__)
    for arg in ('cohort', 'cache', 'truth', 'output'):
        parse.add_argument('--' + arg, required=True)
    parse.add_argument('--device', default='cuda:0')
    parse.add_argument('--support', type=int, default=4)
    parse.add_argument('--steps', type=int, default=16)
    parse.add_argument('--validation', type=int, default=3)
    run(parse.parse_args())
