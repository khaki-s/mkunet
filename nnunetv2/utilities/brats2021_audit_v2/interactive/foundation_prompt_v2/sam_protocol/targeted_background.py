"""Training-only controlled ablation of negative points in SAM's proposed expansion."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..common import check_file, file_identity, read_json, write_json
from . import background, candidates, dense, residual
from .cohorts import rows

SCHEMA = 'targeted_wt_background_ablation_v1'


def _selected(cohort):
    return rows(cohort, 'train', truth=True)[::8][:12]


def _points(seed, growth, mri, visible, spacing):
    points, box = dense.prompt_geometry(seed)
    lo_x, lo_y, hi_x, hi_y = box.astype(int)
    corridor = np.zeros_like(seed)
    corridor[lo_y:hi_y + 1, lo_x:hi_x + 1] = True
    eligible = visible & (seed | (growth & corridor))
    negative = background.negative_points(seed, mri, eligible, spacing)
    coords = np.concatenate((points, negative), axis=0)
    labels = np.concatenate((np.ones(len(points), np.int32), np.zeros(len(negative), np.int32)))
    return coords, labels, box


@torch.inference_mode()
def _paired(model, mri, before, previous, spacing, device):
    original = before > 0
    result = previous.copy()
    colors = background._normalize(mri, (3, 2))
    visible = np.any(mri[2:] != 0, axis=0)
    growth = previous.any(axis=0) & ~original
    values = mri[3][mri[3] != 0]
    low, high = np.percentile(values, (1, 99))
    counts = {'changed_slices': 0, 'negative_points': 0}
    for z in np.flatnonzero(original.any(axis=(0, 1))):
        p, labels, box = _points(original[..., z], growth[..., z],
                                [colors[i][..., z] for i in (3, 2)], visible[..., z], spacing[:2])
        n = int(np.count_nonzero(labels == 0))
        if not n:
            continue
        gray = np.clip((mri[3, ..., z] - low) / max(float(high - low), 1.e-6), 0., 1.)
        rgb = Image.fromarray(np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=-1))
        amp = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
        with amp:
            state = model.processor.set_image(rgb)
            priors = dense.soft_prompts(original[..., z], model.mask_size, spacing[:2])
            for variant, prior in enumerate(priors):
                predictions, ranking, _ = model.model.predict_inst(
                    state, point_coords=p, point_labels=labels, box=box, mask_input=prior,
                    multimask_output=True, return_logits=False, normalize_coords=True)
                quality = np.asarray(ranking).reshape(-1)
                predictions = np.asarray(predictions)
                if predictions.shape != (len(quality), *original.shape[:2]) or not np.isfinite(quality).any():
                    raise ValueError('Unexpected output for the targeted WT prompts')
                best = np.argmax(np.where(np.isfinite(quality), quality, -np.inf))
                result[variant, ..., z] = predictions[best] > 0
        del state
        counts['changed_slices'] += 1
        counts['negative_points'] += n
    return result, counts


def targeted_cache(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = None
    for row in _selected(args.cohort):
        key = row['case_id']
        folder = Path(args.output) / key
        clean = {key: value for key, value in row.items() if key != 'gt'}
        cache = folder / 'targeted.json'
        all_original, _, previous = dense.read_masks(args.previous, clean)
        mode = {'schema': SCHEMA, 'rows': clean, 'previous_sam': previous['arrays'],
                'code': file_identity(__file__)['sha256'], 'point_box': 'same original FLAIR; add background points'}
        if cache.exists():
            saved = read_json(cache)
            if saved['binding'] != mode or saved['gt_used'] is not False:
                raise ValueError('Mismatched targeted prompt cache')
            check_file(saved['arrays'])
            continue
        mri, before, ref = dense.load_native(clean)
        if model is None:
            if args.vision_only:
                from .image_memory import load_image_only

                image_model, processor = load_image_only(device)
                model = dense.SamDenseRefiner(str(device), model=image_model, processor=processor)
            else:
                model = dense.SamDenseRefiner(str(device))
        masks, counts = _paired(model, mri, before, all_original[:, 0],
                                np.linalg.norm(ref.affine[:3, :3], axis=0), device)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'targeted.npz'
        dense._atomic_npz(path, masks=masks)
        write_json(cache, {'binding': mode, 'gt_used': False, 'arrays': file_identity(path), **counts})
        print('Targeted WT background', key, counts, flush=True)
    print('SAM reserved GiB', torch.cuda.max_memory_reserved(device)/2**30, flush=True)
    return args.output


def _real_response(engine, image, before, raw, spacing):
    engine.engine.set_image(np.ascontiguousarray(image[None]))
    support = []
    direct = []
    for variant in raw:
        up = candidates.locate_component(variant, before, image, spacing, True)
        down = candidates.locate_component(variant, before, image, spacing, False)
        positive = engine.finite(before, up)
        negative = engine.finite(before, down)
        support.append((positive != before) | (negative != before))
        direct.append((before | (positive & variant)) & (negative | variant))
    return support, direct


def targeted_score(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.min_free_mib < 5200:
        raise ValueError('Minimum free memory must be at least 5200 MiB after initialization')
    if torch.cuda.mem_get_info(device)[0] < args.min_free_mib * 2**20:
        raise RuntimeError(f'Need {args.min_free_mib} MiB free for the original 192³ nnInteractive execution')
    engine = candidates.NativeExecutor(str(device))
    result = []
    for row in _selected(args.cohort):
        key = row['case_id']
        clean = {k: v for k, v in row.items() if k != 'gt'}
        receipt = read_json(Path(args.raw) / key / 'targeted.json')
        if receipt['binding']['rows'] != clean or receipt['gt_used'] is not False:
            raise ValueError('Targeted prompts must have been generated without GT')
        check_file(receipt['arrays'])
        original, _, first = dense.read_masks(args.previous, clean)
        if receipt['binding']['previous_sam'] != first['arrays']:
            raise ValueError('Prompt-only ablation requires exactly the original paired candidates')
        with np.load(receipt['arrays']['path'], allow_pickle=False) as arrays:
            targeted = arrays['masks']
        mri, baseline, ref = dense.load_native(clean)
        base = baseline > 0
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        experiment = [(name, raw, *_real_response(engine, mri[3], base, raw, spacing))
                      for name, raw in (('original', original[:, 0]), ('targeted_negatives', targeted))]
        truth = residual._truth(row, ref) > 0

        def dice(mask):
            n = int(mask.sum() + truth.sum())
            return 200. * np.count_nonzero(mask & truth) / n if n else 100.

        initial = dice(base)
        values = {}
        for name, raw, support, final in experiment:
            for i, variant in enumerate(dense.VARIANTS):
                values[f'{name}/{variant}'] = {'raw_gain_pp': dice(raw[i]) - initial,
                    'actual_192_gain_pp': dice(final[i]) - initial,
                    'nninteractive_candidate_oracle_pp': dice(np.where(support[i], truth, base)) - initial,
                    'raw_added_fp': int(np.count_nonzero(raw[i] & ~base & ~truth)),
                    'raw_recovered_fn': int(np.count_nonzero(raw[i] & ~base & truth))}
        result.append({'case_id': key, 'baseline_WT_pct': initial, **values})
        print('Targeted prompt ablation scored', key, flush=True)
    overview = {method: {metric: float(np.mean([case[method][metric] for case in result]))
                         for metric in result[0][method]}
                for method in result[0] if method not in ('case_id', 'baseline_WT_pct')}
    write_json(args.output, {'schema': SCHEMA, 'split': 'train_only', 'n_cases': len(result),
                            'mean': overview, 'per_case': result,
                            'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30})
    return args.output
