"""Test sparse SAM-supported strokes with the original 192³ interactive checkpoint."""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch

from ...inference_runtime import prediction_context
from ..common import check_file, file_identity, read_json, write_json
from . import candidates, dense
from .cohorts import rows


def local_stroke(component, point, spacing, radius_mm=6.):
    center = np.asarray(point)
    width = np.ceil(radius_mm / spacing).astype(int)
    low = np.maximum(0, center - width)
    high = np.minimum(component.shape, center + width + 1)
    patch = tuple(slice(a, b) for a, b in zip(low, high))
    points = np.argwhere(component[patch]) + low
    if len(points) <= 1:
        return None
    relative = (points - center) * spacing
    nearby = np.linalg.norm(relative, axis=1) <= radius_mm
    points, relative = points[nearby], relative[nearby]
    if len(points) < 3:
        return None
    _, _, basis = np.linalg.svd(relative - relative.mean(axis=0), full_matrices=False)
    axis = basis[0]
    aligned = relative @ axis
    transverse = np.linalg.norm(relative - aligned[:, None] * axis, axis=1)
    selected = points[transverse <= 1.05 * np.min(spacing)]
    local = np.zeros(tuple(high - low), bool)
    local[tuple((selected - low).T)] = True
    labels, _ = ndimage.label(local, ndimage.generate_binary_structure(3, 3))
    seed = labels[tuple(center - low)]
    line = labels == seed
    if not seed or np.count_nonzero(line) < 3:
        return None
    return line.astype(np.uint8), np.column_stack((low, high)).tolist()


def response(executor, baseline, prompt, stroke):
    payload, bounds = stroke
    initial = baseline.astype(np.uint8)
    destination = initial.copy()
    engine = executor.engine
    engine.set_target_buffer(destination)
    positive = prompt['positive']
    with prediction_context(engine):
        engine.add_initial_seg_interaction(initial, run_prediction=False)
        engine.add_scribble_interaction(payload, positive, run_prediction=False, interaction_bbox=bounds)
        engine.add_bbox_interaction(prompt['box'], positive, run_prediction=False)
        engine.add_point_interaction(prompt['point'], positive, run_prediction=True)
    return destination.astype(bool)


@torch.inference_mode()
def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < candidates.MIN_FREE_MIB * 2**20:
        raise RuntimeError('Insufficient free CUDA memory for native 192³ responses')
    torch.cuda.reset_peak_memory_stats(device)
    runner = None
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    for row in selected:
        _, _, sam = dense.read_masks(args.dense, row)
        control = read_json(Path(args.control) / row['case_id'] / 'responses.json')
        if (sam['arrays'] != control['sam'] or control['row'] != row or control['gt_used']
                or control['kinds'] != list(candidates.RESPONSE_NAMES) or 'native_margin' in control['executor']):
            raise ValueError('Control and stroke treatments require the same frozen SAM3 prompts')
        check_file(control['arrays'])
        out = Path(args.output) / row['case_id']
        receipt = out / 'responses.json'
        method = {**candidates.executor_configuration(), 'stroke_probe_code': file_identity(__file__)['sha256'],
                  'control': control['arrays']}
        if receipt.is_file():
            existing = read_json(receipt)
            if existing['row'] != row or existing['executor'] != method or existing['sam'] != sam['arrays']:
                raise ValueError('Saved stroke response has different inputs')
            check_file(existing['arrays'])
            continue
        mri, label, ref = dense.load_native(row)
        base = label > 0
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        with np.load(control['arrays']['path'], allow_pickle=False) as data:
            packed = data['packed']
        with np.load(sam['arrays']['path'], allow_pickle=False) as data:
            core = data['core_components'][0]
            allowed = data['components'][0]
        if runner is None:
            runner = candidates.NativeExecutor(str(device))
        channel = next(k for k in dense.PREFERRED_MODALITIES[0] if row['availability'][k])
        runner.engine.set_image(np.ascontiguousarray(mri[channel][None], dtype=np.float32))
        updated = np.unpackbits(packed[:, 0, 2], axis=-1, count=label.size).reshape(2, *label.shape).astype(bool)
        count = 0
        for action in sam['actions']:
            if action['head'] != 0 or action['positive']:
                continue
            comp = core == action['id']
            support = allowed == action['id']
            performed = {}
            for v, candidate in enumerate(action['variants']):
                if not candidate['accept']:
                    continue
                prompt = candidate['prompt']
                key = tuple(prompt['point']), tuple(map(tuple, prompt['box']))
                if key not in performed:
                    stroke = local_stroke(comp, prompt['point'], spacing)
                    performed[key] = response(runner, base, prompt, stroke) if stroke else None
                if performed[key] is not None:
                    updated[v, support] = (base & performed[key])[support]
                    count += 1
        packed[:, 0, 2] = np.packbits(updated.reshape(2, -1), axis=-1)
        out.mkdir(parents=True, exist_ok=True)
        arrays = out / 'responses.npz'
        dense._atomic_npz(arrays, packed=packed, shape=np.asarray(label.shape))
        write_json(receipt, {'row': row, 'executor': method, 'sam': sam['arrays'],
                             'kinds': list(candidates.RESPONSE_NAMES), 'arrays': file_identity(arrays),
                             'gt_used': False, 'wt_removal_responses': count,
                             'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30})
        print(f'{row["case_id"]}: {count} negative SAM3 stroke responses', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('cohort', 'dense', 'control', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    run(parser.parse_args())
