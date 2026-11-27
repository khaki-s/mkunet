"""Convert paired SAM masks into broad, native 192³ nnInteractive candidates."""
import fcntl
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from scipy import ndimage

from ...inference_runtime import NATIVE_PATCH_SIZE, new_session, prediction_context
from ..common import check_file, file_identity, read_json, write_json
from ..conv3d_workspace import install_depth_tiles
from . import dense
from .cohorts import rows

DEFAULT_MODEL = '/data02/chujingyi/model/nnInteractive/nnInteractive_v1.0'
RESPONSE_NAMES = ('dense', 'add_point_box', 'remove_point_box')
DEFAULT_FREE_MIB = 8192
MIN_FREE_MIB = 5200
# These archived executors differ only in the memory entry check and the shard lock.
_PREVIOUS_EXECUTOR_CODES = {
    '27ac99d1d746f2d9daecb03c8654e7b0e255b10901a5fb21c7948edac126a800',
    'be8cd769864a939c85525064eba119faa019fb71ca571de9949d33a4c35706a9',
    '56ad055e5683a35bb6d0ae97747f17c104ced9ea98346951b01b69df4cbd6f23',
    'c1b7be4b39fd1670bd5c21b5958cde32ce06579c7b28edd4b20921698f0a76bc',
}


def model_files(folder=DEFAULT_MODEL):
    return [file_identity(Path(folder) / name) for name in ('plans.json', 'dataset.json', 'fold_0/checkpoint_final.pth')]


def executor_configuration():
    return {'code': file_identity(__file__)['sha256'], 'nninteractive_model': model_files(),
            'input_patch': list(NATIVE_PATCH_SIZE)}


def compatible_executor(saved, current):
    old_code = saved.get('code')
    return saved == current or (old_code in _PREVIOUS_EXECUTOR_CODES and saved == {**current, 'code': old_code})


def image_profile(image, baseline):
    visible = image != 0
    interior = baseline & visible
    body = image[::2, ::2, ::2]
    body = body[body != 0]
    if not len(body):
        return np.zeros(image.shape, dtype=np.float32)
    q15, q85 = np.percentile(body, (15, 85))
    reference = np.median(image[interior]) if interior.any() else np.median(body)
    normalized = (image - reference) / max(float(q85 - q15), 1e-5)
    return np.clip(normalized, -6, 6)


def locate_component(raw, baseline, image, spacing, positive):
    """Choose a supported native component, weighting continuity and image evidence."""
    disagreement = (raw & ~baseline) if positive else (baseline & ~raw)
    disagreement &= image != 0
    if not disagreement.any():
        return None
    labels, count = ndimage.label(disagreement)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ranked = np.argsort(sizes[1:], kind='stable')[-8:][::-1] + 1
    response = image_profile(image, baseline)
    slices = ndimage.find_objects(labels)
    best = None
    for part in ranked:
        if sizes[part] < 8:
            continue
        window = slices[part - 1]
        comp = labels[window] == part
        # For additions, signal similar to the initial lesion is useful. For
        # removals, tissue dissimilar to it is more plausible background.
        appearance = np.abs(response[window])
        contrast = float(np.median(appearance[comp]))
        image_evidence = np.exp(-contrast) if positive else 1. - np.exp(-contrast)
        axial_span = window[2].stop - window[2].start
        value = np.sqrt(sizes[part]) * (.65 + .35 * image_evidence)
        value *= np.sqrt(min(4, axial_span) / 4.)
        if best is None or value > best[0]:
            best = value, part, window, comp, appearance
    if best is None:
        return None
    _, part, window, comp, appearance = best
    depth = ndimage.distance_transform_edt(np.pad(comp, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    interior = np.exp(-appearance) if positive else 1. - np.exp(-appearance)
    priority = depth * (.65 + .35 * interior)
    point_in = np.array(np.unravel_index(np.argmax(priority), comp.shape))
    offset = np.array([s.start for s in window])
    point = point_in + offset
    plane, _ = ndimage.label(comp[:, :, point_in[2]])
    which = plane[tuple(point_in[:2])]
    points = np.argwhere(plane == which) + offset[:2]
    lo = np.maximum(0, np.maximum(points.min(0) - 1, point[:2] - 21))
    hi = np.minimum(baseline.shape[:2], np.minimum(points.max(0) + 2, point[:2] + 22))
    bounds = [[int(a), int(b)] for a, b in zip(lo, hi)] + [[int(point[2]), int(point[2] + 1)]]
    return {'point': point.tolist(), 'box': bounds, 'positive': bool(positive),
            'component_voxels': int(sizes[part]), 'span_slices': int(comp.shape[2]),
            'appearance_distance': float(np.median(appearance[comp]))}


class NativeExecutor:
    def __init__(self, device, model_folder=DEFAULT_MODEL):
        self.device = torch.device(device)
        self.engine = new_session(model_folder, 'cpu', cudnn_benchmark=False)
        if self.device.type == 'cuda':
            self.engine.device = self.device
            install_depth_tiles(self.engine.network, 16)
            self.engine.network.to(self.device)
            with prediction_context(self.engine):
                if hasattr(self.engine, 'warmup'):
                    self.engine.warmup()
        if tuple(self.engine.configuration_manager.patch_size) != NATIVE_PATCH_SIZE:
            raise ValueError('The official nnInteractive checkpoint patch must be 192³')

    def dense(self, seed):
        if not seed.any():
            return seed.copy()
        output = np.zeros(seed.shape, dtype=np.uint8)
        self.engine.set_target_buffer(output)
        with prediction_context(self.engine):
            self.engine.add_initial_seg_interaction(np.ascontiguousarray(seed, dtype=np.uint8), run_prediction=True)
        return output.astype(bool)

    def finite(self, seed, prompt):
        if prompt is None:
            return seed.copy()
        output = np.zeros(seed.shape, dtype=np.uint8)
        self.engine.set_target_buffer(output)
        with prediction_context(self.engine):
            self.engine.add_initial_seg_interaction(np.ascontiguousarray(seed, dtype=np.uint8), run_prediction=False)
            self.engine.add_bbox_interaction(prompt['box'], include_interaction=prompt['positive'], run_prediction=False)
            self.engine.add_point_interaction(prompt['point'], include_interaction=prompt['positive'], run_prediction=True)
        # The change must follow the prompt direction even if the network
        # adjusted unrelated parts of the original mask in the same pass.
        return (output.astype(bool) | seed) if prompt['positive'] else (output.astype(bool) & seed)

    def respond(self, mri, baseline, masks, availability, spacing):
        references = np.stack([baseline > h for h in range(3)])
        if masks.shape != (2, *references.shape):
            raise ValueError('Incompatible SAM and original mask shapes')
        results = np.empty((2, 3, len(RESPONSE_NAMES), *baseline.shape), dtype=bool)
        sources = [next(i for i in prefs if availability[i]) for prefs in dense.PREFERRED_MODALITIES]
        selected_calls = []
        active_channel = None
        for h, base in enumerate(references):
            channel = sources[h]
            image = mri[channel]
            if channel != active_channel:
                self.engine.set_image(np.ascontiguousarray(image[None], dtype=np.float32))
                active_channel = channel
            for v in range(2):
                raw = masks[v, h]
                results[v, h, 0] = self.dense(raw)
                for index, direction in enumerate((True, False), 1):
                    prompt = locate_component(raw, base, image, spacing, direction)
                    results[v, h, index] = self.finite(base, prompt)
                    if prompt:
                        selected_calls.append({'variant': dense.VARIANTS[v], 'head': h, **prompt})
        return results, selected_calls, sources


def read_responses(root, row):
    folder = Path(root) / row['case_id']
    receipt = read_json(folder / 'responses.json')
    if receipt['row'] != row or receipt.get('gt_used') or receipt['kinds'] != list(RESPONSE_NAMES):
        raise ValueError('Interactive response is bound to another patient or input')
    check_file(receipt['arrays'])
    with np.load(receipt['arrays']['path'], allow_pickle=False) as values:
        packed = values['packed']
        shape = tuple(values['shape'])
        values = np.unpackbits(packed, axis=-1, count=np.prod(shape))
    return values.reshape((*packed.shape[:-1], *shape)).astype(bool), receipt


def build_responses(selected, args, raw_dir, output):
    device = torch.device(args.device)
    min_free_mib = getattr(args, 'min_free_mib', DEFAULT_FREE_MIB)
    if min_free_mib < MIN_FREE_MIB:
        raise ValueError(f'At least {MIN_FREE_MIB} MiB must be available for the native 192³ response stage')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    executor = None
    output = Path(output)
    config = executor_configuration()
    start = monotonic()
    fresh = 0
    for row in selected:
        masks, _, source = dense.read_masks(raw_dir, row)
        folder = output / row['case_id']
        receipt_path = folder / 'responses.json'
        if receipt_path.exists():
            prior = read_json(receipt_path)
            if (prior['row'] != row or not compatible_executor(prior['executor'], config)
                    or prior['sam'] != source['arrays']):
                raise ValueError(f'{row["case_id"]}: cannot reuse a changed candidate configuration')
            check_file(prior['arrays'])
            continue
        mri, baseline, reference = dense.load_native(row)
        if executor is None:
            if device.type == 'cuda' and torch.cuda.mem_get_info(device)[0] < min_free_mib * 2**20:
                raise RuntimeError(f'Insufficient free CUDA memory for native 192³ refinement (need {min_free_mib} MiB)')
            executor = NativeExecutor(str(device))
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        masks, prompts, channels = executor.respond(mri, baseline, masks, row['availability'], spacing)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'responses.npz'
        packed = np.packbits(masks.reshape((2, 3, len(RESPONSE_NAMES), -1)), axis=-1)
        dense._atomic_npz(path, packed=packed, shape=np.asarray(baseline.shape))
        write_json(receipt_path, {'row': row, 'executor': config, 'sam': source['arrays'], 'kinds': list(RESPONSE_NAMES),
                                 'prompts': prompts, 'modalities': channels, 'gt_used': False,
                                 'arrays': file_identity(path)})
        fresh += 1
        print(f'nnInteractive {row["case_id"]} mask={row["mask_code"]} elapsed={monotonic()-start:.1f}s', flush=True)
    return {'completed': len(selected), 'new_cases': fresh, 'seconds': monotonic() - start,
            'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else 0,
            'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else 0}


def respond(args):
    selected = rows(args.cohort, 'train') + rows(args.cohort, 'validation')
    selected = selected[args.shard_index::args.num_shards]
    path = Path(args.output) / f'response_shard{args.shard_index}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stats = build_responses(selected, args, args.dense, args.output)
        if stats['new_cases'] or not path.exists():
            write_json(path, stats)
    return path
