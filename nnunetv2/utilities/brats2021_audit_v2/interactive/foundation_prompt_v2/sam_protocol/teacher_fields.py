"""Build signed WT distances and train-only corrective-point targets."""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..common import file_identity, read_json, write_json
from . import residual
from .cohorts import rows

RADIUS_MM = 36.


def fields(baseline, spacing, truth=None):
    """Keep a physical prior and the depth of genuine errors on the native grid."""
    original = baseline > 0
    distance = np.full(original.shape, -RADIUS_MM, np.float16)
    if original.any():
        padding = int(np.ceil(RADIUS_MM / min(spacing))) + 1
        bounds = residual._window(original, padding=padding)
        patch = np.pad(original[bounds], 1)
        signed = ndimage.distance_transform_edt(patch, sampling=spacing)
        signed -= ndimage.distance_transform_edt(~patch, sampling=spacing)
        distance[bounds] = signed[1:-1, 1:-1, 1:-1].clip(-RADIUS_MM, RADIUS_MM)
    if truth is None:
        return distance, None
    error = np.zeros(original.shape, np.float16)
    target = truth > 0
    # Restrict WT deletions to the edema label to preserve the original core.
    for difference in (target & ~original, (baseline == 1) & ~target):
        if not difference.any():
            continue
        bounds = residual._window(difference, padding=1)
        patch = np.pad(difference[bounds], 1)
        depth = ndimage.distance_transform_edt(patch, sampling=spacing)[1:-1, 1:-1, 1:-1]
        error[bounds] = np.maximum(error[bounds], depth.clip(0., 12.))
    return distance, error


def _write(path, array):
    temp = path.with_suffix('.part.npy')
    np.save(temp, array, allow_pickle=False)
    temp.replace(path)


def run(args):
    labeled = args.split == 'train'
    cases = rows(args.cohort, args.split, truth=labeled)[args.shard_index::args.num_shards]
    source_code = file_identity(__file__)
    for row in cases:
        source = Path(args.cache) / args.split / row['case_id']
        receipt = read_json(source / 'volume.json')
        if receipt['binding']['row'] != row or receipt['labeled'] != labeled:
            raise ValueError('WT target preparation requires the frozen MRI cache')
        folder = Path(args.output) / args.split / row['case_id']
        path = folder / 'fields.json'
        binding = {'row': row, 'volume': file_identity(source / 'volume.json'), 'code': source_code}
        if path.is_file():
            if read_json(path)['binding'] != binding:
                raise ValueError('The cached signed distances have different inputs')
            continue
        old = np.load(source / 'baseline.npy', mmap_mode='r', allow_pickle=False)
        gt = np.load(source / 'truth.npy', mmap_mode='r', allow_pickle=False) if labeled else None
        spacing = np.linalg.norm(np.asarray(receipt['affine'])[:3, :3], axis=0)
        distance, error = fields(old, spacing, gt)
        folder.mkdir(parents=True, exist_ok=True)
        _write(folder / 'distance.npy', distance)
        if error is not None:
            _write(folder / 'error.npy', error)
        write_json(path, {'binding': binding, 'labeled': labeled})
        error_count = int(np.count_nonzero(error)) if labeled else None
        print(f'{row["case_id"]} labeled={labeled} teacher voxels={error_count}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cohort', 'cache', 'output'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    run(p.parse_args())
