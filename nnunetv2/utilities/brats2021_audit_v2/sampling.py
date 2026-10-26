"""Train-fold-only exclusive coordinates; reuse nnU-Net crop/augment code.

Run once: python -m nnunetv2.utilities.brats2021_audit.sampling --prepare
No label coordinates are inferred by subtracting sparsely sampled region lists.
"""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


DEFAULT_ROOT = '/root/autodl-tmp/data/nnUNet_preprocessed/Dataset137_BraTS2021'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_stats(folder, key):
    # Include all on-disk case representations: changing segmentation or its
    # properties invalidates the prepared coordinates. Not a content hash/QC.
    return {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(Path(folder).glob(key + '*'))
            if p.is_file() and (p.name.startswith(key + '.') or p.name.startswith(key + '_seg.'))}


def prepare_index(root=DEFAULT_ROOT, fold=0):
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    root = Path(root)
    splits = json.loads((root / 'splits_final.json').read_text())
    split = splits[fold]
    train, val = split['train'], split['val']
    if len(set(train)) != len(train) or set(train) & set(val):
        raise ValueError('Duplicate training IDs or overlapping split')
    plans = json.loads((root / 'nnUNetPlans.json').read_text())
    folder = root / plans['configurations']['3d_fullres']['data_identifier']
    output = root / f'brats2021_audit_sampling_fold{fold}'
    if (output / 'manifest.json').exists():
        validate_index(output, folder, train)
        print(f'Existing index verified: {output}', flush=True)
        return output
    output.mkdir(exist_ok=True)
    dataset = infer_dataset_class(str(folder))(str(folder), train)
    rng = np.random.RandomState(20260913)
    cases = {}
    for i, key in enumerate(train):
        _, seg, _, _ = dataset.load_case(key)
        seg = np.asarray(seg)[0]
        # -1 is the standard nnU-Net outside-nonzero-mask marker, never a foreground center.
        if not np.isin(seg, [-1, 0, 1, 2, 3]).all():
            raise ValueError(f'Unexpected internal labels: {key}')
        coords, counts = {}, {}
        for label in (1, 2, 3):
            flat = np.flatnonzero(seg == label)
            counts[str(label)] = int(flat.size)
            if len(flat) > 10000:
                flat = rng.choice(flat, 10000, replace=False)
            spatial = np.stack(np.unravel_index(flat, seg.shape), axis=1)
            coords[str(label)] = np.column_stack((np.zeros(len(flat), dtype=np.int32), spatial)).astype(np.int32)
        np.savez_compressed(output / f'{key}.npz', **coords)
        cases[key] = {'counts': counts, 'source_stats': source_stats(folder, key),
                      'coordinates_sha256': digest(output / f'{key}.npz')}
        if (i + 1) % 20 == 0:
            print(f'Indexed training cases {i + 1}/{len(train)}', flush=True)
    manifest = {'schema': 1, 'fold': fold, 'train': train, 'val': val, 'cases': cases,
                'hashes': {name: digest(root / name) for name in
                           ('splits_final.json', 'nnUNetPlans.json', 'dataset.json')}}
    temp = output / 'manifest.tmp'
    temp.write_text(json.dumps(manifest, indent=2) + '\n')
    temp.replace(output / 'manifest.json')
    print(f'Index complete: {output}; empty NCR: {sum(c["counts"]["2"] == 0 for c in cases.values())}', flush=True)
    return output


def validate_index(index, folder, train_ids):
    index, folder = Path(index), Path(folder)
    if not (index / 'manifest.json').exists():
        raise RuntimeError('Prepare training coordinates first: python -m nnunetv2.utilities.brats2021_audit.sampling --prepare')
    manifest = json.loads((index / 'manifest.json').read_text())
    if set(manifest['train']) != set(train_ids) or set(train_ids) & set(manifest['val']):
        raise ValueError('Sampling index does not match this training fold')
    for name, sha in manifest['hashes'].items():
        if digest(folder.parent / name) != sha:
            raise ValueError(f'Stale sampling index: {name} changed')
    for key in train_ids:
        entry = manifest['cases'][key]
        if source_stats(folder, key) != entry['source_stats']:
            raise ValueError(f'Stale sampling index: case files changed for {key}')
        if digest(index / f'{key}.npz') != entry['coordinates_sha256']:
            raise ValueError(f'Corrupt coordinate file: {key}')
    return manifest


class ExclusiveClassDataLoader(nnUNetDataLoader):
    """Mixture: 50% original loader / 25% NCR / 12.5% ED / 12.5% ET.

    Choose a case uniformly from the requested class's nonempty TRAIN cohort,
    then a voxel uniformly from its reproducibly subsampled coordinate list.
    Empty NCR cases remain eligible in the original 50% branch. The forced
    center can disappear under augmentation; ratios describe pre-DA proposals.
    """
    def __init__(self, *args, index_folder, manifest, **kwargs):
        super().__init__(*args, **kwargs)
        if set(self.indices) != set(manifest['train']):
            raise ValueError('Loader IDs must exactly match indexed training IDs')
        self.index_folder = Path(index_folder)
        self.pools = {label: [key for key in self.indices if manifest['cases'][key]['counts'][str(label)] > 0]
                      for label in (1, 2, 3)}
        if any(not pool for pool in self.pools.values()):
            raise ValueError('At least one training case per exclusive class is required')

    def get_indices(self):
        keys = list(super().get_indices())
        self.draw_labels = np.random.choice([0, 2, 1, 3], self.batch_size, p=[.5, .25, .125, .125])
        for j, label in enumerate(self.draw_labels):
            if label:
                keys[j] = np.random.choice(self.pools[int(label)])
        self.draw_keys = keys
        return keys

    def _oversample_last_XX_percent(self, sample_idx):
        self.sample_index = sample_idx
        return bool(self.draw_labels[sample_idx]) or super()._oversample_last_XX_percent(sample_idx)

    def _probabilistic_oversampling(self, sample_idx):
        self.sample_index = sample_idx
        return bool(self.draw_labels[sample_idx]) or super()._probabilistic_oversampling(sample_idx)

    @lru_cache(maxsize=32)
    def _coordinates(self, key):
        with np.load(self.index_folder / f'{key}.npz') as arrays:
            return {int(label): arrays[label] for label in arrays.files}

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        label = int(self.draw_labels[self.sample_index])
        if label:
            locations = self._coordinates(self.draw_keys[self.sample_index])
            return super().get_bbox(data_shape, True, {label: locations[label]}, label, verbose)
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    prepare_index(args.root, args.fold)
