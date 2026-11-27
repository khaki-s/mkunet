"""Whole-slice, GT-independent inputs for a domain-adapted SAM decoder.

Only the image encoder is cached. Images retain the complete native slice and
the official encoder performs its own documented input transform. The cache
never contains GT; targets are loaded separately when a training caller asks.
Empty baseline slices are deliberately preserved, so this first protocol does
not claim to discover lesions absent from the initial segmentation.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (check_file, file_identity, load_case, manifest, read_json,
                     read_labels, same_grid, write_json)

HEADS = ('WT', 'TC', 'ET')
DEFAULT_MODALITIES = {'WT': 3, 'TC': 1, 'ET': 1}
FEATURE_KEYS = ('image_embed', 'high_res_0', 'high_res_1')


def validate_heads(heads):
    heads = tuple(heads.split(',')) if isinstance(heads, str) else tuple(heads)
    if not heads or len(set(heads)) != len(heads) or any(h not in HEADS for h in heads):
        raise ValueError('heads must be distinct entries from WT,TC,ET')
    return heads


def modality_ranges(mri):
    """Volume-level MRI scaling, independent of predicted or reference labels."""
    images = np.asarray(mri, dtype=np.float32)
    if images.ndim != 4 or images.shape[0] != 4 or not np.isfinite(images).all():
        raise ValueError('Expected finite MRI[4,X,Y,Z]')
    result = []
    for image in images:
        nonzero = image[image != 0]
        lo, hi = np.percentile(nonzero, (1, 99)) if nonzero.size else (0., 1.)
        result.append((float(lo), max(float(hi - lo), 1e-6)))
    return result


def slice_rgb(mri, modality, z, ranges=None):
    """Return the complete native H,W slice as replicated grayscale RGB uint8."""
    if ranges is None:
        ranges = modality_ranges(mri)
    lo, span = ranges[modality]
    gray = (np.clip((mri[modality, :, :, z] - lo) / span, 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def slice_records(case_id, baseline, heads=('WT',), head_modalities=None):
    """Schedule every nonempty initial-mask slice, never select slices by GT."""
    heads = validate_heads(heads)
    modalities = dict(DEFAULT_MODALITIES if head_modalities is None else head_modalities)
    if any(h not in modalities or type(modalities[h]) is not int or modalities[h] not in range(4)
           for h in heads):
        raise ValueError('Each requested head must map to one of the four MRI channels')
    baseline = np.asarray(baseline)
    if baseline.ndim != 3 or not np.isin(baseline, (0, 1, 2, 3)).all():
        raise ValueError('Expected native internal labels BG/ED/NCR/ET=0/1/2/3')
    result = []
    for head in heads:
        h = HEADS.index(head)
        for z in np.flatnonzero((baseline > h).any(axis=(0, 1))):
            modality = modalities[head]
            result.append({'case_id': str(case_id), 'head': head, 'head_index': h,
                           'z': int(z), 'modality': modality,
                           'native_shape': list(baseline.shape[:2]),
                           'feature_key': f'{case_id}__m{modality}__z{int(z):03d}'})
    return result


def save_features(path, features):
    """Atomic, pickle-free float16 cache of all three original feature scales."""
    if set(features) != set(FEATURE_KEYS):
        raise ValueError(f'Encoder must return exactly {FEATURE_KEYS}')
    arrays = {}
    for key in FEATURE_KEYS:
        value = features[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device='cpu', dtype=torch.float16).numpy()
        value = np.asarray(value, dtype=np.float16)
        if value.ndim != 4 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError(f'Invalid finite batch-one encoder feature {key}: {value.shape}')
        arrays[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        # No spatial downsampling, quantization, or feature pruning. Float16 is
        # the same representation used by frozen GPU autocast, not a resize.
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return file_identity(path)


def read_features(identity, *, verify=True):
    if verify:
        check_file(identity)
    with np.load(identity['path'], allow_pickle=False) as archive:
        if set(archive.files) != set(FEATURE_KEYS):
            raise ValueError('Unexpected data or targets in the image-feature archive')
        result = {k: torch.from_numpy(np.array(archive[k], copy=True)) for k in FEATURE_KEYS}
    if any(x.ndim != 4 or x.shape[0] != 1 or not torch.isfinite(x).all() for x in result.values()):
        raise ValueError('Invalid cached SAM features')
    return result


def cache_cases(manifest_path, output, encoder: Callable, *, contract,
                heads=('WT',), limit_cases=0, num_shards=1, shard_index=0,
                head_modalities=None):
    """Cache frozen encoder output; ``encoder(rgb)`` never receives masks or GT.

    ``manifest_path`` retains the entire original split even for a diagnostic
    ``limit_cases`` subset. Non-training sources must be deployment manifests.
    Multiple workers use separate output folders and disjoint shard indices.
    Resuming checks source/configuration and file hashes, including partial
    cases. There must be only one writer per output folder.
    """
    source = manifest(manifest_path)
    if source['split'] != 'train':
        manifest(manifest_path, deployment=True)
    heads = validate_heads(heads)
    modalities = dict(DEFAULT_MODALITIES if head_modalities is None else head_modalities)
    if type(limit_cases) is not int or limit_cases < 0:
        raise ValueError('limit_cases must be a nonnegative integer')
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid case shard')
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    path = root / 'cache.json'
    binding = {'schema': 'sam_domain_feature_cache_v1', 'source': file_identity(manifest_path),
               'contract': contract, 'heads': list(heads), 'selected_heads': list(heads),
               'head_modalities': modalities,
               'limit_cases': limit_cases, 'num_shards': num_shards, 'shard_index': shard_index,
               'gt_in_features_or_prompts': False, 'spatial_crop': False,
               'native_slice_resize': False, 'encoder_input_resolution': 1008}
    if path.exists():
        index = read_json(path)
        if index['binding'] != binding:
            raise ValueError('Cache source/protocol differs; use a new cache folder')
        for identity in index['features'].values():
            check_file(identity)
    else:
        index = {'binding': binding, 'features': {}, 'completed': {}, 'samples': [],
                 'full_cohort': False}
        write_json(path, index)
    selected = source['cases'][:limit_cases] if limit_cases else source['cases']
    rows = selected[shard_index::num_shards]
    valid_cases = {r['case_id'] for r in rows}
    if not set(index['completed']).issubset(valid_cases):
        raise ValueError('Cache contains foreign patients')
    for row in rows:
        key = row['case_id']
        if key in index['completed']:
            continue
        clean = {k: v for k, v in row.items() if k != 'gt'}
        mri, baseline, _reference = load_case(clean, with_gt=False)
        if row.get('availability', [1] * 4) != [1] * 4:
            raise ValueError('This domain-adaptation experiment requires four available MRIs')
        records = slice_records(key, baseline, heads, modalities)
        ranges = modality_ranges(mri)
        for record in records:
            feature_key = record['feature_key']
            if feature_key in index['features']:
                continue
            rgb = slice_rgb(mri, record['modality'], record['z'], ranges)
            features = encoder(rgb)
            identity = save_features(root / key / f'{feature_key}.npz', features)
            index['features'][feature_key] = identity
            write_json(path, index)
        index['completed'][key] = {'input': clean, 'native_volume_shape': list(baseline.shape),
                                   'samples': records, 'normalization_ranges': ranges}
        index['samples'] = [s for c in index['completed'].values() for s in c['samples']]
        index['full_cohort'] = len(index['completed']) == len(source['cases'])
        write_json(path, index)
        print(f'SAM domain cache {key}: {len(records)} complete native slices; '
              f'cases {len(index["completed"])}/{len(rows)}', flush=True)
    return path


def merge_caches(paths, output):
    """Join complete disjoint case shards without copying large feature files."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new merged cache index path')
    merged = None; shared = None; shards = set()
    for path in paths:
        value = read_json(path)
        binding = dict(value['binding']); shard = binding.pop('shard_index')
        if shared is None:
            shared = binding
            merged = {'features': {}, 'completed': {}, 'samples': []}
        if binding != shared or shard in shards:
            raise ValueError('Mismatched or duplicate feature-cache shards')
        check_file(binding['source']); source = manifest(binding['source']['path'])
        rows = source['cases'][:binding['limit_cases']] if binding['limit_cases'] else source['cases']
        expected = {r['case_id'] for r in rows[shard::binding['num_shards']]}
        if set(value['completed']) != expected:
            raise ValueError('Incomplete or foreign cases in feature-cache shard')
        if merged['completed'].keys() & value['completed'].keys():
            raise ValueError('Overlapping feature-cache patients')
        for identity in value['features'].values():
            check_file(identity)
        merged['features'].update(value['features'])
        merged['completed'].update(value['completed'])
        merged['samples'].extend(value['samples'])
        shards.add(shard)
    if shared is None or shards != set(range(shared['num_shards'])):
        raise ValueError('Every declared cache shard is required')
    merged['binding'] = {**shared, 'shard_index': None,
                         'merged_from': [file_identity(p) for p in paths]}
    merged['full_cohort'] = len(merged['completed']) == len(source['cases'])
    write_json(output, merged)
    return output


class CachedSliceDataset(Dataset):
    """Batch-one full native slices; labels are separate from cached features.

    Training must set ``allowed_splits=('train',)``. Validation losses require
    explicit ``truth_manifest`` when cache inputs use a deployment manifest.
    Test never permits ``with_gt=True``. Each worker keeps at most sixteen native
    case-label volumes; a giant MRI volume is not sent through a DataLoader.
    """
    def __init__(self, cache_index, *, with_gt=False, allowed_splits=None,
                 heads=None, truth_manifest=None, verify_features=True,
                 case_ids=None, case_cache_size=16):
        self.index = read_json(cache_index)
        binding = self.index['binding']
        self.binding = binding
        if binding.get('schema') != 'sam_domain_feature_cache_v1':
            raise ValueError('Expected SAM domain feature cache')
        check_file(binding['source'])
        self.source = manifest(binding['source']['path'], allowed=allowed_splits)
        source_rows = {r['case_id']: r for r in self.source['cases']}
        for key, case in self.index['completed'].items():
            if key not in source_rows or case['input'] != {
                    k: v for k, v in source_rows[key].items() if k != 'gt'}:
                raise ValueError('Cached patient inputs differ from the bound source manifest')
        expected_samples = [s for case in self.index['completed'].values() for s in case['samples']]
        if self.index['samples'] != expected_samples:
            raise ValueError('Cached sample index differs from completed patient records')
        self.with_gt = bool(with_gt)
        if self.source['split'] != 'train':
            manifest(binding['source']['path'], deployment=True)
        if self.with_gt and self.source['split'] == 'test':
            raise ValueError('Test features cannot expose GT to model inference')
        if truth_manifest and not self.with_gt:
            raise ValueError('A GT source is not accepted in an unlabeled dataset')
        if truth_manifest:
            truth = manifest(truth_manifest, allowed=(self.source['split'],))
            self.truth_rows = {r['case_id']: r for r in truth['cases']}
            for clean in self.source['cases']:
                row = self.truth_rows[clean['case_id']]
                if {k: v for k, v in row.items() if k != 'gt'} != {k: v for k, v in clean.items() if k != 'gt'}:
                    raise ValueError('Target manifest images or initial masks differ from cache inputs')
        else:
            self.truth_rows = {r['case_id']: r for r in self.source['cases']}
        chosen = set(validate_heads(heads)) if heads is not None else set(binding['heads'])
        self.heads = [h for h in binding['heads'] if h in chosen]
        chosen_cases = None if case_ids is None else set(case_ids)
        self.case_ids = [key for key in self.index['completed']
                         if chosen_cases is None or key in chosen_cases]
        self.samples = [s for s in self.index['samples'] if s['head'] in chosen
                        and (chosen_cases is None or s['case_id'] in chosen_cases)]
        if not self.samples:
            raise ValueError('No nonempty baseline slices in selected cached cases')
        self.records = self.samples
        self.case_cache_size = int(case_cache_size)
        if self.case_cache_size < 1:
            raise ValueError('case_cache_size must be positive')
        self.verify_features = bool(verify_features)
        self._checked_features = set()
        self._cases = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, key):
        if key in self._cases:
            self._cases.move_to_end(key)
            return self._cases[key]
        row = self.index['completed'][key]['input']
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        baseline = read_labels(reference)
        truth = None
        if self.with_gt:
            gt = self.truth_rows[key].get('gt')
            if gt is None:
                raise ValueError('Explicit GT manifest is required to request supervised targets')
            check_file(gt); image = nib.load(gt['path']); same_grid(image, reference)
            truth = read_labels(image)
        result = baseline, truth
        self._cases[key] = result
        while len(self._cases) > self.case_cache_size:
            self._cases.popitem(last=False)
        return result

    def __getitem__(self, index):
        record = self.samples[index]
        identity = self.index['features'][record['feature_key']]
        features = read_features(identity, verify=self.verify_features and
                                 record['feature_key'] not in self._checked_features)
        self._checked_features.add(record['feature_key'])
        baseline, truth = self._load_labels(record['case_id'])
        h, z = record['head_index'], record['z']
        binary = (baseline[:, :, z] > h).copy()
        if tuple(binary.shape) != tuple(record['native_shape']) or not binary.any():
            raise ValueError('Cached slice no longer matches its native baseline')
        item = {**record, 'features': features, 'baseline': torch.from_numpy(binary)}
        if self.with_gt:
            item['target'] = torch.from_numpy((truth[:, :, z] > h).astype(np.float32))
        return item


def single_slice_collate(items):
    if len(items) != 1:
        raise ValueError('Use per-GPU batch_size=1; accumulate gradients for larger effective batches')
    return items[0]
