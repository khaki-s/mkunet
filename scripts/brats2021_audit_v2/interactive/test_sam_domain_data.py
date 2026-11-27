"""GT-boundary and native-geometry tests, not segmentation performance claims."""
import importlib
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch
import foundation_prompt

d = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_domain_data')


class DomainDataTests(unittest.TestCase):
    def fixture(self, root):
        splits = root / 'splits.json'
        d.write_json(splits, [{'train': ['a', 'b'], 'val': ['v']}])
        white = root / 'test.json'; d.write_json(white, {'cases': [{'case_id': 't'}]})
        shape = (13, 17, 5)
        paths = {}
        for split, ids in [('train', ['a', 'b']), ('validation', ['v']), ('test', ['t'])]:
            rows = []
            for key in ids:
                mask = np.zeros(shape, np.uint8); mask[3:9, 4:12, 1:4] = 3
                pred = root / f'{key}_prediction.nii.gz'
                nib.save(nib.Nifti1Image(mask, np.eye(4)), pred)
                truth = mask.copy(); truth[9, 4:12, 1:4] = 1
                gt = root / f'{key}_gt.nii.gz'; nib.save(nib.Nifti1Image(truth, np.eye(4)), gt)
                modalities = {}
                for mod in ('T1', 'T1ce', 'T2', 'FLAIR'):
                    path = root / f'{key}_{mod}.nii.gz'
                    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
                    nib.save(nib.Nifti1Image(image, np.eye(4)), path)
                    modalities[mod] = d.file_identity(path)
                rows.append({'case_id': key, 'modalities': modalities,
                             'prediction': d.file_identity(pred), 'gt': d.file_identity(gt),
                             'availability': [1, 1, 1, 1]})
            value = {'schema': 'foundation_cases_v2', 'split': split, 'fold': 0,
                     'splits_file': d.file_identity(splits), 'contains_gt': True,
                     'test_whitelist': d.file_identity(white), 'cases': rows}
            source = root / f'{split}_source.json'; d.write_json(source, value)
            deploy = root / f'{split}_deployment.json'
            d.write_json(deploy, {**value, 'contains_gt': False,
                                 'cases': [{k: v for k, v in r.items() if k != 'gt'} for r in rows]})
            paths[split] = (source, deploy)
        return paths

    def encoder(self, received):
        def encode(rgb):
            received.append(rgb)
            self.assertEqual(rgb.shape, (13, 17, 3))
            self.assertEqual(rgb.dtype, np.uint8)
            return {key: torch.full((1, i + 1, 4, 4), float(rgb.mean()))
                    for i, key in enumerate(d.FEATURE_KEYS)}
        return encode

    def test_cache_reuse_native_geometry_targets_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.fixture(root); seen = []
            path = d.cache_cases(paths['train'][0], root / 'cache', self.encoder(seen),
                                 contract={'fake': True}, heads=('WT', 'TC', 'ET'))
            # Per patient WT has three FLAIR images; TC/ET reuse three T1ce images.
            self.assertEqual(len(seen), 12)
            value = d.read_json(path)
            self.assertEqual(len(value['samples']), 18)
            self.assertTrue(value['full_cohort'])
            self.assertTrue(all('gt' not in c['input'] for c in value['completed'].values()))
            dataset = d.CachedSliceDataset(path, with_gt=True, allowed_splits=('train',))
            sample = dataset[0]
            self.assertEqual(tuple(sample['baseline'].shape), (13, 17))
            self.assertEqual(tuple(sample['target'].shape), (13, 17))
            self.assertGreater(sample['target'].sum(), sample['baseline'].sum())
            self.assertEqual(dataset.records, dataset.samples)
            self.assertEqual(d.single_slice_collate([sample]), sample)
            self.assertNotIn('target', d.CachedSliceDataset(path)[0])
            d.cache_cases(paths['train'][0], root / 'cache', self.encoder(seen),
                          contract={'fake': True}, heads=('WT', 'TC', 'ET'))
            self.assertEqual(len(seen), 12)
            with self.assertRaisesRegex(ValueError, 'differs'):
                d.cache_cases(paths['train'][0], root / 'cache', self.encoder(seen),
                              contract={'fake': False}, heads=('WT', 'TC', 'ET'))
            identity = next(iter(value['features'].values()))
            with Path(identity['path']).open('ab') as f: f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'changed'):
                d.CachedSliceDataset(path)[0]

    def test_validation_test_gt_separation_and_train_only_optimization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.fixture(root); seen = []
            with self.assertRaisesRegex(ValueError, 'WITHOUT GT'):
                d.cache_cases(paths['validation'][0], root / 'bad', self.encoder(seen), contract={})
            val = d.cache_cases(paths['validation'][1], root / 'val', self.encoder(seen), contract={})
            test = d.cache_cases(paths['test'][1], root / 'test_cache', self.encoder(seen), contract={})
            with self.assertRaisesRegex(ValueError, 'not allowed'):
                d.CachedSliceDataset(val, with_gt=True, allowed_splits=('train',))
            with self.assertRaisesRegex(ValueError, 'cannot expose GT'):
                d.CachedSliceDataset(test, with_gt=True)
            self.assertNotIn('target', d.CachedSliceDataset(test)[0])
            with self.assertRaisesRegex(ValueError, 'Explicit GT manifest'):
                d.CachedSliceDataset(val, with_gt=True)[0]
            labeled = d.CachedSliceDataset(val, with_gt=True, truth_manifest=paths['validation'][0])
            self.assertIn('target', labeled[0])

    def test_shards_and_limit_preserve_complete_source_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths = self.fixture(root); seen = []
            shards = [d.cache_cases(paths['train'][0], root / f'shard{s}', self.encoder(seen),
                                    contract={}, num_shards=2, shard_index=s) for s in range(2)]
            merged = d.merge_caches(shards, root / 'merged.json')
            self.assertTrue(d.read_json(merged)['full_cohort'])
            self.assertEqual(len(d.CachedSliceDataset(merged)), 6)
            with self.assertRaisesRegex(ValueError, 'Every declared'):
                d.merge_caches(shards[:1], root / 'bad_merge.json')
            pilot = d.cache_cases(paths['train'][0], root / 'pilot', self.encoder(seen),
                                  contract={}, limit_cases=1)
            dataset = d.CachedSliceDataset(pilot)
            self.assertFalse(dataset.index['full_cohort'])
            self.assertEqual(len(dataset.source['cases']), 2)
            self.assertEqual(len(dataset), 3)


if __name__ == '__main__':
    unittest.main()
