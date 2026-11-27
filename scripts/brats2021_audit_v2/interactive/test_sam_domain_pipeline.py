"""Independent CPU checks for deployment semantics; not Dice performance evidence."""
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import copy

import nibabel as nib
import numpy as np
import torch

import foundation_prompt

p = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_domain_pipeline')
d = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_domain_data')
m = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_domain_model')


class SyntheticDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.))
        self.calls = 0

    def forward(self, features, baseline):
        self.calls += 1
        return features['logits'] * self.anchor


class SyntheticInferenceDataset:
    """Precomputed synthetic image evidence, independent from label files."""
    def __init__(self, source, baseline, proposals):
        self.source = source
        self.case_ids = list(baseline)
        self.heads = ('WT',)
        self.with_gt = False
        self.records = [{'case_id': key, 'head': 'WT', 'z': z}
                        for key, volume in baseline.items()
                        for z in np.flatnonzero((volume > 0).any(axis=(0, 1)))]
        self.baseline = baseline
        self.proposals = proposals

    def __getitem__(self, index):
        record = self.records[index]
        key, z = record['case_id'], record['z']
        proposal = self.proposals[key][:, :, z]
        logits = torch.from_numpy(np.where(proposal, 12., -12.).astype(np.float32))[None, None]
        return {**record, 'features': {'logits': logits},
                'baseline': self.baseline[key][:, :, z] > 0}


class DomainPipelineTests(unittest.TestCase):
    def baseline(self):
        mask = np.zeros((9, 11, 3), dtype=np.uint8)
        mask[2:7, 3:9, 1] = 1
        mask[3:6, 4:8, 1] = 2
        mask[4, 5, 1] = 3
        return mask

    def fixture(self, root):
        splits = root / 'splits.json'
        p.write_json(splits, [{'train': ['train_case'], 'val': ['val_a', 'val_b']}])
        rows, baseline, proposed = [], {}, {}
        for case_id in ('val_a', 'val_b'):
            base = self.baseline()
            final = base > 0
            final[1:8, 2:10, 1] = True
            truth = base.copy()
            if case_id == 'val_a':
                truth[(final) & (truth == 0)] = 1
            folder = root / case_id
            folder.mkdir()
            b, g = folder/'initial.nii.gz', folder/'truth.nii.gz'
            nib.save(nib.Nifti1Image(base, np.eye(4)), b)
            nib.save(nib.Nifti1Image(truth, np.eye(4)), g)
            modalities = {}
            for m in ('T1', 'T1ce', 'T2', 'FLAIR'):
                path = folder / f'{m}.nii.gz'
                nib.save(nib.Nifti1Image(np.zeros_like(base, dtype=np.float32), np.eye(4)), path)
                modalities[m] = p.file_identity(path)
            rows.append({'case_id': case_id, 'prediction': p.file_identity(b),
                         'modalities': modalities, 'availability': [1]*4,
                         'gt': p.file_identity(g)})
            baseline[case_id], proposed[case_id] = base, final
        truth_source = {'schema': 'foundation_cases_v2', 'split': 'validation',
                        'fold': 0, 'splits_file': p.file_identity(splits), 'cases': rows}
        truth_path = root/'val.json'
        p.write_json(truth_path, truth_source)
        clean = {**truth_source, 'contains_gt': False,
                 'cases': [{k: v for k, v in row.items() if k != 'gt'} for row in rows]}
        return truth_path, SyntheticInferenceDataset(clean, baseline, proposed)

    def test_zero_strength_is_exact_original_and_no_mutation(self):
        base = self.baseline()
        saved = base.copy()
        result = p.compose(base, {'WT': np.ones_like(base, dtype=np.float32)}, 0.)
        np.testing.assert_array_equal(result, saved)
        result[0, 0, 0] = 3
        np.testing.assert_array_equal(base, saved)

    def test_wt_only_preserves_tc_et_for_all_fixed_strengths(self):
        base = self.baseline()
        for foreground in (0., 1.):
            for alpha in (0., .25, .5, .75, 1.):
                result = p.compose(base, {'WT': np.full(base.shape, foreground)}, alpha)
                np.testing.assert_array_equal(result >= 2, base >= 2)
                np.testing.assert_array_equal(result == 3, base == 3)
                self.assertTrue(np.all((result >= 2) <= (result > 0)))

    def test_invalid_prediction_probabilities_fail_closed(self):
        base = self.baseline()
        for invalid in (-.1, 1.1, np.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                p.compose(base, {'WT': np.full(base.shape, invalid)}, 1.)

    def test_absent_region_false_positive_is_penalized_not_omitted(self):
        empty = np.zeros((4, 5, 6), dtype=np.uint8)
        added = empty.copy(); added[1, 2, 3] = 3
        rows = p.metric_rows('case_a', empty, added, empty)
        rows += p.metric_rows('case_b', empty, empty, empty)
        report = p.summarize(rows)
        for head in ('WT', 'TC', 'ET'):
            self.assertEqual(report[head]['n'], 2)
            self.assertEqual(report[head]['mean_gain_pp'], -50.)
            self.assertEqual(report[head]['empty_gt_added_fp'], 1)
            self.assertEqual(report[head]['worsened_fraction'], .5)

    def test_inference_reads_no_gt_and_covers_complete_native_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            truth_path, dataset = self.fixture(Path(directory))
            truth_paths = {r['gt']['path'] for r in p.read_json(truth_path)['cases']}
            original_load = nib.load
            def forbid_gt(path, *args, **kwargs):
                self.assertNotIn(str(path), truth_paths)
                return original_load(path, *args, **kwargs)
            decoder = SyntheticDecoder()
            with patch.object(p.nib, 'load', side_effect=forbid_gt):
                outputs = list(p.infer_cases(decoder, dataset, 'cpu'))
            self.assertEqual(len(outputs), 2)
            self.assertEqual(decoder.calls, 2)
            self.assertFalse(decoder.training)
            for key, baseline, _, probabilities, _ in outputs:
                self.assertEqual(probabilities['WT'].shape, baseline.shape)
                self.assertFalse(probabilities['WT'][:, :, 0].any())
                self.assertFalse(probabilities['WT'][:, :, 2].any())
            dataset.with_gt = True
            with self.assertRaisesRegex(ValueError, 'with_gt=False'):
                list(p.infer_cases(decoder, dataset, 'cpu'))

    def test_validation_selects_one_global_alpha_and_not_patient_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            gt_path, dataset = self.fixture(Path(directory))
            decoder = SyntheticDecoder()
            result = p.validate(decoder, dataset, gt_path, 'cpu', (0., .5, 1.))
            self.assertFalse(result['full_val80'])
            self.assertEqual(result['case_count'], 2)
            expected = []
            gt_rows = {r['case_id']: r for r in p.read_json(gt_path)['cases']}
            for key, baseline, _, probabilities, _ in p.infer_cases(decoder, dataset, 'cpu'):
                truth = p.read_labels(nib.load(gt_rows[key]['gt']['path']))
                final = p.compose(baseline, probabilities, 1.)
                expected.extend(p.metric_rows(key, baseline, final, truth))
            candidate = next(r for r in result['candidates'] if r['alpha'] == 1.)
            self.assertAlmostEqual(candidate['regions']['WT']['mean_gain_pp'],
                                   p.summarize(expected)['WT']['mean_gain_pp'])
            wt_changes = [r['gain_pp'] for r in expected if r['region'] == 'WT']
            self.assertLess(candidate['regions']['WT']['mean_gain_pp'],
                            np.mean(np.maximum(wt_changes, 0.)))
            self.assertNotIn('per_case_alpha', result)

    def test_validation_rejects_mri_input_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            gt_path, dataset = self.fixture(Path(directory))
            dataset.source['cases'][0]['modalities'] = dict(dataset.source['cases'][0]['modalities'])
            dataset.source['cases'][0]['modalities']['FLAIR'] = {'path': '/different_mri', 'sha256': 'wrong'}
            with self.assertRaises(ValueError):
                p.validate(SyntheticDecoder(), dataset, gt_path, 'cpu', (0., 1.))

    def test_real_train_loop_accumulation_matches_reference_batches_including_tail(self):
        class ScalarDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.25))
            def configuration(self):
                return {'synthetic': True}
            def forward(self, features, baseline):
                # DDP moves Tensor args to CUDA before forward. Prompt geometry
                # is CPU/scipy: the training boundary must supply ndarray so
                # np.asarray in the real decoder never receives a CUDA Tensor.
                if not isinstance(baseline, np.ndarray):
                    raise TypeError('CPU prompt geometry must cross the DDP boundary as ndarray')
                return self.weight * features['signal']
        class ScalarDataset:
            def __init__(self, key):
                self.case_ids = [key]; self.heads = ('WT',)
                self.index = {'binding': {'contract': {'synthetic': True}}}
                self.source = {'initial_checkpoint': {'synthetic': True}}
            def __len__(self):
                return 6
            def __getitem__(self, i):
                return {'features': {'signal': torch.full((1, 1, 2, 2), (i+1)*.05)},
                        'baseline': np.ones((2, 2), bool),
                        'target': torch.full((1, 1, 2, 2), .1 + .03*i)}
        loss = lambda logits, target: torch.nn.functional.mse_loss(logits, target)
        train_data, val_data = ScalarDataset('train_case'), ScalarDataset('val_case')
        fixed_validation = {'selected': {'alpha': .5, 'regions': {
            head: {'mean_gain_pp': .1 if head == 'WT' else 0.} for head in ('WT', 'TC', 'ET')}},
            'case_count': 1, 'full_val80': False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train_cache.json', 'val_cache.json', 'val_manifest.json', 'sam.pth'):
                (root/name).write_text('{}')
            args = SimpleNamespace(train_cache=root/'train_cache.json', val_cache=root/'val_cache.json',
                val_manifest=root/'val_manifest.json', sam_checkpoint=root/'sam.pth', source=str(root),
                output=root/'training', epochs=1, accumulate=4, lr=.01, seed=53, mode='decoder',
                lora_rank=4, workers=0, cpu_threads=1)
            contract = {'sam_checkpoint': p.file_identity(args.sam_checkpoint),
                'sam_source': str(root.resolve()), 'encoder_resolution': 1008,
                'spatial_crop': False, 'sam_source_sha256': {'synthetic': True},
                'implementation': {k: v for k, v in p.implementation().items()
                                   if k != 'sam_domain_pipeline.py'}}
            train_data.index['binding']['contract'] = copy.deepcopy(contract)
            val_data.index['binding']['contract'] = copy.deepcopy(contract)
            with patch.object(d, 'CachedSliceDataset', side_effect=[train_data, val_data]), \
                 patch.object(m, 'build_domain_decoder', side_effect=lambda *a, **kw: ScalarDecoder()), \
                 patch.object(m, 'source_identity', return_value={'synthetic': True}), \
                 patch.object(p, 'validate', return_value=fixed_validation), \
                 patch.object(p, 'segmentation_loss', side_effect=loss), \
                 patch.object(p.torch.cuda, 'is_available', return_value=False), \
                 patch.dict(p.os.environ, {'WORLD_SIZE': '1', 'RANK': '0', 'LOCAL_RANK': '0'}):
                path = p.train(args)
            saved = torch.load(path, map_location='cpu', weights_only=True)
            reference = ScalarDecoder()
            optimizer = torch.optim.AdamW(reference.parameters(), lr=.01, weight_decay=.01)
            sampler = torch.utils.data.DistributedSampler(train_data, num_replicas=1, rank=0,
                                                         shuffle=True, seed=args.seed)
            order = list(sampler)
            for indices in (order[:4], order[4:]):
                optimizer.zero_grad(set_to_none=True)
                losses = [loss(reference(train_data[i]['features'], train_data[i]['baseline']),
                               train_data[i]['target']) for i in indices]
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
                optimizer.step()
            torch.testing.assert_close(saved['state_dict']['weight'], reference.weight.detach())
            self.assertEqual(int(saved['optimizer']['state'][0]['step']), 2)
            self.assertFalse(saved['metadata']['full_train320'])

    def test_test_gate_requires_full_training_independent_validation_and_positive_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_data = SimpleNamespace(source={'split': 'test', 'initial_checkpoint': {'synthetic': True}}, heads=('WT',),
                index={'binding': {'source': {'path': '/gt_free_test_manifest'},
                                   'contract': {'fixed': True}}})
            valid = {'implementation': p.implementation(), 'encoder_contract': {'fixed': True},
                'initial_checkpoint': {'synthetic': True},
                'heads': ('WT',), 'full_train320': True,
                'validation': {'full_val80': True, 'selected': {'alpha': .5,
                    'regions': {head: {'mean_gain_pp': .1 if head == 'WT' else 0.}
                                for head in ('WT', 'TC', 'ET')}}}}
            variants = []
            for key in ('full_train320', 'full_val80'):
                bad = copy.deepcopy(valid)
                if key == 'full_train320': bad[key] = False
                else: bad['validation'][key] = False
                variants.append(bad)
            for head, gain in (('WT', 0.), ('WT', -.1), ('TC', -.1), ('ET', -.1)):
                bad = copy.deepcopy(valid)
                bad['validation']['selected']['regions'][head]['mean_gain_pp'] = gain
                variants.append(bad)
            for i, metadata in enumerate(variants):
                checkpoint = root/f'checkpoint_{i}.pth'
                torch.save({'metadata': metadata, 'state_dict': {}}, checkpoint)
                args = SimpleNamespace(cache='/synthetic_cache', checkpoint=checkpoint,
                                       output=root/f'prediction_{i}', device='cpu')
                with self.subTest(i=i), \
                     patch.object(d, 'CachedSliceDataset', return_value=fake_data), \
                     patch.object(p, 'manifest', return_value=fake_data.source), \
                     patch.object(m, 'build_domain_decoder', side_effect=AssertionError('Must reject before model load')):
                    with self.assertRaises(ValueError): p.predict(args)


if __name__ == '__main__':
    unittest.main()
