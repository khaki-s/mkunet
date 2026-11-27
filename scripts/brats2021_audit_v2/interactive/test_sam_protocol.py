"""Paired mask geometry, residual support and study gate regressions."""
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

import foundation_prompt

PACKAGE = foundation_prompt.name + '.foundation_prompt_v2.sam_protocol.'
dense = importlib.import_module(PACKAGE + 'dense')
candidates = importlib.import_module(PACKAGE + 'candidates')
residual = importlib.import_module(PACKAGE + 'residual')
scoring = importlib.import_module(PACKAGE + 'scoring')
matrix = importlib.import_module(PACKAGE + 'evaluation_matrix')
shape_fusion = importlib.import_module(PACKAGE + 'shape_fusion')
background = importlib.import_module(PACKAGE + 'background')
cohorts = importlib.import_module(PACKAGE + 'cohorts')


class FakeSam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inst_interactive_predictor = SimpleNamespace(model=SimpleNamespace(
            sam_prompt_encoder=SimpleNamespace(mask_input_size=(11, 13))))
        self.prompts = []

    def predict_inst(self, state, **prompt):
        self.prompts.append(prompt)
        blank = np.zeros((1, 11, 13), bool)
        return blank, np.array([.8]), None


class FakeProcessor:
    resolution = 1008

    def __init__(self):
        self.size = []

    def set_image(self, image):
        self.size.append(image.size)
        return len(self.size)


class SamProtocolTests(TestCase):
    def test_native_point_box_preserves_both_edit_directions(self):
        baseline = np.array([False, True, True, False])
        additions = np.array([True, True, True, False])
        removals = np.array([False, False, True, False])
        expected = np.array([True, False, True, False])
        combined = residual.combine_native_point_box(baseline, additions, removals)
        np.testing.assert_array_equal(combined, expected)
        replies = np.stack((baseline, additions, removals))
        indices = np.array([0, 1, 3])
        np.testing.assert_array_equal(residual._native_point_values(replies, baseline, indices), expected[indices])

    def test_multimodal_pair_keeps_signed_points_and_box_fixed(self):
        model, processor = FakeSam(), FakeProcessor()
        refiner = background.BackgroundPairedRefiner('cpu', model=model, processor=processor)
        baseline = np.zeros((11, 13, 1), np.uint8)
        baseline[2:9, 2:10, 0] = 1
        mri = np.full((4, *baseline.shape), .1, dtype=np.float32)
        mri[:, baseline == 1] = .9
        masks, _, report = refiner.generate_pair(mri, baseline, [True]*4, (1., 1., 1.))
        self.assertEqual(masks.shape, (2, 3, *baseline.shape))
        self.assertGreater(report['background_points'], 0)
        self.assertNotEqual(model.prompts[0]['mask_input'].tobytes(), model.prompts[1]['mask_input'].tobytes())
        for field in ('point_coords', 'point_labels', 'box'):
            np.testing.assert_array_equal(model.prompts[0][field], model.prompts[1][field])
        self.assertIn(0, model.prompts[0]['point_labels'])

    def test_sam_profile_uses_the_approved_factory_and_source(self):
        with patch.object(dense, 'sam_contract', return_value={'checkpoint': 'fixed'}):
            approved = background.contract({})
            config, factory = cohorts.sam_cache_spec({}, approved)
            self.assertEqual(config, approved)
            self.assertIs(factory, background.BackgroundPairedRefiner)
            with self.assertRaisesRegex(ValueError, 'profile, weights or source changed'):
                cohorts.sam_cache_spec({}, {**approved, 'background_prompt_code': 'modified'})

    def test_signed_boundary_preserves_native_grid_and_identity_initialization(self):
        mask = np.zeros((15, 17, 19), dtype=bool)
        mask[4:11, 4:13, 5:14] = True
        distance = shape_fusion.signed_distance(mask, (2., 1., 1.))
        self.assertEqual(distance.shape, mask.shape)
        self.assertTrue(np.array_equal(distance > 0, mask))
        self.assertEqual(float(distance[3, 8, 8]), -2.)
        self.assertEqual(float(distance[4, 8, 8]), 2.)
        images = torch.zeros((1, shape_fusion.CHANNELS, *mask.shape))
        images[0, -2] = torch.from_numpy(distance.astype(np.float32)) / shape_fusion.RADIUS_MM
        with torch.no_grad():
            field = shape_fusion.ShapeReadout()(images)
        self.assertEqual(tuple(field.shape), (1, 1, *mask.shape))
        self.assertTrue(np.array_equal(field[0, 0].numpy() > 0, mask))

    def test_pair_shares_identical_point_box_and_native_shape(self):
        model = FakeSam()
        processor = FakeProcessor()
        engine = dense.PairedSamRefiner('cpu', model=model, processor=processor)
        baseline = np.zeros((11, 13, 3), dtype=np.uint8)
        baseline[3:7, 3:8, 1] = 3
        mri = np.ones((4, *baseline.shape), dtype=np.float32)
        candidates, scores, report = engine.generate_pair(mri, baseline, [1, 1, 1, 1], (1, 1, 1))
        self.assertEqual(candidates.shape, (2, 3, *baseline.shape))
        self.assertEqual(scores.shape, (2, 3, baseline.shape[2]))
        self.assertEqual(report['encodes'], 2)
        self.assertEqual(processor.size, [(13, 11), (13, 11)])
        for first, second in zip(model.prompts[::2], model.prompts[1::2]):
            for k in ('point_coords', 'point_labels', 'box'):
                np.testing.assert_array_equal(first[k], second[k])
            initial, weak = first['mask_input'], second['mask_input']
            np.testing.assert_array_equal(np.sign(initial), np.sign(weak))
            self.assertTrue((np.abs(weak) <= np.abs(initial)).all())
            self.assertFalse(np.array_equal(initial, weak))

    def test_one_modality_usable_for_all_heads(self):
        model = FakeSam()
        processor = FakeProcessor()
        engine = dense.PairedSamRefiner('cpu', model=model, processor=processor)
        baseline = np.zeros((11, 13, 3), dtype=np.uint8)
        baseline[4:8, 5:9, 1] = 3
        mri = np.zeros((4, *baseline.shape), dtype=np.float32)
        mri[0] = 1.
        _, _, report = engine.generate_pair(mri, baseline, matrix.availability(1), (1, 1, 1))
        self.assertEqual(report['modalities'], [0, 0, 0])
        self.assertEqual(report['encodes'], 1)
        combinations = [tuple(matrix.availability(i)) for i in matrix.CODES]
        self.assertEqual(len(set(combinations)), 15)
        self.assertEqual(combinations[0], (1, 1, 1, 1))

    def test_point_box_geometry_follows_real_connected_disagreement(self):
        shape = (17, 19, 15)
        base = np.zeros(shape, bool)
        base[6:12, 5:11, 4:11] = True
        changed = base.copy()
        changed[12:14, 6:10, 5:9] = True
        changed[6:8, 5:8, 4:6] = False
        for positive in (True, False):
            p = candidates.locate_component(changed, base, np.ones(shape), [1., 1., 3.], positive)
            pos = tuple(p['point'])
            self.assertEqual(changed[pos], positive)
            self.assertNotEqual(base[pos], positive)
            self.assertEqual(p['box'][2][1]-p['box'][2][0], 1)
            self.assertTrue(all(lo <= point < hi for point, (lo, hi) in zip(pos, p['box'])))

    def test_edit_support_only_where_real_response_varies(self):
        shape = (9, 10, 11)
        initial = np.zeros(shape, dtype=np.uint8)
        initial[3:7, 3:7, 3:7] = 1
        raw = initial > 0
        raw[2, 2, 2] = True
        mri = np.ones((4, *shape), np.float32)
        masks = np.stack([initial > 0] * 3)
        present, _ = residual.observations(mri, initial, raw, masks, np.zeros(shape[2]), np.eye(4), 0)
        self.assertEqual(len(present), 0)
        masks[0, 4, 4, 7] = True
        present, features = residual.observations(mri, initial, raw, masks, np.zeros(shape[2]), np.eye(4), 0)
        np.testing.assert_array_equal(present, [np.ravel_multi_index((4,4,7), shape)])
        self.assertTrue(np.isfinite(features).all())

    def test_exact_gain_and_stop(self):
        initial = np.array([[[1,1,0,0,1,0]]], dtype=bool)
        gt = np.array([[[1,0,1,0,1,0]]], dtype=bool)
        pred = np.array([.95, .1, .9, .2, .95, .1])
        loc = np.arange(len(pred))
        gains = residual._gains(pred, loc, initial, gt)
        self.assertAlmostEqual(gains[-1,-1], 0.)
        self.assertAlmostEqual(gains.max(), 100./3)
        adds, removals = residual._changed(pred, loc, initial, None, None)
        self.assertFalse(adds.any() or removals.any())

    def test_gate_does_not_create_formal_permission_for_subthreshold_gain(self):
        with TemporaryDirectory() as root:
            target = Path(root) / 'permit.json'
            with patch.object(scoring, '_validated_report', return_value={
                'regions':{'WT':{'n_paired':25,'gain_pp':.999}}
            }):
                with self.assertRaisesRegex(ValueError, 'Formal training locked'):
                    scoring.admit(SimpleNamespace(score='unneeded', output=target))
            self.assertFalse(target.exists())

    def test_test_workers_can_release_sam_before_interactive(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'train.json'
            source.write_text('{}')
            permit = {'cohort': {'path': 'locked'}, 'models': [], 'variant': 'sdf', 'trainer': 'gpu_readout'}
            args = SimpleNamespace(output=root, mask_code=15, model=root, shard_index=0, num_shards=1,
                                   matrix=root, permit=root, phase='sam')
            data = {'new_cases': 1, 'completed': 1}
            with (patch.object(matrix, '_configuration', return_value=permit),
                  patch.object(matrix, 'test_rows', return_value=[{'case_id': 'case'}]),
                  patch.object(matrix, 'read_cohort', return_value={'train_manifest': {'path': source}}),
                  patch.object(matrix, 'sam_cache_spec', return_value=({}, dense.PairedSamRefiner)),
                  patch.object(scoring, 'model_details', return_value=('gpu_readout', 'sdf', {}, [], {})),
                  patch.object(dense, 'build_cache', return_value=data) as sam,
                  patch.object(candidates, 'build_responses', return_value=data) as interactive):
                report = matrix.test_predict(args)
                sam.assert_called_once()
                interactive.assert_not_called()
                args.phase = 'interactive'
                self.assertEqual(matrix.test_predict(args), report)
                sam.assert_called_once()
                interactive.assert_called_once()
                self.assertEqual(json.loads(report.read_text()), {'SAM3': data, 'nnInteractive': data})

    def test_summaries_have_all_15_types_and_prioritize_mask_15(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            matrix_file, permit_file = root/'matrix.json', root/'permit.json'
            matrix_file.write_text('{}')
            permit_file.write_text('{}')
            permit = {'models':[]}
            for code in matrix.CODES:
                scores = [{'region': name, 'baseline_dice': .9,
                           'dice': .905 if code == 15 and name == 'WT' else (.91 if code == 15 else .89),
                           'gain_pp': .5 if code == 15 and name == 'WT' else (1. if code == 15 else -1.),
                           'ref_mm3': 3., 'pred_mm3': 3.,
                           'baseline_fp_mm3':0.,'baseline_fn_mm3':1.,'fp_mm3':0.,'fn_mm3':0.,
                           'hd95_mm':0.,'hd95_status':'finite'} for name in ('WT','TC','ET','ED','NCR')]
                patient_metrics = [dict(**r, case_id=f'case_{i}', mask_code=code, n_modalities=code.bit_count())
                                   for i in range(100) for r in scores]
                for row in patient_metrics[:5]:
                    if row['region'] == 'ET':
                        row.update(baseline_dice=None, dice=None, gain_pp=None, ref_mm3=0., pred_mm3=0.)
                sub = root/'metrics'/f'mask_{code:02d}'
                sub.mkdir(parents=True)
                (sub/'summary.json').write_text(json.dumps({'schema':'sam_test100_subset_score_v1',
                    'n_cases':100, 'mask_code':code, 'test_matrix':scoring.file_identity(matrix_file),
                    'formal_permit':scoring.file_identity(permit_file), 'model':[], 'modalities':['T1'],
                    'outputs':[0]*100, 'per_case':patient_metrics, 'regions':scoring.summary(patient_metrics)}))
            with patch.object(matrix, '_configuration', return_value=permit):
                path = matrix.test_summary(SimpleNamespace(output=root,matrix=matrix_file,permit=permit_file))
            result = json.loads(path.read_text())
            self.assertFalse(result['all_targets_passed'])
            self.assertTrue(result['all_minimums_passed'])
            self.assertAlmostEqual(result['primary_gain_pp']['WT'], .5)
            self.assertTrue(result['wt_half_pp_reached'])
            self.assertFalse(result['wt_preferred_1pp_reached'])
            self.assertEqual(result['primary_results']['ET']['n_valid_dice'], 99)
            self.assertAlmostEqual(result['primary_gain_pp']['ET'], 1.)
            self.assertEqual(result['n_masks'], 15)
            self.assertEqual(len(result['by_mask']), 75)
