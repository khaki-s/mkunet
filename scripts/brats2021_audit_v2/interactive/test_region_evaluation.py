"""Synthetic integration tests; no checkpoints, patient inference, or training."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
from nnunetv2.utilities.brats2021_audit_v2.interactive.evaluation import (
    assert_locked, evaluate_trajectory, labels_from_logits, labels_to_logits, register_prompt,
    summarize_run, run_evaluation)
from nnunetv2.utilities.brats2021_audit_v2.interactive import evaluation as evaluation_module
from nnunetv2.utilities.brats2021_audit_v2.interactive.evaluation_manifest import (
    check_file, prepare_manifest, read_json, write_json)
from nnunetv2.utilities.brats2021_audit_v2.interactive.prompt_sampling import select_prompt
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import PointPrompt, SpatialROI, StructuredAction


class RecordingEngine:
    """Deliberately simple upstream API double. Not available from the real CLI."""
    supported_interactions = {'bbox3d': False}

    def __init__(self):
        self.calls = []
        self.predictions = 0
        self.prediction_flags = []

    def set_image(self, image):
        self.image = image

    def set_target_buffer(self, target):
        self.buffer = target

    def add_initial_seg_interaction(self, initial, run_prediction=False):
        self.buffer[:] = torch.from_numpy(initial)

    def add_point_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.calls.append(('point', coordinates, include_interaction))
        self.prediction_flags.append(run_prediction)
        self.buffer[coordinates] = include_interaction
        self.predictions += int(run_prediction)

    def add_bbox_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.calls.append(('box', coordinates, include_interaction))
        self.prediction_flags.append(run_prediction)
        self.buffer[tuple(slice(a, b) for a, b in coordinates)] = include_interaction
        self.predictions += int(run_prediction)


CONFIG = {'budgets': [0, 1, 3, 5], 'targets': ['ED', 'ET', 'NCR', 'joint'],
          'seed': 5, 'roi_size': [16, 16, 16], 'box_axis': 2, 'box_padding': 0,
          'box_max_extent': 8, 'tolerance': 1e-6}


class RegionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def session(self, labels):
        raw = np.ones((4, *labels.shape), np.float32)
        engine = RecordingEngine()
        backend = NNInteractiveBackend('.', raw[1:2], np.eye(4), modality_index=1,
                                        inference_session=engine, replay_box_history=True)
        session = CorrectionSession(raw, [1, 1, 1, 1], labels_to_logits(labels), np.eye(4), backend,
                                    case_id='synthetic', split='validation')
        return session, engine

    def test_all_class_transitions_one_finite_prompt(self):
        for before in range(4):
            for after in range(4):
                if before == after:
                    continue
                with self.subTest(before=before, after=after):
                    pred = np.zeros((16, 16, 16), np.uint8)
                    gt = pred.copy()
                    pred[8, 8, 8], gt[8, 8, 8] = before, after
                    session, engine = self.session(pred)
                    result = evaluate_trajectory(session, gt, target='joint', kind='point', config=CONFIG)
                    self.assertEqual(result['budgets']['1']['actual_prompts'], 1)
                    self.assertEqual(result['budgets']['5']['actual_prompts'], 1)
                    self.assertTrue(result['budgets']['5']['carried_forward'])
                    self.assertEqual(result['stop_reason'], 'no_current_target_error')
                    self.assertTrue(np.array_equal(labels_from_logits(session.logits), gt))
                    self.assertEqual(engine.predictions, 1)
                    self.assertEqual(len(engine.calls), 1)
                    # The only upstream spatial annotation is one coordinate.
                    self.assertEqual(engine.calls[0][0], 'point')
                    self.assertEqual(engine.image.shape, (1, 16, 16, 16))

    def test_no_error_no_tool_call_and_budget_zero_identity(self):
        labels = np.zeros((16, 16, 16), np.uint8)
        labels[8, 8, 8] = 2
        session, engine = self.session(labels)
        result = evaluate_trajectory(session, labels, target='NCR', kind='point', config=CONFIG)
        self.assertEqual(result['budgets']['0']['metrics'], result['baseline'])
        self.assertEqual(result['budgets']['5']['actual_prompts'], 0)
        self.assertEqual(engine.predictions, 0)
        self.assertTrue(result['budgets']['5']['carried_forward'])

    def test_new_prediction_selects_new_error_and_trajectories_reset(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        gt = pred.copy()
        gt[6, 6, 6] = 1
        gt[9, 9, 9] = 3
        session, engine = self.session(pred)
        result = evaluate_trajectory(session, gt, target='joint', kind='point', config=CONFIG)
        positions = [tuple(s['prompt']['voxel']) for s in result['steps']]
        self.assertEqual(len(positions), 2)
        self.assertEqual(len(set(positions)), 2)
        self.assertEqual(engine.predictions, 2)
        reset, _ = self.session(pred)
        et_only = evaluate_trajectory(reset, gt, target='ET', kind='point', config=CONFIG)
        self.assertEqual(et_only['baseline'], result['baseline'])
        self.assertEqual(et_only['budgets']['5']['actual_prompts'], 1)
        self.assertEqual(labels_from_logits(reset.logits)[6, 6, 6], 0)

    def test_box_has_no_hidden_point_and_replays_old_box(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        gt = pred.copy()
        gt[5:7, 5:7, 8] = 3
        gt[10:12, 10:12, 8] = 3
        session, engine = self.session(pred)
        result = evaluate_trajectory(session, gt, target='ET', kind='box', config=CONFIG)
        self.assertEqual(result['budgets']['5']['actual_prompts'], 2)
        self.assertEqual(engine.predictions, 2)
        self.assertTrue(all(c[0] == 'box' for c in engine.calls))
        self.assertEqual(len(engine.calls), 3)  # one first turn; prior+new second turn
        self.assertEqual(len(session.points), 0)
        self.assertEqual(int(session.confirmed_mask.sum()), 0)

    def test_new_click_is_last_after_history_replay(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        gt = pred.copy()
        gt[6, 6, 6] = 3
        gt[9, 9, 9] = 3
        session, engine = self.session(pred)
        result = evaluate_trajectory(session, gt, target='ET', kind='point', config=CONFIG)
        self.assertEqual(engine.predictions, 2)
        self.assertEqual(len(engine.calls), 3)
        first, second = (tuple(step['prompt']['voxel']) for step in result['steps'])
        self.assertEqual(engine.calls[-2][1], first)
        self.assertEqual(engine.calls[-1][1], second)

    def test_pair_submits_new_box_then_point_once_and_counts_two_annotations(self):
        for before, after in ((0, 3), (3, 2)):
            with self.subTest(before=before, after=after):
                pred = np.zeros((16, 16, 16), np.uint8)
                gt = pred.copy()
                pred[8, 8, 8], gt[8, 8, 8] = before, after
                session, engine = self.session(pred)
                result = evaluate_trajectory(session, gt, target='joint', kind='point_box', config=CONFIG)
                prompt = result['steps'][0]['prompt']
                self.assertEqual([call[0] for call in engine.calls], ['box', 'point'])
                self.assertEqual(engine.calls[-1][1], tuple(prompt['voxel']))
                self.assertEqual(engine.prediction_flags, [False, True])
                self.assertEqual(engine.predictions, 1)
                self.assertEqual(len(session.points), 1)
                self.assertEqual(len(session.boxes), 1)
                action = result['steps'][0]['action_result']['action']
                self.assertTrue(action['box_prompt'])
                self.assertEqual(len(action['positive_point_ids'] + action['negative_point_ids']), 1)
                self.assertNotEqual(action['box_id'], next(iter(session.points)))
                self.assertEqual(result['budget_unit'], 'rounds')
                self.assertEqual(result['annotations_per_round'], 2)
                for budget in ('1', '3', '5'):
                    state = result['budgets'][budget]
                    self.assertEqual(state['actual_prompts'], 2)
                    self.assertEqual(state['actual_points'], 1)
                    self.assertEqual(state['actual_boxes'], 1)
                    self.assertEqual(state['actual_rounds'], 1)
                    self.assertEqual(state['actual_tool_calls'], 1)
                self.assertTrue(result['budgets']['5']['carried_forward'])
                self.assertTrue(np.array_equal(labels_from_logits(session.logits), gt))

    def test_pair_reselects_new_error_and_history_replay_does_not_cost_annotations(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        gt = pred.copy()
        gt[6, 6, 6] = gt[9, 9, 9] = 3
        session, engine = self.session(pred)
        result = evaluate_trajectory(session, gt, target='ET', kind='point_box', config=CONFIG)
        positions = [tuple(s['prompt']['voxel']) for s in result['steps']]
        self.assertEqual(len(set(positions)), 2)
        self.assertEqual(engine.predictions, 2)
        self.assertEqual(len(engine.calls), 6)  # 2 new, then 2 old + 2 new
        self.assertEqual(engine.calls[-2][0], 'box')
        self.assertEqual(engine.calls[-1][1], positions[-1])
        self.assertEqual(engine.prediction_flags, [False, True, False, False, False, True])
        self.assertEqual(result['budgets']['5']['actual_prompts'], 4)
        self.assertEqual(result['budgets']['5']['actual_rounds'], 2)
        self.assertEqual(result['budgets']['5']['actual_tool_calls'], 2)
        self.assertEqual(result['stop_reason'], 'no_current_target_error')

    def test_pair_five_rounds_means_ten_annotations_and_no_error_means_zero(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        gt = pred.copy()
        for x in (3, 5, 7, 9, 11, 13):
            gt[x, 8, 8] = 3
        session, engine = self.session(pred)
        result = evaluate_trajectory(session, gt, target='ET', kind='point_box', config=CONFIG)
        state = result['budgets']['5']
        self.assertEqual(state['actual_prompts'], 10)
        self.assertEqual(state['actual_points'], 5)
        self.assertEqual(state['actual_boxes'], 5)
        self.assertEqual(state['actual_rounds'], 5)
        self.assertEqual(engine.predictions, 5)
        empty, no_engine = self.session(pred)
        stopped = evaluate_trajectory(empty, pred, target='joint', kind='point_box', config=CONFIG)
        self.assertEqual(stopped['budgets']['5']['actual_prompts'], 0)
        self.assertEqual(stopped['budgets']['5']['actual_rounds'], 0)
        self.assertTrue(stopped['budgets']['5']['carried_forward'])
        self.assertEqual(no_engine.predictions, 0)

    def test_negative_et_preserves_tc_and_wt(self):
        pred = np.zeros((16, 16, 16), np.uint8)
        pred[8, 8, 8] = 3
        gt = pred.copy()
        gt[8, 8, 8] = 2
        session, _ = self.session(pred)
        result = evaluate_trajectory(session, gt, target='ET', kind='point', config=CONFIG)
        metrics = result['budgets']['1']['metrics']
        self.assertEqual(metrics['WT']['dice'], 1)
        self.assertEqual(metrics['TC']['dice'], 1)
        self.assertEqual(metrics['NCR']['dice'], 1)

    def test_full_context_uses_global_coordinates_but_edits_only_roi(self):
        pred = np.zeros((32, 32, 32), np.uint8)
        session, engine = self.session(pred)
        session.corrector.full_image_context = True
        session.register_roi('local', SpatialROI((16, 16, 16), (32, 32, 32)))
        session.register_point(PointPrompt('new', (24, 24, 24), 'ET', True))
        session.apply_action(StructuredAction('ET', 'include', 'local', positive_point_ids=('new',)))
        self.assertEqual(engine.image.shape, (1, 32, 32, 32))
        self.assertEqual(engine.calls[-1][1], (24, 24, 24))
        result = labels_from_logits(session.logits)
        self.assertEqual(result[24, 24, 24], 3)
        self.assertEqual(int(np.count_nonzero(result)), 1)

    def fixture_manifest(self, root, split='validation'):
        for name, ids in [('val', ['v1', 'v2']), ('test', ['t1'])]:
            rows = []
            for key in ids:
                path = root / f'{key}.nii.gz'
                nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), path)
                rows.append({'case_id': key, 'modalities': {m: str(path) for m in ('T1', 'T1ce', 'T2', 'FLAIR')}})
            write_json(root / f'{name}.json', {'cases': rows})
        write_json(root / 'split.json', [{'train': ['train1'], 'val': ['v1', 'v2']}])
        ids = ['v1', 'v2'] if split == 'validation' else ['t1']
        write_json(root / 'provenance.json', {'status': 'complete', 'contract': {
            'case_ids': ids, 'provenance': {'checkpoint': 'best.pth', 'checkpoint_sha256': 'a'*64}}})
        return argparse.Namespace(split=split, source_manifest=root / ('val.json' if split == 'validation' else 'test.json'),
            splits_file=root / 'split.json', fold=0, predictions=root, gt=root,
            prediction_provenance=root / 'provenance.json', test_manifest=root / 'test.json' if split == 'test' else None,
            allow_unverified_initialization=False, output=root / 'prepared.json')

    def test_split_membership_and_input_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture_manifest(root)
            manifest = prepare_manifest(args)
            self.assertEqual(len(manifest['cases']), 2)
            image = manifest['cases'][0]['prediction']
            check_file(image)
            Path(image['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                check_file(image)
            args.output = root / 'invalid.json'
            args.source_manifest = root / 'test.json'
            with self.assertRaisesRegex(ValueError, 'exactly match'):
                prepare_manifest(args)

    def test_test_whitelist_and_lock_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.fixture_manifest(root, split='test')
            self.assertEqual(prepare_manifest(args)['split'], 'test')
            args.output = root / 'invalid.json'
            args.test_manifest = None
            with self.assertRaisesRegex(ValueError, 'test-manifest'):
                prepare_manifest(args)
        science = {'prompt_kind': 'point', 'protocol': copy.deepcopy(CONFIG)}
        assert_locked({'science': copy.deepcopy(science)}, science)
        other = copy.deepcopy(science)
        other['protocol']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'protocol'):
            assert_locked({'science': other}, science)

    def test_incomplete_summary_exposes_missing_case(self):
        labels = np.zeros((16, 16, 16), np.uint8)
        session, _ = self.session(labels)
        record = evaluate_trajectory(session, labels, target='ET', kind='point', config=CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / 'run.json', {'identity': 'test', 'split': 'validation', 'pilot': True,
                'case_ids': ['synthetic', 'missing'], 'science': {'prompt_kind': 'point', 'protocol': CONFIG}})
            write_json(root / 'records/ET/synthetic.json', record)
            summary = summarize_run(root)
            self.assertFalse(summary['coverage']['by_target']['ET']['full_cohort_complete'])
            self.assertEqual(summary['coverage']['by_target']['ET']['missing'], 1)
            self.assertTrue((root / 'region_summary.csv').is_file())

    def test_batch_resume_and_changed_protocol_rejected_without_real_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_args = self.fixture_manifest(root)
            prepare_manifest(prepare_args)
            config = dict(CONFIG, modality_index=1, model_folder=str(root))
            write_json(root / 'protocol.json', config)
            args = argparse.Namespace(manifest=root / 'prepared.json', protocol=root / 'protocol.json',
                kind='point', device='cpu', limit=None, resume=True, save_predictions='none',
                output=root / 'run', locked_protocol=None)

            def fake_backend(folder, raw, affine, **kwargs):
                kwargs['inference_session'] = kwargs['inference_session'] or RecordingEngine()
                return NNInteractiveBackend(folder, raw, affine, **kwargs)

            def fake_science(manifest, protocol, kind):
                return {'protocol': protocol, 'prompt_kind': kind}

            with patch.object(evaluation_module, 'NNInteractiveBackend', side_effect=fake_backend) as backend, \
                 patch.object(evaluation_module, 'scientific_identity', side_effect=fake_science), \
                 patch.object(evaluation_module, 'version', return_value='synthetic-test'):
                self.assertEqual(run_evaluation(args)['status'], 'complete')
                self.assertEqual(backend.call_count, 8)
                backend.reset_mock()
                self.assertEqual(run_evaluation(args)['status'], 'complete')
                self.assertEqual(backend.call_count, 0)
                # A failed condition is retried, not silently removed from the cohort.
                record_path = root / 'run/records/ED/v1.json'
                failed = read_json(record_path)
                failed['status'] = 'failed'
                write_json(record_path, failed)
                run_evaluation(args)
                self.assertEqual(backend.call_count, 1)
                self.assertEqual(read_json(record_path)['status'], 'complete')
                config['seed'] += 1
                write_json(root / 'protocol.json', config)
                with self.assertRaisesRegex(ValueError, 'Resume identity mismatch'):
                    run_evaluation(args)
                config['seed'] -= 1
                write_json(root / 'protocol.json', config)
                (root / 'v1.nii.gz').write_bytes(b'changed after a complete run')
                with self.assertRaisesRegex(ValueError, 'Input changed'):
                    run_evaluation(args)
                self.assertEqual(read_json(root / 'run/run.json')['status'], 'incomplete')
                self.assertFalse(read_json(root / 'run/summary.json')['valid_for_full_cohort_reporting'])


if __name__ == '__main__':
    unittest.main()
