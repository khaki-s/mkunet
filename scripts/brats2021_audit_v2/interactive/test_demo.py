"""GT boundaries, native coordinates, legacy-mask identity and manual prompt regression."""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nnunetv2.utilities.brats2021_audit_v2.interactive.demo import (
    DemoState, bounded_roi, labels_from_logits, load_case, metrics, prepare, verified_gt)
from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import RuleController
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import action_from_json
from nnunetv2.utilities.brats2021_audit_v2.interactive.atomic import atomic_point_transaction
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession


class CapturingEngine:
    """API test double only; never reachable from production CLI."""
    supported_interactions = {'bbox3d': False}

    def set_image(self, image):
        self.image = image

    def set_target_buffer(self, target):
        self.target = target

    def add_initial_seg_interaction(self, initial, run_prediction=False):
        self.target[:] = torch.from_numpy(initial)

    def add_point_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.last_point = coordinates
        self.target[coordinates] = include_interaction

    def add_bbox_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.last_box = coordinates
        self.target[tuple(slice(a, b) for a, b in coordinates)] = include_interaction


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def state(self, folder):
        shape = (40, 38, 36)
        raw = np.ones((4, *shape), np.float32)
        logits = np.full((3, *shape), -4.59512, np.float32)
        logits[:, 10:20, 11:21, 12:22] = 4.59512
        engine = CapturingEngine()
        backend = NNInteractiveBackend('.', raw[1:2], np.eye(4), modality_index=1, inference_session=engine)
        session = CorrectionSession(raw, [1, 1, 1, 1], logits, np.eye(4), backend)
        case = Path(folder) / 'case.npz'
        np.savez(case, mri=raw)
        return DemoState(session, RuleController(), raw, folder, case_path=case,
                         controller_name='rule', roi_size=16), engine

    def test_hard_mask_roundtrip_and_gt_free_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = np.zeros((7, 8, 9), np.uint8)
            labels[1:6, 2:7, 3:8] = 1
            labels[2:5, 3:6, 4:7] = 2
            labels[3, 4, 5] = 3
            affine = np.diag([-1., 2., 3., 1.])
            nib.save(nib.Nifti1Image(labels, affine), root / 'pred.nii.gz')
            nib.save(nib.Nifti1Image(np.ones(labels.shape, np.float32), affine), root / 'mri.nii.gz')
            prepare(argparse.Namespace(output=root / 'case.npz', prediction=root / 'pred.nii.gz',
                    t1=None, t1ce=root / 'mri.nii.gz', t2=None, flair=None, case_id='case', split='test'))
            case = load_case(root / 'case.npz')
            self.assertTrue(np.array_equal(labels_from_logits(case['logits']), labels))
            self.assertTrue(np.array_equal(case['affine'], affine))
            self.assertFalse(case['metadata']['gt_loaded'])
            with np.load(root / 'case.npz') as data:
                self.assertNotIn('atomic_labels', data.files)
            np.savez(root / 'leaky.npz', atomic_labels=labels)
            with self.assertRaisesRegex(ValueError, 'GT-free'):
                load_case(root / 'leaky.npz')
            np.savez(root / 'region_leak.npz', region_gt=labels)
            with self.assertRaisesRegex(ValueError, 'GT-free'):
                load_case(root / 'region_leak.npz')

    def test_negative_et_is_not_background_and_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            original = state.session.logits.clone()
            result = state.apply({'target': 'ET', 'text': 'ET分多了，保留周围水肿',
                                  'points': [{'voxel': [15, 16, 17], 'positive': False}]})
            self.assertEqual(result['action']['operation'], 'exclude')
            self.assertEqual(labels_from_logits(state.session.logits.numpy())[15, 16, 17], 2)
            self.assertTrue(torch.equal(original[:2], state.session.logits[:2]))
            state.undo()
            self.assertTrue(torch.equal(original, state.session.logits))

    def test_positive_box_promotes_parents_and_preserves_outside(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            original = state.session.logits.clone()
            result = state.apply({'target': 'ET', 'text': 'ET漏分了，补上框内ET', 'points': [],
                                 'box': {'start': [24, 24, 26], 'stop': [28, 29, 27], 'positive': True}})
            self.assertEqual(labels_from_logits(state.session.logits.numpy())[25, 25, 26], 3)
            self.assertEqual(engine.last_box[2][1] - engine.last_box[2][0], 1)
            self.assertTrue(torch.equal(original[:, 0], state.session.logits[:, 0]))
            saved = state.export()
            report = json.loads((Path(saved['saved']) / 'interactions.json').read_text())
            self.assertFalse(report['gt_loaded'])
            self.assertEqual(report['prompt_source'], 'human_browser')
            self.assertEqual(len(report['boxes']), 1)

    def test_native_slice_render_pixel_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            image = Image.open(io.BytesIO(state.slice_png(17, 1)))
            self.assertEqual(image.size, (40, 38))
            # x=15,y=16 maps to the existing ET red overlay; no axis swap.
            self.assertGreater(image.getpixel((15, 16))[0], image.getpixel((15, 16))[1])
            state.slice_png(17, 2)  # Merely viewing T2 does not change the T1ce backend.
            self.assertEqual(state.info()['correction_modality'], {'index': 1, 'name': 'T1ce'})

    def test_geometry_and_spatial_bounds_rejected(self):
        with self.assertRaises(ValueError):
            bounded_roi([{'voxel': [40, 0, 0]}], None, (40, 38, 36), 16)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'gt.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), np.eye(4)), path)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                verified_gt(path, (3, 4, 5), np.diag([2, 1, 1, 1]))

    def test_region_metric_contract(self):
        labels = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        self.assertEqual(metrics(labels, labels)['ET']['dice'], 1.)
        prediction = labels.copy()
        prediction[0, 0, 3] = 2
        scores = metrics(prediction, labels)
        self.assertEqual(scores['WT']['dice'], 1.)
        self.assertEqual(scores['TC']['dice'], 1.)
        self.assertEqual(scores['ET']['dice'], 0.)

    def test_explicit_ed_ncr_assignment_and_atomic_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            result = state.assign_class({'class_name': 'ED', 'voxel': [15, 16, 17]})
            self.assertEqual(result['visual_tool_calls'], 2)
            self.assertEqual(labels_from_logits(state.session.logits.numpy())[15, 16, 17], 1)
            state.undo()
            state.undo()
            result = state.assign_class({'class_name': 'NCR', 'voxel': [15, 16, 17]})
            self.assertEqual(labels_from_logits(state.session.logits.numpy())[15, 16, 17], 2)
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            original = state.session.logits.clone()
            call_count = [0]
            old = engine.add_point_interaction
            def fail_second(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    raise RuntimeError('intentional test-double failure')
                return old(*args, **kwargs)
            engine.add_point_interaction = fail_second
            with self.assertRaises(RuntimeError):
                state.assign_class({'class_name': 'ED', 'voxel': [15, 16, 17]})
            self.assertTrue(torch.equal(original, state.session.logits))
            self.assertEqual(state.session.history, [])
            self.assertEqual(state.commands, [])

    def test_prior_et_confirmation_can_be_replaced_replayed_and_undone(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            state.apply({'target': 'ET', 'text': 'ET漏分了，补上正点',
                         'points': [{'voxel': [15, 16, 17], 'positive': True}]})
            before = state.session.logits.clone()
            before_labels = state.session.confirmed_labels.clone()
            before_known = state.session.confirmed_mask.clone()
            state.assign_class({'class_name': 'ED', 'voxel': [15, 16, 17]})
            self.assertEqual(labels_from_logits(state.session.logits.numpy())[15, 16, 17], 1)
            self.assertEqual(state.session.confirmed_labels[:, 15, 16, 17].tolist(), [1, 0, 0])
            self.assertTrue(state.session.confirmed_mask[:, 15, 16, 17].all())
            replay, _ = self.state(folder)
            for key, value in state.session.rois.items():
                replay.session.register_roi(key, value)
            for value in state.session.points.values():
                replay.session.register_point(value)
            for command in state.commands:
                if 'atomic_assignment' in command:
                    assignment = command['atomic_assignment']
                    with atomic_point_transaction(replay.session, assignment['voxel'], assignment['class_name']):
                        for action in command['actions']:
                            replay.session.apply_action(action_from_json(json.dumps(action), replay.session))
                else:
                    replay.session.apply_action(action_from_json(json.dumps(command['action']), replay.session))
            self.assertTrue(torch.equal(replay.session.logits, state.session.logits))
            self.assertTrue(torch.equal(replay.session.confirmed_labels, state.session.confirmed_labels))
            self.assertTrue(torch.equal(replay.session.confirmed_mask, state.session.confirmed_mask))
            state.undo()
            state.undo()
            self.assertTrue(torch.equal(state.session.logits, before))
            self.assertTrue(torch.equal(state.session.confirmed_labels, before_labels))
            self.assertTrue(torch.equal(state.session.confirmed_mask, before_known))

    def test_failed_replacement_restores_prior_point_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            state, engine = self.state(folder)
            state.apply({'target': 'ET', 'text': 'ET漏分了，补上正点',
                         'points': [{'voxel': [15, 16, 17], 'positive': True}]})
            before = state.session.logits.clone()
            before_labels = state.session.confirmed_labels.clone()
            before_known = state.session.confirmed_mask.clone()
            old = engine.add_point_interaction
            calls = [0]
            def fail_second(*args, **kwargs):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError('intentional test-double failure')
                return old(*args, **kwargs)
            engine.add_point_interaction = fail_second
            with self.assertRaises(RuntimeError):
                state.assign_class({'class_name': 'ED', 'voxel': [15, 16, 17]})
            self.assertTrue(torch.equal(state.session.logits, before))
            self.assertTrue(torch.equal(state.session.confirmed_labels, before_labels))
            self.assertTrue(torch.equal(state.session.confirmed_mask, before_known))
            self.assertEqual(len(state.session.history), 1)
            self.assertEqual(len(state.commands), 1)


if __name__ == '__main__':
    unittest.main()
