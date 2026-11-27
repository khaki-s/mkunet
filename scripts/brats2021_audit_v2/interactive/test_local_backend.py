"""CPU checks for the optional local proposal filter; no model weights or GT."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
from nnunetv2.utilities.brats2021_audit_v2.interactive.local_backend import (
    SpatiallyConstrainedNNInteractiveBackend)
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import (
    BoxPrompt, PointPrompt, SpatialROI, StructuredAction)


class CapturingEngine:
    """Official API test double; never exposed in the production evaluator."""
    supported_interactions = {'bbox3d': False}

    def __init__(self):
        self.calls = []
        self.predictions = 0

    def set_image(self, image):
        self.image = image

    def set_target_buffer(self, output):
        self.output = output

    def add_initial_seg_interaction(self, initial, run_prediction=False):
        self.output[:] = torch.from_numpy(initial)

    def add_point_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.calls.append(('point', coordinates, include_interaction))
        self.output[coordinates] = include_interaction
        self.predictions += int(run_prediction)

    def add_bbox_interaction(self, coordinates, include_interaction=True, run_prediction=True):
        self.calls.append(('box', coordinates, include_interaction))
        self.output[tuple(slice(a, b) for a, b in coordinates)] = include_interaction
        self.predictions += int(run_prediction)


def state(labels=None, affine=None, shape=(9, 9, 9)):
    if labels is None:
        labels = np.zeros(shape, dtype=np.uint8)
    shape = labels.shape
    logits = np.stack([np.where(labels > head, 4.59512, -4.59512)
                       for head in range(3)]).astype(np.float32)
    return SimpleNamespace(
        shape=shape,
        mri=torch.stack([torch.full(shape, float(index + 1)) for index in range(4)]),
        modality_mask=torch.ones(4, dtype=torch.bool),
        logits=torch.from_numpy(logits),
        affine=np.eye(4) if affine is None else affine,
        points={}, boxes={}, history=[],
        confirmed_mask=torch.zeros((3, *shape), dtype=torch.bool),
        confirmed_labels=torch.zeros((3, *shape)),
    )


def point_action(session, target='WT', positive=True, voxel=(4, 4, 4), point_id='p'):
    session.points[point_id] = PointPrompt(point_id, voxel, target, positive)
    return StructuredAction(target, 'include' if positive else 'exclude', 'roi',
                            positive_point_ids=(point_id,) if positive else (),
                            negative_point_ids=() if positive else (point_id,))


class LocalBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def backend(self, session, engine=None, **kwargs):
        return SpatiallyConstrainedNNInteractiveBackend(
            '.', session.mri[1:2].numpy(), session.affine, modality_index=1,
            inference_session=engine or CapturingEngine(), **kwargs)

    def evaluate_fake_proposal(self, backend, session, action, proposal, roi=None):
        roi = roi or SpatialROI((0, 0, 0), tuple(session.logits.shape[1:]))
        model_input = torch.zeros((1, 21, *roi.shape))
        with patch.object(NNInteractiveBackend, 'predict_delta', return_value=proposal) as parent:
            output = backend.predict_delta(model_input, session=session, action=action, roi=roi)
            self.assertEqual(set(parent.call_args.kwargs), {'session', 'action', 'roi'})
        return output

    def test_include_exclude_direction_and_other_heads_zero(self):
        session = state()
        backend = self.backend(session, radius_mm=3.0, same_source_class=False)
        self.assertTrue(backend.full_image_context)
        proposal = torch.full((1, 3, 9, 9, 9), 2.0)
        proposal[0, 0, 4, 4, 5] = -3.0
        include = point_action(session)
        result = self.evaluate_fake_proposal(backend, session, include, proposal)
        self.assertEqual(float(result[0, 0, 4, 4, 4]), 2.0)
        self.assertEqual(float(result[0, 0, 4, 4, 5]), 0.0)
        self.assertEqual(int(torch.count_nonzero(result[:, 1:])), 0)
        exclude = point_action(session, positive=False, point_id='negative')
        result = self.evaluate_fake_proposal(backend, session, exclude, proposal)
        self.assertEqual(float(result[0, 0, 4, 4, 4]), 0.0)
        self.assertEqual(float(result[0, 0, 4, 4, 5]), -3.0)

    def test_anisotropic_radius_mm_and_outside_support_zero(self):
        session = state(affine=np.diag([2.0, 1.0, 3.0, 1.0]))
        backend = self.backend(session, radius_mm=2.1, same_source_class=False)
        action = point_action(session)
        result = self.evaluate_fake_proposal(backend, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 0, 5, 4, 4]), 1.0)  # 2 mm
        self.assertEqual(float(result[0, 0, 6, 4, 4]), 0.0)  # 4 mm
        self.assertEqual(float(result[0, 0, 4, 6, 4]), 1.0)  # 2 mm
        self.assertEqual(float(result[0, 0, 4, 4, 5]), 0.0)  # 3 mm
        self.assertEqual(float(result[0, 0, 0, 0, 0]), 0.0)

    def test_full_affine_shear_metric_for_points(self):
        affine = np.eye(4)
        affine[0, 1] = 1.0
        session = state(affine=affine)
        backend = self.backend(session, radius_mm=1.1, same_source_class=False)
        result = self.evaluate_fake_proposal(backend, session, point_action(session),
                                             torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 0, 5, 3, 4]), 1.0)  # physical displacement (0,-1,0)
        self.assertEqual(float(result[0, 0, 5, 5, 4]), 0.0)  # physical displacement (2,1,0)

    def test_source_class_protection_uses_current_prediction_only(self):
        labels = np.zeros((9, 9, 9), np.uint8)
        labels[4, 4, 4] = 1
        labels[4, 4, 5] = 1
        labels[4, 5, 4] = 2
        session = state(labels)
        backend = self.backend(session, radius_mm=3.0, same_source_class=True)
        result = self.evaluate_fake_proposal(backend, session, point_action(session),
                                             torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 0, 4, 4, 5]), 1.0)
        self.assertEqual(float(result[0, 0, 4, 5, 4]), 0.0)  # NCR preserved
        self.assertEqual(float(result[0, 0, 3, 4, 4]), 0.0)  # BG preserved
        self.assertEqual(backend.last_constraints['point_source_classes'], [1])
        self.assertFalse(backend.last_constraints['gt_used'])

    def test_multiple_points_union_keeps_each_own_class_and_radius(self):
        labels = np.zeros((9, 9, 9), np.uint8)
        labels[3, 4, 4] = 1
        labels[6, 4, 4] = 2
        labels[3, 5, 4] = 2
        labels[6, 5, 4] = 1
        session = state(labels)
        point_action(session, target='TC', voxel=(3, 4, 4), point_id='a')
        point_action(session, target='TC', voxel=(6, 4, 4), point_id='b')
        action = StructuredAction('TC', 'include', 'roi', positive_point_ids=('a', 'b'))
        backend = self.backend(session, radius_mm=1.1)
        result = self.evaluate_fake_proposal(backend, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 1, 3, 4, 4]), 1.0)
        self.assertEqual(float(result[0, 1, 6, 4, 4]), 1.0)
        self.assertEqual(float(result[0, 1, 3, 5, 4]), 0.0)
        self.assertEqual(float(result[0, 1, 6, 5, 4]), 0.0)

    def test_box_physical_support_without_source_class_or_hidden_point(self):
        session = state(affine=np.diag([2.0, 1.0, 3.0, 1.0]))
        session.boxes['box'] = BoxPrompt('box', (3, 3, 4), (6, 6, 5), 'ET', True)
        action = StructuredAction('ET', 'include', 'roi', box_prompt=True, box_id='box')
        backend = self.backend(session, radius_mm=1.1, same_source_class=True)
        result = self.evaluate_fake_proposal(backend, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 2, 3, 3, 4]), 1.0)
        self.assertEqual(float(result[0, 2, 4, 6, 4]), 1.0)  # 1mm from rectangle
        self.assertEqual(float(result[0, 2, 6, 4, 4]), 0.0)  # 2mm from rectangle
        self.assertEqual(float(result[0, 2, 4, 4, 5]), 0.0)  # 3mm from slice plane
        self.assertEqual(session.points, {})
        self.assertEqual(backend.last_constraints['point_source_classes'], [])
        self.assertFalse(backend.last_constraints['box_source_class_filter'])

    def test_box_distance_handles_shear_perpendicular_to_slice(self):
        affine = np.eye(4)
        affine[0, 2] = 1.0
        session = state(affine=affine)
        session.boxes['box'] = BoxPrompt('box', (2, 2, 4), (6, 6, 5), 'WT', True)
        action = StructuredAction('WT', 'include', 'roi', box_prompt=True, box_id='box')
        backend = self.backend(session, radius_mm=1.1)
        result = self.evaluate_fake_proposal(backend, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(float(result[0, 0, 3, 3, 5]), 1.0)  # nearest point slides along rectangle
        self.assertEqual(float(result[0, 0, 3, 3, 6]), 0.0)

    def test_modality_by_head_and_missing_modality_rejection_before_forward(self):
        session = state()
        backend = self.backend(session, modality_by_head={'WT': 3, 'TC': 1, 'ET': 1})
        action = point_action(session)
        self.evaluate_fake_proposal(backend, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(backend.modality_index, 3)
        self.assertTrue(np.all(backend.raw_image == 4.0))
        session.modality_mask[3] = False
        with patch.object(NNInteractiveBackend, 'predict_delta') as parent:
            with self.assertRaisesRegex(ValueError, 'absent'):
                backend.predict_delta(torch.zeros((1, 21, 9, 9, 9)), session=session, action=action,
                                      roi=SpatialROI((0, 0, 0), (9, 9, 9)))
            parent.assert_not_called()
        default = self.backend(session)
        self.evaluate_fake_proposal(default, session, action, torch.ones((1, 3, 9, 9, 9)))
        self.assertEqual(default.modality_index, 1)
        self.assertTrue(np.all(default.raw_image == 2.0))

    def test_actual_parent_receives_full_image_global_box_only_and_returns_roi(self):
        session = state(shape=(17, 17, 17))
        session.boxes['box'] = BoxPrompt('box', (7, 7, 8), (10, 10, 9), 'ET', True)
        action = StructuredAction('ET', 'include', 'roi', box_prompt=True, box_id='box')
        roi = SpatialROI((4, 4, 4), (13, 13, 13))
        engine = CapturingEngine()
        backend = self.backend(session, engine=engine, radius_mm=2.0)
        result = backend.predict_delta(torch.zeros((1, 21, *roi.shape)), session=session, action=action, roi=roi)
        self.assertEqual(tuple(engine.image.shape), (1, 17, 17, 17))
        self.assertEqual(engine.calls, [('box', [[7, 10], [7, 10], [8, 9]], True)])
        self.assertEqual(engine.predictions, 1)
        self.assertEqual(tuple(result.shape), (1, 3, 9, 9, 9))
        self.assertEqual(int(torch.count_nonzero(result[:, :2])), 0)
        self.assertEqual(session.points, {})


if __name__ == '__main__':
    unittest.main()
