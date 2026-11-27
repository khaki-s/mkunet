"""CPU-only protocol tests with an injected SAM API double; no claimed FM validation."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'nnunetv2/utilities/brats2021_audit_v2/interactive/foundation_prompt_v2/sam3_candidates.py'
spec = importlib.util.spec_from_file_location('candidates_under_test', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeSAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.inst_interactive_predictor = SimpleNamespace(model=SimpleNamespace(
            sam_prompt_encoder=SimpleNamespace(mask_input_size=(12, 14))))
        self.calls = []

    def predict_inst(self, state, **kwargs):
        self.calls.append(kwargs)
        mask = kwargs['mask_input'][0] > 0
        updated = mask.copy()
        updated[8:10, 5:7] = True
        updated[3:5, 5:7] = False
        return np.stack((mask, updated)), np.array([.2, .9]), None


class FakeProcessor:
    def set_image(self, image):
        self.last_size = image.size
        return {'backbone_out': {'vision_features': torch.ones(1, 256, 2, 2)}}


class CandidatesTest(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((12, 14, 3), dtype=np.uint8)
        self.labels[3:8, 4:9, 1] = 3
        self.mri = np.broadcast_to(np.arange(12, dtype=np.float32)[None, :, None, None],
                                   (4, 12, 14, 3)).copy()

    def create(self, **kwargs):
        model, processor = FakeSAM(), FakeProcessor()
        return module.Sam3PromptCandidates(model=model, processor=processor, device='cpu',
                                           min_component_voxels=2, **kwargs), model, processor

    def test_gt_free_fixed_slots_and_coordinate_mapping(self):
        provider, model, processor = self.create()
        result = provider.generate(self.mri, self.labels)
        self.assertEqual(result['context'].shape, (2620,))
        self.assertEqual(result['metadata']['sam_predict_calls'], 12)
        self.assertTrue(all(item is not None for item in result['actions']))
        self.assertEqual(processor.last_size, (14, 12))
        slots = result['context'][1036:].reshape(6, 264)
        np.testing.assert_array_equal(slots[:, 256], np.ones(6))
        self.assertTrue(np.all(slots[:, 257:260] >= 0) and np.all(slots[:, 257:260] <= 1))
        self.assertFalse(model.parameter.requires_grad)
        self.assertFalse(model.training)
        for index, prompt in enumerate(result['actions']):
            xyz = tuple(prompt['voxel'])
            self.assertEqual(prompt['positive'], index % 2 == 0)
            self.assertNotEqual(bool(self.labels[xyz] > index // 2), prompt['positive'])
            self.assertFalse(prompt['gt_geometry_used'])
        # Center of native mask is X=5,Y=6 -> SAM pixel x=6,y=5.
        np.testing.assert_array_equal(model.calls[0]['point_coords'], [[6, 5]])

    def test_box_only_budget_and_negative_geometry(self):
        provider, _, _ = self.create(kind='box', targets=('ET',), box_max_extent=5)
        actions = provider.generate(self.mri, self.labels)['actions']
        self.assertTrue(all(action is None for action in actions[:4]))
        for action in actions[4:]:
            self.assertNotIn('voxel', action)
            lo, hi, anchor = map(np.asarray, (action['box_start'], action['box_stop'], action['anchor']))
            self.assertTrue(np.all(lo <= anchor) and np.all(anchor < hi))
            self.assertEqual(hi[2] - lo[2], 1)
            self.assertTrue(np.all(hi[:2] - lo[:2] <= 5))

    def test_history_and_absent_regions(self):
        provider, _, _ = self.create(repeat_radius_mm=100.)
        initial = provider.generate(self.mri, self.labels)
        repeated = provider.generate(self.mri, self.labels, history=initial['actions'])
        self.assertEqual(repeated['actions'], [None] * 6)
        self.assertTrue(np.all(repeated['context'][1036:] == 0))
        absent = provider.generate(self.mri, np.zeros_like(self.labels))
        self.assertEqual(absent['actions'], [None] * 6)
        self.assertEqual(absent['metadata']['missing_region_limitation'], ['WT', 'TC', 'ET'])

    def test_missing_modalities_and_validation(self):
        provider, model, _ = self.create()
        result = provider.generate(self.mri, self.labels, [1, 0, 0, 0])
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(np.all(result['context'][256:1024] == 0))
        with self.assertRaises(ValueError):
            provider.generate(self.mri, self.labels, [0, 0, 0, 0])
        with self.assertRaises(ValueError):
            provider.generate(self.mri, self.labels.astype(float))
        check = module.validate_source('/nonexistent_sam', '/nonexistent_ckpt', check_imports=False)
        self.assertFalse(check['ready_for_import'])
        self.assertFalse(check['weight_load_tested'])

    def test_head_specific_modalities_are_enforced(self):
        policy = {'WT': [3], 'TC': [1], 'ET': [1]}
        provider, model, _ = self.create(candidate_modalities_by_head=policy)
        result = provider.generate(self.mri, self.labels)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual([a['sam_modality'] for a in result['actions']],
                         ['FLAIR', 'FLAIR', 'T1ce', 'T1ce', 'T1ce', 'T1ce'])
        unavailable = provider.generate(self.mri, self.labels, [1, 0, 1, 0])
        self.assertEqual(unavailable['actions'], [None]*6)
        with self.assertRaises(ValueError):
            self.create(candidate_modalities_by_head={'WT': [3], 'TC': [1], 'ET': []})


if __name__ == '__main__':
    unittest.main()
