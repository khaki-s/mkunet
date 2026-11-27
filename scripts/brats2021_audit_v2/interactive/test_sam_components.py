"""Regression tests for native 3D SAM prompts and exact gain labels."""
import importlib
import unittest

import numpy as np

import foundation_prompt

MODULE = foundation_prompt.name + '.foundation_prompt_v2.sam_components'
components = importlib.import_module(MODULE + '.sam_prompt_components')
pipeline = importlib.import_module(MODULE + '.sam_component_pipeline')


class ComponentPromptTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.zeros((24, 25, 17), np.uint8)
        self.baseline[5:17, 5:17, 5:14] = 1
        self.baseline[8:12, 8:12, 6:11] = 2
        self.baseline[9:11, 9:11, 7:10] = 3
        binary = np.stack([self.baseline > h for h in range(3)])
        self.masks = np.stack([binary, binary])
        self.masks[0, 0, 17:20, 7:14, 7:12] = True
        self.masks[0, 0, 5:8, 7:12, 7:12] = False
        self.masks[0, 1, 10:16, 5:7, 7:12] = True
        self.masks[0, 2, 9:11, 9:11, 7:10] = False
        self.mri = np.random.default_rng(20).uniform(.001, 1, (4, *self.baseline.shape)).astype(np.float32)
        self.affine = np.diag([.9, 1.1, 2.5, 1.])
        self.data = components.component_features(self.mri, self.baseline, self.masks, self.affine)

    def test_original_grid_3d_components_cover_both_polarities(self):
        proposals = list(components.propose(self.data, [.7] * 3))
        self.assertGreater(len(proposals), 3)
        self.assertEqual(set(bool(p[0]['positive']) for p in proposals), {False, True})
        for prompt, locations, _ in proposals:
            group = prompt['group']
            head, positive = components.GROUPS[group]
            center = np.ravel_multi_index(prompt['voxel'], self.baseline.shape)
            self.assertIn(center, locations)
            self.assertEqual(self.masks[0, head].ravel()[center], positive)
            self.assertNotEqual(self.baseline.ravel()[center] > head, positive)
            self.assertGreater(prompt['component_slices'], 1)
            self.assertEqual(prompt['box_stop'][2] - prompt['box_start'][2], 1)
            for l, c, h, n in zip(prompt['box_start'], prompt['voxel'],
                                  prompt['box_stop'], self.baseline.shape):
                self.assertTrue(0 <= l <= c < h <= n)

    def test_low_probability_still_generates_valid_half_open_box_for_feedback(self):
        for prompt, _, _ in components.propose(self.data, [.02] * 3):
            for l, c, h in zip(prompt['box_start'], prompt['voxel'], prompt['box_stop']):
                self.assertTrue(l <= c < h)

    def test_no_sam_disagreement_cannot_call_corrector(self):
        raw = np.stack([self.baseline > i for i in range(3)])
        data = components.component_features(self.mri, self.baseline, np.stack([raw, raw]), self.affine)
        self.assertEqual(list(pipeline.action_options(data, None, self.baseline, [1.] * 3)), [])

    def test_connected_profile_expands_support_but_anchors_are_sam_voxels(self):
        data = components.component_features(self.mri, self.baseline, self.masks, self.affine, connected=True)
        self.assertGreater(len(data['index_0']), len(self.data['index_0']))
        candidates = list(components.propose(data, [.72] * 3))
        self.assertTrue(candidates)
        for prompt, _, _ in candidates:
            h, sign = components.GROUPS[prompt['group']]
            source = tuple(prompt['voxel'])
            self.assertNotEqual(self.baseline[source] > h, sign)
            self.assertTrue(np.any(self.masks[(slice(None), h, *source)] == sign))

    def test_progressive_edit_uses_one_supported_connected_response(self):
        shape = (6, 6, 6)
        large = [(x, y, 1) for x in (1, 2, 3) for y in (1, 2)]
        small = [(x, y, 5) for x in (1, 2) for y in (1, 2)]
        picked = np.ravel_multi_index(np.array(large + small).T, shape)
        probability = np.array([.8] * len(large) + [.99] * len(small))
        np.testing.assert_array_equal(pipeline.connected_edit(picked, probability, shape, .6),
                                      np.arange(len(large)))

    def test_acceptance_requires_real_nninteractive_response_and_hierarchy(self):
        def unchanged(initial, prompt, history):
            return initial, {}

        def directional(initial, prompt, history):
            output = initial.copy()
            head, positive = components.GROUPS[prompt['group']]
            region = self.masks[0, head]
            eligible = (initial == head) if positive else (initial == head + 1)
            output[eligible & (region == positive)] = head + int(positive)
            return output, {}

        equal = list(pipeline.action_options(self.data, unchanged, self.baseline, [.8] * 3))
        self.assertTrue(equal)
        self.assertTrue(all(not options for _, _, options in equal))
        options = list(pipeline.action_options(self.data, directional, self.baseline, [.8] * 3))
        for prompt, _, candidates in options:
            head, positive = components.GROUPS[prompt['group']]
            for _, picked, vector in candidates:
                result = self.baseline.copy()
                result.ravel()[picked] = head + int(positive)
                for neighbor in set(range(3)) - {head}:
                    np.testing.assert_array_equal(self.baseline > neighbor, result > neighbor)
                self.assertGreater(vector[0], 0.)

    def test_exact_gt_dice_and_stop(self):
        for counts, correct, number, direction in (((10, 9, 8), 2, 3, True),
                                                    ((10, 9, 8), 2, 3, False)):
            a, b, tp = counts
            new_a = a + (number if direction else -number)
            new_tp = tp + (correct if direction else correct - number)
            assert_value = 100 * (2 * new_tp / (new_a + b) - 2 * tp / (a + b))
            self.assertAlmostEqual(pipeline._dice_change(counts, correct, number, direction), assert_value)
        candidates = [{'vector': [-.2]}, {'vector': [.05]}]
        self.assertIsNone(pipeline._pick(candidates, None, 0., None))
        self.assertIsNone(pipeline._pick(candidates, None, 0., .06))
        self.assertEqual(pipeline._pick(candidates, None, 0., .01), 1)

    def test_independent_route_does_not_execute_disabled_polarity(self):
        allow = {'model': None, 'margin': .01, 'weight': 0.}
        stop = {**allow, 'margin': None}
        positive = {'vector': [1.] + [0.] * 10 + [1.]}
        negative = {'vector': [.02] + [0.] * 11}
        selected, _ = pipeline._direction_choice([positive, negative], [stop, allow])
        self.assertIs(selected, negative)
        selected, _ = pipeline._direction_choice([positive, negative], [stop, stop])
        self.assertIsNone(selected)


if __name__ == '__main__':
    unittest.main()
