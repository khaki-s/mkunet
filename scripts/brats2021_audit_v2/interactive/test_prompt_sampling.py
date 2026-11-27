#!/usr/bin/env python
"""CPU checks for finite, deterministic GT-oracle point/box annotations."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[3] / 'nnunetv2/utilities/brats2021_audit_v2/interactive/prompt_sampling.py'
SPEC = importlib.util.spec_from_file_location('brats_prompt_sampling_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
select_prompt = MODULE.select_prompt


class PromptSamplingTests(unittest.TestCase):
    def prompt(self, pred, gt, **kwargs):
        return select_prompt(pred, gt, np.eye(4), target='joint', kind='point', seed=7, **kwargs)

    def test_all_twelve_transitions(self):
        for old in range(4):
            for new in range(4):
                if old == new:
                    continue
                with self.subTest(old=old, new=new):
                    pred = np.full((3, 4, 5), old, np.uint8)
                    gt = np.full_like(pred, new)
                    item = self.prompt(pred, gt)
                    head = ('WT', 'TC', 'ET').index(item['head'])
                    state = np.array([old > h for h in range(3)])
                    if item['positive']:
                        state[:head + 1] = True
                    else:
                        state[head:] = False
                    self.assertEqual(int(state.sum()), new)
                    self.assertEqual(item['pred_class'], old)
                    self.assertEqual(item['gt_class'], new)
                    self.assertEqual(bool(gt[tuple(item['voxel'])] > head), item['positive'])

    def test_no_errors_and_target_specific_stopping(self):
        gt = np.zeros((9, 9, 9), np.uint8)
        self.assertIsNone(self.prompt(gt, gt))
        pred = gt.copy()
        pred[4, 4, 4] = 1
        self.assertIsNone(select_prompt(pred, gt, np.eye(4), target='ET', kind='point', seed=0))
        self.assertIsNotNone(select_prompt(pred, gt, np.eye(4), target='ED', kind='point', seed=0))

    def test_largest_component_interior_and_determinism(self):
        pred = np.zeros((13, 13, 13), np.uint8)
        gt = pred.copy()
        gt[2:7, 2:7, 2:7] = 3
        gt[10:12, 10:12, 10:12] = 3
        p = self.prompt(pred, gt)
        self.assertEqual(p['voxel'], [4, 4, 4])
        self.assertEqual(p['error_component_voxels'], 125)
        self.assertEqual(p, self.prompt(pred, gt))

    def test_reselect_after_current_prediction_changes(self):
        pred = np.zeros((13, 13, 13), np.uint8)
        gt = pred.copy()
        gt[2:7, 2:7, 2:7] = 3
        gt[10:12, 10:12, 10:12] = 3
        first = self.prompt(pred, gt)
        pred[2:7, 2:7, 2:7] = 3
        second = self.prompt(pred, gt)
        self.assertNotEqual(first['voxel'], second['voxel'])
        self.assertTrue(all(v >= 10 for v in second['voxel']))

    def test_box_geometry_and_no_hidden_point(self):
        pred = np.zeros((11, 12, 13), np.uint8)
        gt = pred.copy()
        gt[1:10, 1:11, 2:12] = 3
        for axis in range(3):
            item = select_prompt(pred, gt, np.eye(4), target='ET', kind='box', seed=2,
                                 box_axis=axis, box_max_extent=4)
            start, stop = item['box_start'], item['box_stop']
            self.assertEqual(stop[axis] - start[axis], 1)
            self.assertNotIn('voxel', item)
            for a, b, anchor, dim in zip(start, stop, item['anchor'], pred.shape):
                self.assertTrue(0 <= a <= anchor < b <= dim)
                self.assertLessEqual(b-a, 4)
            self.assertTrue(item['box_extent_clipped'])
            self.assertEqual(item, select_prompt(pred, gt, np.eye(4), target='ET', kind='box', seed=2,
                                                box_axis=axis, box_max_extent=4))

    def test_negative_box_never_covers_gt_positive_head(self):
        # A false-positive ET ring encloses true ET. Its naive bounding box
        # would wrongly claim the true ET center as a negative annotation.
        pred = np.zeros((11, 11, 3), np.uint8)
        gt = pred.copy()
        pred[2:9, 2:9, 1] = 3
        gt[4:7, 4:7, 1] = 3
        pred[4:7, 4:7, 1] = 3
        item = select_prompt(pred, gt, np.eye(4), target='ET', kind='box', seed=4,
                             box_padding=2)
        self.assertFalse(item['positive'])
        slices = tuple(slice(a, b) for a, b in zip(item['box_start'], item['box_stop']))
        head = ('WT', 'TC', 'ET').index(item['head'])
        self.assertFalse((gt[slices] > head).any())
        self.assertTrue(item['negative_box_gt_pure'])
        self.assertTrue(item['box_negative_shrunk'])

    def test_degenerate_negative_box_and_json_only(self):
        gt = np.full((7, 7, 3), 3, np.uint8)
        gt[3, 3, 1] = 0
        pred = np.full_like(gt, 3)
        item = select_prompt(pred, gt, np.eye(4), target='ET', kind='box', seed=0)
        self.assertEqual(item['box_area_voxels'], 1)
        self.assertTrue(item['box_degenerate'])
        self.assertEqual(json.loads(json.dumps(item)), item)
        self.assertFalse(any(isinstance(value, np.ndarray) for value in item.values()))

    def test_pair_selects_one_error_and_shares_box_anchor_head_and_polarity(self):
        for old, new in ((0, 3), (3, 2), (1, 0), (1, 2)):
            with self.subTest(old=old, new=new):
                pred = np.full((7, 8, 9), old, np.uint8)
                gt = np.full_like(pred, new)
                with patch.object(MODULE, '_largest_error', wraps=MODULE._largest_error) as choose:
                    pair = select_prompt(pred, gt, np.eye(4), target='joint', kind='point_box', seed=9)
                    self.assertEqual(choose.call_count, 1)
                point = select_prompt(pred, gt, np.eye(4), target='joint', kind='point', seed=9)
                box = select_prompt(pred, gt, np.eye(4), target='joint', kind='box', seed=9)
                self.assertEqual(pair['voxel'], point['voxel'])
                self.assertEqual(pair['voxel'], pair['anchor'])
                for key, value in box.items():
                    if key != 'kind':
                        self.assertEqual(pair[key], value)
                self.assertEqual(pair['head'], point['head'])
                self.assertEqual(pair['positive'], point['positive'])
                self.assertTrue(all(a <= v < b for a, v, b in
                                    zip(pair['box_start'], pair['voxel'], pair['box_stop'])))
                self.assertEqual(pair['annotation_count'], 2)
                self.assertEqual(json.loads(json.dumps(pair)), pair)
                self.assertFalse(any(isinstance(value, np.ndarray) for value in pair.values()))

    def test_pair_no_error_stops_without_annotations(self):
        labels = np.zeros((5, 5, 5), np.uint8)
        self.assertIsNone(select_prompt(labels, labels, np.eye(4),
                                        target='joint', kind='point_box', seed=0))

    def test_invalid_inputs(self):
        x = np.zeros((3, 3, 3), np.uint8)
        for change in ({'target': 'NET'}, {'kind': 'scribble'}, {'seed': -1},
                       {'box_axis': 3}, {'box_padding': -1}, {'box_max_extent': 0}):
            args = dict(target='ET', kind='point', seed=0)
            args.update(change)
            with self.assertRaises(ValueError):
                select_prompt(x, x, np.eye(4), **args)
        with self.assertRaises(ValueError):
            self.prompt(x.astype(float), x)
        with self.assertRaises(ValueError):
            self.prompt(x, np.full_like(x, 4))
        with self.assertRaises(ValueError):
            select_prompt(x, x, np.zeros((4,4)), target='ET', kind='point', seed=0)


if __name__ == '__main__':
    unittest.main()
