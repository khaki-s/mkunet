"""Synthetic CPU checks for paired denominators, cohort coverage and empty regions."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[3] / 'nnunetv2/utilities/brats2021_audit_v2/interactive/interaction_metrics.py'
SPEC = importlib.util.spec_from_file_location('interaction_metrics_under_test', MODULE)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)
region_metrics, aggregate_results = METRICS.region_metrics, METRICS.aggregate_results


def volume(label=None):
    a = np.zeros((2, 2, 2), np.uint8)
    if label is not None:
        a[0, 0, 0] = label
    return a


def record(case, gt, before, after, *, target='NCR', kind='point', steps=1):
    baseline = region_metrics(before, gt, 2.0)
    corrected = region_metrics(after, gt, 2.0)
    states = {}
    for budget in (0, 1, 3, 5):
        actual = min(budget, steps)
        states[str(budget)] = {
            'metrics': baseline if budget == 0 else corrected,
            'actual_prompts': actual, 'actual_tool_calls': actual,
            'cumulative_interaction_seconds': actual * 2.0,
            'cumulative_selection_seconds': actual * .2,
            'cumulative_inference_seconds': actual * 1.5,
            'carried_forward': budget > steps,
        }
    return {'case_id': case, 'target': target, 'status': 'complete',
            'prompt_kind': kind, 'baseline': baseline, 'budgets': states}


def row(summary, budget=5, region='NCR'):
    return next(r for r in summary['groups'] if r['budget'] == budget and r['region'] == region)


class InteractionMetricsTests(unittest.TestCase):
    def test_internal_regions_and_physical_volumes(self):
        gt = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8).reshape(2, 2, 2)
        pred = gt.copy()
        pred[0, 1, 1] = 1  # ET -> ED: WT remains correct, TC and ET lose a voxel.
        values = region_metrics(pred, gt, 1.5)
        self.assertEqual(values['WT']['dice'], 1.0)
        self.assertEqual(values['TC']['fn'], 1)
        self.assertEqual(values['ET']['fn_mm3'], 1.5)
        self.assertEqual(values['ED']['fp_mm3'], 1.5)
        self.assertEqual(values['NCR']['dice'], 1.0)

    def test_empty_gt_is_not_converted_to_perfect_dice(self):
        empty = region_metrics(volume(), volume())
        self.assertTrue(all(r['dice'] is None for r in empty.values()))
        fp = region_metrics(volume(2), volume(), 3.)
        self.assertEqual(fp['NCR']['dice'], 0.)
        self.assertEqual(fp['NCR']['fp_mm3'], 3.)
        for invalid in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                region_metrics(volume(), volume(), invalid)
        with self.assertRaises(ValueError):
            region_metrics(volume(4), volume())

    def test_paired_deltas_and_failed_missing_coverage(self):
        records = [
            record('a', volume(2), volume(), volume(2)),  # Dice 0 -> 1.
            record('b', volume(), volume(2), volume()),  # Dice 0 -> None; FP cured.
            record('c', volume(), volume(), volume(2)),  # Dice None -> 0; new FP.
            record('d', volume(2), volume(2), volume()),  # Dice 1 -> 0.
            {'case_id': 'e', 'target': 'NCR', 'status': 'failed', 'prompt_kind': 'point',
             'baseline': region_metrics(volume(2), volume(2))},
        ]
        result = aggregate_results(records, list('abcdef'), ['NCR'])
        coverage = result['coverage']
        self.assertEqual((coverage['complete_pairs'], coverage['failed_pairs'], coverage['missing_pairs']), (4, 1, 1))
        self.assertFalse(coverage['full_cohort_complete'])
        self.assertEqual(coverage['by_target']['NCR']['missing_case_ids'], ['f'])
        r = row(result)
        self.assertEqual(r['dice_valid_n'], 3)
        self.assertEqual(r['baseline_valid_n'], 4)
        self.assertEqual(r['paired_n'], 2)
        self.assertEqual(r['mean_delta_pp'], 0.)
        self.assertEqual((r['improved_n'], r['degraded_n'], r['unchanged_n']), (1, 1, 0))
        self.assertEqual(r['improved_fraction'], .5)
        self.assertEqual(r['gt_positive_n'], 2)
        self.assertEqual(r['gt_positive_mean_delta_pp'], 0.)
        self.assertEqual(r['gt_empty_n'], 2)
        self.assertEqual(r['gt_empty_fp_positive_n'], 1)
        self.assertEqual(r['gt_empty_fp_prevalence'], .5)
        self.assertEqual(r['dice_undefined_to_defined_n'], 1)
        self.assertEqual(r['dice_defined_to_undefined_n'], 1)
        json.dumps(result, allow_nan=False)

    def test_early_stop_carries_scores_not_prompt_counts(self):
        r = record('one', volume(3), volume(), volume(3), target='ET', steps=1)
        result = aggregate_results([r], ['one'], ['ET'])
        self.assertEqual(row(result, 1, 'ET')['dice_mean'], row(result, 5, 'ET')['dice_mean'])
        t = next(r for r in result['timing_groups'] if r['budget'] == 5)
        self.assertEqual(t['actual_prompts_mean'], 1.)
        self.assertEqual(t['cumulative_interaction_seconds_mean'], 2.)
        self.assertEqual(t['carried_forward_n'], 1)

    def test_wt_tc_retreat_is_case_paired(self):
        # Correct ET initially; a bad correction removes the complete tumor.
        r = record('one', volume(3), volume(3), volume(), target='ET')
        result = aggregate_results([r], ['one'], ['ET'])
        timing = next(r for r in result['timing_groups'] if r['budget'] == 1)
        self.assertEqual(timing['WT_retreat_n'], 1)
        self.assertEqual(timing['TC_retreat_fraction'], 1.)
        self.assertEqual(timing['WT_or_TC_retreat_fraction'], 1.)

    def test_duplicate_case_and_mixed_prompt_kinds_rejected(self):
        a = record('a', volume(2), volume(), volume(2))
        with self.assertRaises(ValueError):
            aggregate_results([a], ['a', 'a'], ['NCR'])
        with self.assertRaises(ValueError):
            aggregate_results([a, a], ['a'], ['NCR'])
        with self.assertRaises(ValueError):
            aggregate_results([a], ['a'], ['NCR', 'NCR'])
        with self.assertRaises(ValueError):
            aggregate_results([a], ['other'], ['NCR'])
        b = record('b', volume(2), volume(), volume(2), kind='box')
        with self.assertRaises(ValueError):
            aggregate_results([a, b], ['a', 'b'], ['NCR'])

    def test_missing_budget_and_inconsistent_truth_rejected(self):
        original = record('a', volume(2), volume(), volume(2))
        broken = copy.deepcopy(original)
        del broken['budgets']['3']
        with self.assertRaises(ValueError):
            aggregate_results([broken], ['a'], ['NCR'])
        broken = copy.deepcopy(original)
        broken['budgets']['5']['metrics'] = region_metrics(volume(), volume())
        with self.assertRaises(ValueError):
            aggregate_results([broken], ['a'], ['NCR'])
        broken = copy.deepcopy(original)
        broken['budgets']['1']['actual_prompts'] = 2
        with self.assertRaises(ValueError):
            aggregate_results([broken], ['a'], ['NCR'])

    def test_all_failed_cohort_has_no_fake_scores(self):
        result = aggregate_results([
            {'case_id': 'a', 'target': 'joint', 'status': 'failed', 'prompt_kind': 'point'}
        ], ['a', 'b'], ['joint'])
        self.assertEqual(result['coverage']['failed_pairs'], 1)
        self.assertEqual(result['coverage']['missing_pairs'], 1)
        self.assertTrue(all(r['dice_mean'] is None and r['mean_delta_pp'] is None for r in result['groups']))
        self.assertTrue(all(r['improved_fraction'] is None for r in result['groups']))
        json.dumps(result, allow_nan=False)

    def paired_record(self, steps=5):
        item = record('one', volume(3), volume(), volume(3), target='ET', kind='point_box', steps=steps)
        item['budget_unit'] = 'rounds'
        for state in item['budgets'].values():
            rounds = state['actual_prompts']
            state.update(actual_prompts=2 * rounds, actual_rounds=rounds,
                         actual_points=rounds, actual_boxes=rounds, budget_unit='rounds')
        return item

    def test_pair_round_budget_reports_ten_annotations_separately(self):
        result = aggregate_results([self.paired_record()], ['one'], ['ET'])
        self.assertEqual(result['budget_unit'], 'rounds')
        self.assertEqual(result['annotations_per_round'], 2)
        timing = next(r for r in result['timing_groups'] if r['budget'] == 5)
        self.assertEqual(timing['actual_prompts_mean'], 10.)
        self.assertEqual(timing['actual_points_mean'], 5.)
        self.assertEqual(timing['actual_boxes_mean'], 5.)
        self.assertEqual(timing['actual_rounds_mean'], 5.)
        self.assertEqual(timing['actual_tool_calls_mean'], 5.)
        json.dumps(result, allow_nan=False)

    def test_pair_early_stop_and_inconsistent_counts_rejected(self):
        item = self.paired_record(steps=1)
        result = aggregate_results([item], ['one'], ['ET'])
        timing = next(r for r in result['timing_groups'] if r['budget'] == 5)
        self.assertEqual(timing['actual_prompts_mean'], 2.)
        self.assertEqual(timing['actual_rounds_mean'], 1.)
        for field, wrong in [('actual_prompts', 1), ('actual_points', 2),
                             ('actual_boxes', 0), ('actual_rounds', 2),
                             ('actual_tool_calls', 2), ('budget_unit', 'annotations')]:
            with self.subTest(field=field):
                broken = copy.deepcopy(item)
                broken['budgets']['1'][field] = wrong
                with self.assertRaises(ValueError):
                    aggregate_results([broken], ['one'], ['ET'])
        del item['budgets']['1']['actual_boxes']
        with self.assertRaises(ValueError):
            aggregate_results([item], ['one'], ['ET'])

    def test_pair_cannot_mix_with_point_and_legacy_counts_remain_readable(self):
        paired = self.paired_record()
        old = record('old', volume(3), volume(), volume(3), target='ET')
        with self.assertRaises(ValueError):
            aggregate_results([paired, old], ['one', 'old'], ['ET'])
        result = aggregate_results([old], ['old'], ['ET'])
        self.assertEqual(result['budget_unit'], 'annotations')
        timing = next(r for r in result['timing_groups'] if r['budget'] == 5)
        self.assertEqual(timing['actual_points_mean'], 1.)
        self.assertEqual(timing['actual_boxes_mean'], 0.)
        self.assertEqual(timing['actual_rounds_mean'], 1.)

    def test_failed_pair_record_keeps_round_semantics_without_fake_counts(self):
        result = aggregate_results([
            {'case_id': 'one', 'target': 'joint', 'status': 'failed', 'prompt_kind': 'point_box'}
        ], ['one'], ['joint'])
        self.assertEqual(result['budget_unit'], 'rounds')
        self.assertEqual(result['coverage']['failed_pairs'], 1)
        self.assertTrue(all(t['actual_prompts_mean'] is None for t in result['timing_groups']))
        self.assertTrue(all(t['actual_rounds_mean'] is None for t in result['timing_groups']))


if __name__ == '__main__':
    unittest.main()
