"""Regressions for legal region edits and patient-wise selection."""
import unittest

import numpy as np

from .dataset import slice_schedule
from .policy import Policy, Rule, Window, candidates
from .select_val80 import nested


class RefinerPolicyTests(unittest.TestCase):
    def test_unsigned_edits_preserve_inactive_regions(self):
        labels = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        records = [{"head": h, "z": z} for h in range(3) for z in range(4)]
        scores = np.full((12, 1, 1), .10, dtype=np.float16)
        query = Window.build(labels, records, scores, np.full_like(scores, .50))

        tc = Policy((Rule(1, .50, None),))
        joint = Policy((Rule(1, .50, None), Rule(2, .50, None)), joint=True)
        np.testing.assert_array_equal(query.apply_volume(labels, tc), [[[0, 1, 1, 3]]])
        np.testing.assert_array_equal(query.apply_volume(labels, joint), [[[0, 1, 1, 1]]])
        np.testing.assert_array_equal(query.apply_volume(labels, Policy()), labels)

    def test_empty_inner_region_can_be_scheduled_from_parent_support(self):
        labels = np.zeros((2, 2, 3), dtype=np.uint8)
        labels[0, 0, 1] = 2
        probabilities = np.zeros((3, *labels.shape), dtype=np.float32)
        probabilities[2, 0, 0, 1] = .30
        probabilities[2, 0, 0, 2] = .30
        self.assertEqual(slice_schedule(labels, probabilities, (2,), 0), [(2, 1)])

    def test_nested_policy_only_activates_positive_units(self):
        options = candidates((2,))
        values = np.zeros((80, len(options), 3))
        self.assertEqual(nested(values, options)[1], 0)
        values[:, 1, 2] = .35
        stats, selected, gains, choices = nested(values, options)
        self.assertTrue(stats[2]["pass"])
        self.assertFalse(stats[0]["pass"])
        self.assertEqual(options[selected].heads, (2,))
        self.assertEqual(sorted(i for f in choices for i in f["heldout_indices"]), list(range(80)))
        np.testing.assert_allclose(gains[:, 2], .35)


if __name__ == "__main__":
    unittest.main()
