"""Regressions for the 3D residual refiner: prior projection, model wiring, policy, selection."""
import unittest

import numpy as np
import torch

from . import data3d as D
from .model3d import build_model, regions_from_softmax_t
from .policy3d import Policy, Rule, Window, candidates
from .select3d import nested
from .sam_branch import Adapter2D, ambiguity, assemble_context, select_planes


class PriorTests(unittest.TestCase):
    def test_nested_projection_is_legal_and_matches_regions(self):
        prob = np.array([0.8, 0.9, 0.2]).reshape(3, 1, 1, 1).astype(np.float32)
        q4 = D.nested_projection(prob)
        np.testing.assert_allclose(q4.sum(0), 1.0, atol=1e-6)
        region = D.regions_from_softmax(q4)
        # ET<=TC<=WT and r equals the clamped nested prior.
        self.assertLessEqual(region[2, 0, 0, 0], region[1, 0, 0, 0] + 1e-6)
        self.assertLessEqual(region[1, 0, 0, 0], region[0, 0, 0, 0] + 1e-6)
        np.testing.assert_allclose(region[0, 0, 0, 0], 0.8, atol=1e-6)

    def test_zero_residual_reproduces_baseline(self):
        prob = np.random.default_rng(0).uniform(0, 1, (3, 32, 32, 32)).astype(np.float32)
        prior = torch.from_numpy(D.prior_logits(prob))[None]
        model = build_model().eval()
        features = torch.zeros((1, D.assemble_input(
            np.zeros((4, 32, 32, 32), np.float32), prob, np.zeros((32, 32, 32), np.int64), [1, 1, 1, 1]
        ).shape[0], 32, 32, 32))
        with torch.no_grad():
            logits = model(features, prior)
        region = regions_from_softmax_t(torch.softmax(logits, 1))[0].numpy()
        # Zero residual reproduces the baseline up to the documented 1e-3 prior clamp.
        np.testing.assert_allclose(region, D.nested_projection_region(prob), atol=5e-3)


class PolicyTests(unittest.TestCase):
    def _window(self, base, scores):
        roi = np.ones(base.shape, bool)
        prior = np.full((3, *base.shape), 0.5, np.float32)
        return Window.build(base, roi, prior, {"R": scores})

    def test_delete_and_add_preserve_inactive_regions(self):
        base = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        scores = np.zeros((3, *base.shape), np.float32)
        window = self._window(base, scores)
        tc = Policy("R", (Rule(1, 0.5, None),))
        np.testing.assert_array_equal(window.apply(tc), [[[0, 1, 1, 3]]])
        joint = Policy("R", (Rule(1, 0.5, None), Rule(2, 0.5, None)), joint=True)
        np.testing.assert_array_equal(window.apply(joint), [[[0, 1, 1, 1]]])
        np.testing.assert_array_equal(window.apply(Policy("P")), base)

    def test_candidate_arms_are_restricted(self):
        options = candidates((2,), ("P", "R"))
        self.assertTrue(all(p.arm in ("P", "R") for p in options))
        self.assertTrue(any(p.arm == "R" for p in options[1:]))


class SelectionTests(unittest.TestCase):
    def test_only_positive_region_activates(self):
        options = candidates((2,), ("P", "R"))
        values = np.zeros((80, len(options), 3))
        self.assertEqual(nested(values, options, ("P", "R"))[1], 0)
        values[:, 1, 2] = 0.3
        stats, final, gains, choices = nested(values, options, ("P", "R"))
        self.assertTrue(stats[2]["pass"])
        self.assertFalse(stats[0]["pass"])
        self.assertEqual(sorted(i for f in choices for i in f["heldout_indices"]), list(range(80)))


class SamBranchTests(unittest.TestCase):
    def test_context_has_25_channels(self):
        shape = (8, 8, 5)
        region = np.random.default_rng(1).uniform(0, 1, (3, *shape)).astype(np.float32)
        prior = region.copy()
        prob = region.copy()
        base = np.zeros(shape, np.int64)
        mri = np.zeros((4, *shape), np.float32)
        context = assemble_context(np.zeros((8, 8), np.float32), 0, 2, region, prior, prob,
                                   base, mri, [1, 1, 0, 0], enable_sam=True)
        self.assertEqual(context.shape, (25, 8, 8))

    def test_adapter_is_zero_initialised(self):
        adapter = Adapter2D().eval()
        with torch.no_grad():
            out = adapter(torch.randn(1, 25, 16, 16))
        np.testing.assert_allclose(out.numpy(), 0.0, atol=1e-6)

    def test_select_planes_respects_gap(self):
        region = np.zeros((3, 4, 4, 10), np.float32)
        prior = np.zeros_like(region)
        region[0, :, :, 3] = 0.5
        region[0, :, :, 4] = 0.5
        region[0, :, :, 8] = 0.5
        roi = np.ones((4, 4, 10), bool)
        planes = select_planes(region, prior, roi, k=2, min_gap=2)
        self.assertTrue(all(abs(a - b) >= 2 for a in planes for b in planes if a != b))


if __name__ == "__main__":
    unittest.main()
