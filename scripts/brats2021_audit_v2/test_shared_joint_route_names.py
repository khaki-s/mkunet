"""Metadata regression: new Joint node must not be reported as Generalist."""
import unittest

import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit.agent_evaluate import aggregate_case_routes
from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import aggregate_case_routes as aggregate_v2
from nnunetv2.utilities.brats2021_audit_v2.agent_v1.diagnostics import output_summary


def route(count, names=None):
    weight = np.zeros((1, 3, count))
    weight[:, :, 0] = 1 if count == 4 else .5
    if count == 5:
        weight[:, :, 4] = .5
    result = dict(weights=weight.tolist(), beta=[[.2, .2, .2]],
                  local_logit_rms=np.ones((count, 1, 3)).tolist(),
                  refined_logit_rms=np.ones((count, 1, 3)).tolist())
    if names is not None:
        result['expert_names'] = names
    return result


class RouteNamesTests(unittest.TestCase):
    def test_legacy_defaults_and_new_joint(self):
        for count, names, last in [(4, None, 'FLAIR'), (5, None, 'Generalist'),
                                  (5, ['T1', 'T1ce', 'T2', 'FLAIR', 'Joint'], 'Joint')]:
            for aggregate in (aggregate_case_routes, aggregate_v2):
                result = aggregate([route(count, names)], 'case', 1)
                self.assertEqual(result[count - 1]['modality'], last)
                self.assertAlmostEqual(sum(r['weight'] for r in result[:count]), 1.)
                if count == 5:
                    self.assertEqual(result[count - 1]['present'], 1)
                    self.assertAlmostEqual(result[count - 1]['effective_weight'], .1)

    def test_conflicting_metadata_is_rejected(self):
        for aggregate in (aggregate_case_routes, aggregate_v2):
            with self.assertRaisesRegex(ValueError, 'expert names'):
                aggregate([
                    route(5, ['T1', 'T1ce', 'T2', 'FLAIR', 'Generalist']),
                    route(5, ['T1', 'T1ce', 'T2', 'FLAIR', 'Joint']),
                ], 'case', 1)

    def test_summary_retains_node_identity(self):
        names = ('T1', 'T1ce', 'T2', 'FLAIR', 'Joint')
        tensor = torch.ones(1, 3, 4, 4, 4)
        parts = dict(modality_mask=torch.ones(1, 4, dtype=torch.bool),
                     weights=torch.ones(1, 3, 5) / 5, beta=torch.ones(1, 3) / 5,
                     joint=tensor, local_experts=[tensor] * 5,
                     experts=[tensor] * 5, attention=None, expert_names=names)
        summary = output_summary(parts)
        self.assertEqual(summary['expert_names'], list(names))


if __name__ == '__main__':
    unittest.main()
