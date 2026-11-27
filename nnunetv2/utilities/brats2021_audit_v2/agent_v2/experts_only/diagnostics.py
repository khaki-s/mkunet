"""Expert-only diagnostics, never naming a small branch as the removed J."""
import math

import torch

from ...agent_v1.diagnostics import output_summary
from ..generalist.blocks import EXPERT_NAMES
from ..generalist.diagnostics import GeneralistRoutingAccumulator


class ExpertsOnlyRoutingAccumulator(GeneralistRoutingAccumulator):
    def finish(self):
        rows = super().finish()
        for row in rows.values():
            row.update(joint_present=False, beta_kind='fixed_one_for_csv_compatibility',
                       beta_trainable=False)
        return rows


def experts_only_summary(network, parts, *, include_gradients=True):
    result = output_summary(parts)
    result.update(expert_mask=parts['expert_mask'].cpu().tolist(),
                  active_samples=dict(zip(EXPERT_NAMES, parts['expert_mask'].sum(0).cpu().tolist())))
    if include_gradients:
        groups = {'router': network.semantic.router, 'communication': network.communication}
        groups.update({f'expert_{name}': module for name, module in zip(EXPERT_NAMES, network.semantic.experts)})
        gradients = {}
        for name, module in groups.items():
            values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
            norm = float(torch.stack(values).sum().sqrt()) if values else 0.
            gradients[name] = dict(gradient_l2=norm if math.isfinite(norm) else None,
                                   parameters_with_gradient=len(values), nonfinite_gradient=not math.isfinite(norm))
        result['gradients'] = gradients
    return result
