"""Five communication nodes, four experts; never mislabel Joint as extra UNet."""
import math

import torch

from ...agent_v1.diagnostics import output_summary
from ...diagnostics import patch_metrics
from ..generalist.diagnostics import GeneralistRoutingAccumulator
from .network import NODE_NAMES


class SharedJointRoutingAccumulator(GeneralistRoutingAccumulator):
    def finish(self):
        result = super().finish()
        for value in result.values():
            value['expert_names'] = NODE_NAMES
            value['independent_expert_networks'] = 4
            value['joint_is_node_5'] = True
        return result


def shared_joint_summary(network, parts, *, include_gradients=True):
    result = output_summary(parts)
    result.update(expert_names=NODE_NAMES, expert_mask=parts['expert_mask'].cpu().tolist(),
                  joint_is_node_5=True, independent_expert_networks=len(network.semantic.experts))
    result['active_samples'] = dict(zip(NODE_NAMES, parts['expert_mask'].sum(0).cpu().tolist()))
    if not include_gradients:
        return result
    groups = {'joint_encoder': network.encoder, 'joint_decoder': network.decoder,
              'router': network.semantic.router, 'communication': network.communication}
    groups.update({f'expert_{name}': module for name, module in zip(NODE_NAMES[:4], network.semantic.experts)})
    groups.update({name: getattr(network.communication, name) for name in ('query', 'key', 'value', 'gate')})
    gradients = {}
    for name, module in groups.items():
        values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        value = float(torch.stack(values).sum().sqrt()) if values else 0.
        gradients[name] = {'gradient_l2': value if math.isfinite(value) else None,
                           'parameters_with_gradient': len(values), 'nonfinite_gradient': not math.isfinite(value)}
    result['gradients'] = gradients
    return result


@torch.no_grad()
def shared_joint_interventions(network, parts, target):
    if not parts['active']:
        return {'scope': 'fusion inactive'}
    beta = parts['beta'][:, :, None, None, None]
    reference = patch_metrics(parts['final'], target)
    def score(nodes, weights):
        value = (1 - beta) * parts['joint'].float() + beta * network.semantic.fuse(nodes, weights)
        metrics = patch_metrics(value, target)
        return {'metrics': metrics, 'full_minus_intervened_dice': {
            region: reference[region]['dice'] - row['dice']
            if reference[region]['dice'] is not None and row['dice'] is not None else None
            for region, row in metrics.items()}}
    result = {'raw_joint_only': patch_metrics(parts['joint'], target),
              'communicated_joint_only': patch_metrics(parts['experts'][4], target),
              'unrefined_fusion': score(parts['local_experts'], parts['weights'])}
    for node, name in enumerate(NODE_NAMES):
        if not parts['expert_mask'][:, node].any():
            continue
        weights = parts['weights'].clone()
        weights[:, :, node] = 0
        denominator = weights.sum(-1, keepdim=True)
        if (denominator > 0).all():
            result[f'remove_output_{name}'] = score(parts['experts'], weights / denominator)
        if bool(network.communication_active):
            senders = parts['expert_mask'].clone()
            senders[:, node] = False
            refined, _ = network.communication(parts['local_experts'], parts['states'],
                                               parts['expert_mask'], sender_mask=senders)
            result[f'remove_messages_from_{name}'] = score(refined, parts['weights'])
    return {'scope': 'fixed validation patch; Joint output removal leaves raw joint bypass intact',
            'interventions': result}
