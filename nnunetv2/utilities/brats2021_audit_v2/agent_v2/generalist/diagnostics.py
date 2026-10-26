"""Five named experts; physical masks retain the original four-bit convention."""
from collections import defaultdict
import math

import torch

from ...agent_v1.diagnostics import output_summary
from ...diagnostics import patch_metrics
from ...missing_modality import mask_codes
from .blocks import EXPERT_NAMES


class GeneralistRoutingAccumulator:
    def __init__(self):
        self.rows = defaultdict(list)

    def add(self, parts):
        for b, code in enumerate(mask_codes(parts['modality_mask']).cpu().tolist()):
            self.rows[code].append(torch.cat((parts['weights'][b].detach().flatten(),
                                             parts['beta'][b].detach())).float().cpu())

    def finish(self):
        result = {}
        for code, rows in sorted(self.rows.items()):
            avg = torch.stack(rows).mean(0)
            result[str(code)] = dict(samples=len(rows), expert_names=EXPERT_NAMES,
                weights_WT_TC_ET=avg[:15].reshape(3, 5).tolist(), beta=avg[15:].tolist())
        self.rows.clear()
        return result


def generalist_summary(network, parts, *, include_gradients=True):
    result = output_summary(parts)
    result.update(expert_names=EXPERT_NAMES, expert_mask=parts['expert_mask'].cpu().tolist())
    if parts['message_gate'] is not None:
        result['message_gate'] = parts['message_gate'].cpu().tolist()
    if not include_gradients:
        return result
    groups = {'joint_encoder': network.encoder, 'joint_decoder': network.decoder,
              'router': network.semantic.router, 'communication': network.communication}
    groups.update({f'expert_{name}': module for name, module in zip(EXPERT_NAMES, network.semantic.experts)})
    groups.update({name: getattr(network.communication, name) for name in ('query', 'key', 'value', 'gate')})
    gradients = {}
    for name, module in groups.items():
        norms = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        norm = float(torch.stack(norms).sum().sqrt()) if norms else 0.
        gradients[name] = dict(gradient_l2=norm if math.isfinite(norm) else None,
                               parameters_with_gradient=len(norms), nonfinite_gradient=not math.isfinite(norm))
    result['gradients'] = gradients
    result['active_samples'] = dict(zip(EXPERT_NAMES, parts['expert_mask'].sum(0).cpu().tolist()))
    return result


@torch.no_grad()
def generalist_interventions(network, parts, target):
    """Ablate output edges and message edges separately, at fixed physical input."""
    if not parts['active']:
        return {'scope': 'fusion inactive'}
    beta = parts['beta'][:, :, None, None, None]
    reference = patch_metrics(parts['final'], target)
    def score(experts, weights):
        logits = (1-beta)*parts['joint'].float() + beta*network.semantic.fuse(experts, weights)
        metrics = patch_metrics(logits, target)
        return {'metrics': metrics, 'full_minus_intervened_dice': {
            r: reference[r]['dice']-v['dice'] if reference[r]['dice'] is not None and v['dice'] is not None else None
            for r, v in metrics.items()}}
    result = {'joint_only': patch_metrics(parts['joint'], target),
              'generalist_only': patch_metrics(parts['local_experts'][4], target),
              'unrefined_fusion': score(parts['local_experts'], parts['weights'])}
    for m, name in enumerate(EXPERT_NAMES):
        if not parts['expert_mask'][:, m].any():
            continue
        weights = parts['weights'].clone(); weights[:, :, m] = 0
        denominator = weights.sum(-1, keepdim=True)
        if (denominator > 0).all():
            result[f'remove_output_{name}'] = score(parts['experts'], weights/denominator)
        if bool(network.communication_active):
            senders = parts['expert_mask'].clone(); senders[:, m] = False
            refined, _ = network.communication(parts['local_experts'], parts['states'],
                                                parts['expert_mask'], sender_mask=senders)
            result[f'remove_messages_from_{name}'] = score(refined, parts['weights'])
    return {'scope': 'fixed validation patch and inputs; diagnostic, not test-set causal attribution',
            'interventions': result}
