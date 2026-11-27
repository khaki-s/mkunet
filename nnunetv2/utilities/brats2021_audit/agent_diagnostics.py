"""Small detached diagnostics; routing is not causal modality contribution."""
from collections import defaultdict

import torch
import math

from .missing_modality import MODALITIES, mask_codes


def gradient_summary(network, mask):
    groups = {'joint_encoder': network.encoder, 'joint_decoder': network.decoder}
    if network.semantic is not None:
        groups.update({f'expert_{m}': e for m, e in zip(MODALITIES, network.semantic.experts)})
        groups['router'] = network.semantic.router
    if network.communication is not None:
        groups['communication'] = network.communication
    if network.lomix is not None:
        groups['lomix'] = network.lomix
    result = {}
    for name, module in groups.items():
        grads = [p.grad.detach().float() for p in module.parameters() if p.grad is not None]
        norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()) if grads else 0.
        result[name] = {'gradient_l2': norm if math.isfinite(norm) else None,
                        'nonfinite_gradient': not math.isfinite(norm), 'parameters_with_gradient': len(grads)}
    result['active_samples'] = dict(zip(MODALITIES, mask.sum(0).cpu().tolist()))
    return result


def output_summary(parts):
    if 'weights' not in parts:
        return {'mask_codes': mask_codes(parts['modality_mask']).cpu().tolist()}
    with torch.no_grad():
        # Strided diagnostic sample bounds temporary memory on the full patch.
        rms = lambda z: z.detach().float()[:, :, ::4, ::4, ::4].square().mean((2, 3, 4)).sqrt().cpu().tolist()
        result = {'mask_codes': mask_codes(parts['modality_mask']).cpu().tolist(),
                  'weights': parts['weights'].detach().cpu().tolist(), 'beta': parts['beta'].detach().cpu().tolist(),
                  'joint_logit_rms': rms(parts['joint']) if 'joint' in parts else None,
                  'local_logit_rms': [rms(z) for z in parts['local_experts']],
                  'refined_logit_rms': [rms(z) for z in parts['experts']],
                  'scope': 'training patch routing, not case-level causal importance'}
        if parts.get('joint_present') is False:
            result.update(joint_present=False, fusion_kind=parts['fusion_kind'],
                          beta_kind='fixed_one_for_csv_compatibility', beta_trainable=False)
        if 'expert_names' in parts:
            result['expert_names'] = list(parts['expert_names'])
        if parts['attention'] is not None:
            result['message_attention'] = parts['attention'].detach().cpu().tolist()
        return result


class RoutingAccumulator:
    def __init__(self):
        self.rows = defaultdict(list)

    def add(self, parts):
        codes = mask_codes(parts['modality_mask']).cpu().tolist()
        for b, code in enumerate(codes):
            values = (torch.cat((parts['weights'][b].detach().flatten(), parts['beta'][b].detach())).float().cpu()
                      if 'weights' in parts else torch.empty(0))
            self.rows[code].append(values)

    def finish(self):
        result = {}
        for code, rows in sorted(self.rows.items()):
            result[str(code)] = {'samples': len(rows)}
            if rows[0].numel():
                avg = torch.stack(rows).mean(0)
                result[str(code)].update(weights_WT_TC_ET=avg[:12].reshape(3, 4).tolist(), beta=avg[12:].tolist())
        self.rows.clear()
        return result


@torch.no_grad()
def branch_interventions(network, parts, target):
    """Fixed-input diagnostics; joint retains all currently available inputs.

    Output-edge removal and sender-edge removal are intentionally distinct
    from the separate full-pipeline missing-input experiment.
    """
    from .diagnostics import patch_metrics
    if network.semantic is None or not parts['active']:
        return {'scope': 'fusion inactive or no experts', 'interventions': {}}
    mask, beta = parts['modality_mask'], parts['beta'][:, :, None, None, None]
    reference = patch_metrics(parts['final'], target)
    def score(experts, weights):
        logits = (1 - beta) * parts['joint'].float() + beta * network.semantic.fuse(experts, weights)
        metrics = patch_metrics(logits, target)
        delta = {c: reference[c]['dice'] - row['dice'] if reference[c]['dice'] is not None and row['dice'] is not None else None
                 for c, row in metrics.items()}
        return {'metrics': metrics, 'full_minus_intervened_dice': delta}
    uniform = (mask.float() / mask.sum(1, keepdim=True))[:, None].expand(-1, 3, -1)
    result = {'uniform_output_weights': score(parts['experts'], uniform)}
    for m, name in enumerate(MODALITIES):
        if not mask[:, m].any():
            continue
        if (mask.sum(1) > mask[:, m].long()).all():
            weights = parts['weights'].clone(); weights[:, :, m] = 0
            weights = weights / weights.sum(-1, keepdim=True)
            result[f'remove_output_{name}'] = score(parts['experts'], weights)
        if network.communication is not None and bool(network.communication_active):
            senders = mask.clone(); senders[:, m] = False
            refined, _ = network.communication(parts['local_experts'], parts['states'], mask, sender_mask=senders)
            result[f'remove_messages_from_{name}'] = score(refined, parts['weights'])
    return {'scope': 'fixed-input branch intervention; joint inputs retained; not physical missingness', 'interventions': result}
