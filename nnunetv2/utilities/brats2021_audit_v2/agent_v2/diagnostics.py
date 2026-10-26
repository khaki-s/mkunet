"""True logit residuals and fixed-input communication interventions."""
import math
import torch
from torch.nn import functional as F

from ..missing_modality import MODALITIES
from ..diagnostics import patch_metrics


@torch.no_grad()
def residual_summary(parts, stride=4):
    result = {'stride': stride, 'scope': 'strided training/validation patch; not whole-case effect',
              'agent_axis': list(MODALITIES), 'region_axis': ['WT', 'TC', 'ET'], 'agents': {}}
    for m, name in enumerate(MODALITIES):
        local = parts['local_experts'][m].detach().float()[:, :, ::stride, ::stride, ::stride]
        refined = parts['experts'][m].detach().float()[:, :, ::stride, ::stride, ::stride]
        residual = (refined-local).square().mean((2,3,4)).sqrt()
        scale = local.square().mean((2,3,4)).sqrt()
        prob = (refined.sigmoid()-local.sigmoid()).abs().mean((2,3,4))
        result['agents'][name] = {'present': parts['modality_mask'][:,m].cpu().tolist(),
                                 'residual_rms': residual.cpu().tolist(),
                                 'relative_residual': (residual/(scale+1e-6)).cpu().tolist(),
                                 'prob_delta_l1': prob.cpu().tolist()}
    a = parts.get('attention')
    if a is not None:
        a = a.detach().float()
        null = parts.get('null_attention')
        entropy = -(a * a.clamp_min(1e-12).log()).sum(-1)
        if null is not None:
            entropy -= null.float() * null.float().clamp_min(1e-12).log()
            result['null_attention'] = null.cpu().tolist()
        result['attention_entropy_receiver_region'] = entropy.cpu().tolist()
        result['accepted_peer_mass'] = a.sum(-1).cpu().tolist()
        result['entropy_note'] = 'k=1 has no peer choice; absent receivers have zero mass'
    return result


def communication_gradient_summary(module):
    result = {}
    for group in ('project','query','key','value','correct','rho','null_score'):
        params = [p for n,p in module.named_parameters() if n.split('.')[0] == group]
        grads = [p.grad.detach().float() for p in params if p.grad is not None]
        elements = sum(g.numel() for g in grads)
        norm = float(torch.stack([g.square().sum() for g in grads]).sum().sqrt()) if grads else 0.
        result[group] = {'parameter_elements': sum(p.numel() for p in params),
                         'gradient_elements': elements, 'gradient_l2': norm if math.isfinite(norm) else None,
                         'gradient_rms': norm/max(1,elements)**.5 if math.isfinite(norm) else None,
                         'nonfinite': not math.isfinite(norm)}
    return result


@torch.no_grad()
def no_peer_intervention(network, parts, target, stride=4):
    """Recompute only the correction; retain images, joint, router and local states."""
    mask = parts['modality_mask']
    disabled = torch.zeros_like(mask)
    baseline, _ = network.communication(parts['local_experts'], parts['states'], mask, sender_mask=disabled)
    beta = parts['beta'][:,:,None,None,None]
    no_peer = (1-beta)*parts['joint'].float() + beta*network.semantic.fuse(baseline, parts['weights'])
    full = parts['final'].float()
    reference, intervened = patch_metrics(full,target), patch_metrics(no_peer,target)
    agents = {}
    for m,name in enumerate(MODALITIES):
        if not mask[:,m].any():
            continue
        present = mask[:,m]
        before = parts['local_experts'][m][present].float()
        after = parts['experts'][m][present].float()
        local_corrected = baseline[m][present].float()
        y = target[present].float()
        agents[name] = {'refined_minus_local_bce': float(F.binary_cross_entropy_with_logits(after,y)-F.binary_cross_entropy_with_logits(before,y)),
                        'with_peer_minus_no_peer_bce': float(F.binary_cross_entropy_with_logits(after,y)-F.binary_cross_entropy_with_logits(local_corrected,y)),
                        'peer_residual_rms': (after[:,:,::stride,::stride,::stride]-local_corrected[:,:,::stride,::stride,::stride]).square().mean((2,3,4)).sqrt().cpu().tolist()}
    return {'scope': 'fixed validation patch, all outgoing peer edges disabled; local correction retained',
            'full': reference, 'no_peer': intervened,
            'peer_gain_dice': {r: (v['dice']-intervened[r]['dice'] if v['dice'] is not None and intervened[r]['dice'] is not None else None) for r,v in reference.items()},
            'peer_logit_rms': (full[:,:,::stride,::stride,::stride]-no_peer[:,:,::stride,::stride,::stride]).square().mean((2,3,4)).sqrt().cpu().tolist(),
            'agents': agents}
