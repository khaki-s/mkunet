"""Semantic expert diagnostics with fixed validation patches and no extra forwards.

The probe reuses four expert logits for counterfactual fusion. Removing one
expert renormalizes the other three weights; it does NOT remove that input
modality from the joint backbone or router and is labelled accordingly.
"""
import hashlib

import torch
from torch.nn import functional as F

from .diagnostics import ModuleDiagnostics, finite_number, patch_metrics, spatial_sample
from .semantic_blocks import MODALITIES, REGIONS


class SemanticDiagnostics(ModuleDiagnostics):
    def __init__(self, network):
        super().__init__(network)
        # This module returns a dictionary, so no generic feature forward hook.
        self.modules = {'semantic': network.semantic}

    def end_step(self, amp_step_skipped=False):
        super().end_step(amp_step_skipped)
        row = self.records['semantic'][-1]
        named = list(self.network.semantic.named_parameters())
        row['grad_none_trainable_tensors'] = sum(p.requires_grad and p.grad is None for _, p in named)
        row['frozen_tensors'] = sum(not p.requires_grad for _, p in named)
        for group, metrics in row['grad_groups'].items():
            parameters = [p for name, p in named if ('.'.join(name.split('.')[:2])
                          if name.startswith('experts.') else name.split('.')[0]) == group]
            metrics['grad_none_trainable_tensors'] = sum(p.requires_grad and p.grad is None for p in parameters)
            metrics['frozen_tensors'] = sum(not p.requires_grad for p in parameters)

    @torch.no_grad()
    def observe_output(self, parts):
        if self.active:
            joint, final = spatial_sample(parts['joint']), spatial_sample(parts['final'])
            self.step_features['semantic'] = {
                'logit_delta_rms_over_joint_rms': finite_number(
                    (final - joint).square().mean().sqrt() / (joint.square().mean().sqrt() + 1e-8))}


def probe_digest(probe):
    h = hashlib.sha256()
    for key in ('data', 'target'):
        tensor = probe[key].contiguous()
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def routing_summary(weights, semantic):
    # All training samples, region rows and explicit modality columns.
    weights = weights.float()
    return {'regions': REGIONS, 'experts': MODALITIES, 'samples': len(weights),
            'scope': 'global per-patch routing, not voxel-local or causal importance',
            'mean': weights.mean(0).cpu().tolist(), 'std': weights.std(0, unbiased=False).cpu().tolist(),
            'min': weights.amin(0).cpu().tolist(), 'max': weights.amax(0).cpu().tolist(),
            'mean_entropy': (-(weights * weights.clamp_min(1e-8).log()).sum(-1)).mean(0).cpu().tolist(),
            'beta_moe': semantic.beta_logit.detach().float().sigmoid().cpu().tolist(),
            'fusion_active': bool(semantic.fusion_active.item()), 'fixed_uniform_control': semantic.uniform}


@torch.no_grad()
def semantic_probe(network, parts, target):
    semantic = network.semantic
    weights, experts = parts['weights'], parts['experts']
    full = patch_metrics(parts['final'], target)
    full_bce = F.binary_cross_entropy_with_logits(parts['final'].float(), target.float())
    result = {'scope': 'fixed validation patches, pooled; not full-case/sliding-window Dice',
              'regions': REGIONS, 'experts': MODALITIES, 'active': parts['active'],
              'weights_per_sample': weights.cpu().tolist(),
              'beta_moe': semantic.beta_logit.detach().float().sigmoid().cpu().tolist(),
              'full': full, 'expert_metrics': {}, 'counterfactuals': {}}
    for name, z in zip(MODALITIES, experts):
        result['expert_metrics'][name] = patch_metrics(z, target)
    result['expert_fusion_metrics'] = patch_metrics(parts['moe'], target)

    def compare(name, logits):
        other = patch_metrics(logits, target)
        result['counterfactuals'][name] = {
            'metrics': other,
            'delta_dice_pp_full_minus_counterfactual': {
                r: 100 * (full[r]['dice'] - other[r]['dice'])
                if full[r]['dice'] is not None and other[r]['dice'] is not None else None for r in full},
            'delta_bce_counterfactual_minus_full': finite_number(
                F.binary_cross_entropy_with_logits(logits.float(), target.float()) - full_bce),
            'mean_abs_probability_change': finite_number(
                (logits.float().sigmoid() - parts['final'].float().sigmoid()).abs().mean())}

    compare('joint_only', parts['joint'])
    uniform_moe = semantic.fuse(experts, torch.full_like(weights, .25))
    compare('uniform_expert_weights', semantic.blend(parts['joint'], uniform_moe, parts['active']))
    for m, name in enumerate(MODALITIES):
        # Preserve logits/router/joint. Remove only this expert's fusion edge.
        remaining = weights.clone()
        remaining[:, :, m] = 0
        denominator = remaining.sum(-1, keepdim=True)
        fallback = torch.ones_like(remaining)
        fallback[:, :, m] = 0
        fallback /= 3
        remaining = torch.where(denominator > 1e-8, remaining / denominator.clamp_min(1e-8), fallback)
        counterfactual = semantic.blend(parts['joint'], semantic.fuse(experts, remaining), parts['active'])
        compare('remove_expert_' + name, counterfactual)
    result['removal_definition'] = 'remove expert fusion edge and renormalize; joint and router still see all four inputs'
    return result
