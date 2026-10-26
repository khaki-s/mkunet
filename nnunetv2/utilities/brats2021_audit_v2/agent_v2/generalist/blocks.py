"""Five experts and receiver-gated attention over available expert states.

Attention output gating follows the headwise option in the official NeurIPS
2025 Gated Attention repository (qiuzh20/gated_attention, f4c2a5f6ffd6).
Self-inclusive targeting follows the TarMAC formulation (ICML 2019), not an
assertion that TarMAC or Gated Attention was evaluated on this segmentation task.
The spatial value maps / three region heads / residual logit decoder are our
3D adaptation. See REFERENCES.md for exact provenance and limitations.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...agent_v1.blocks import FeatureUNet
from ...agent_resize import resize_trilinear
from ...semantic_moe.blocks import EXPERT_WIDTHS, MODALITIES
from ...vendor.simmlm_unet import UNet

EXPERT_NAMES = (*MODALITIES, 'Generalist')


class GeneralistExperts3D(nn.Module):
    """Reuse the four specialists and append an independently supervised UNet.

    Physical inputs remain Bx4. Generalist receives four masked image channels
    plus four broadcast availability bits; it never receives missing images.
    """
    def __init__(self, original, kernel_sizes, strides):
        super().__init__()
        for name in ('uniform', 'checkpoint_experts', 'feature_level', 'feature_channels'):
            setattr(self, name, getattr(original, name))
        self.experts = original.experts
        generalist = UNet(8, 3, 6, list(EXPERT_WIDTHS), kernel_sizes, strides,
                          apply_deep_supervision=False)
        for head in generalist.seg_layers[:-1]:
            head.requires_grad_(False)
        self.experts.append(FeatureUNet(generalist, self.feature_level))
        self.router, self.beta_logit = original.router, original.beta_logit
        self.beta_condition = original.beta_condition
        old_head = self.router.head
        self.router.head = nn.Linear(old_head.in_features, 15)
        with torch.no_grad():
            self.router.head.weight.zero_(); self.router.head.bias.zero_()
            self.router.head.weight.reshape(3, 5, -1)[:, :4].copy_(old_head.weight.reshape(3, 4, -1))
            self.router.head.bias.reshape(3, 5)[:, :4].copy_(old_head.bias.reshape(3, 4))
        self.register_buffer('fusion_active', original.fusion_active.clone())
        self.set_joint_stage(False)

    @staticmethod
    def expert_mask(physical_mask):
        return torch.cat((physical_mask, torch.ones_like(physical_mask[:, :1])), dim=1)

    def set_joint_stage(self, active):
        self.fusion_active.fill_(bool(active))
        self.router.requires_grad_(bool(active) and not self.uniform)
        self.beta_logit.requires_grad_(bool(active))
        if self.beta_condition is not None:
            self.beta_condition.requires_grad_(bool(active))

    def weights(self, x, physical_mask):
        mask = self.expert_mask(physical_mask)
        if self.uniform or not bool(self.fusion_active):
            return (mask.float() / mask.sum(1, keepdim=True))[:, None].expand(-1, 3, -1)
        if tuple(x.shape[2:]) != self.router.patch_size:
            raise ValueError(f'Expected router patch {self.router.patch_size}, got {tuple(x.shape[2:])}')
        scores = self.router.head(self.router.features(x).flatten(1)).reshape(x.shape[0], 3, 5)
        return scores.float().masked_fill(~mask[:, None], -torch.inf).softmax(-1)

    def beta(self, x, physical_mask):
        logits = self.beta_logit.float()[None].expand(x.shape[0], -1)
        if self.beta_condition is not None:
            condition = torch.cat((physical_mask.float(), x.float().mean((2, 3, 4))), 1)
            logits = logits + self.beta_condition(condition).float()
        return logits.sigmoid()

    @staticmethod
    def fuse(experts, weights):
        return sum(z.float() * weights[:, :, i, None, None, None] for i, z in enumerate(experts))

    def local_predictions(self, x, joint, physical_mask, need_features=False):
        mask = self.expert_mask(physical_mask)
        predictions, states = [], []
        for m, expert in enumerate(self.experts):
            indices = torch.where(mask[:, m])[0]
            if not indices.numel():
                predictions.append(joint.new_zeros(joint.shape)); states.append(None)
                continue
            if m < 4:
                inputs = x.index_select(0, indices)[:, m:m + 1]
            else:
                bits = physical_mask.to(x.dtype)[:, :, None, None, None].expand_as(x)
                inputs = torch.cat((x, bits), dim=1)
            call = partial(expert, return_features=True) if need_features else expert
            value = (checkpoint(call, inputs, use_reentrant=False, preserve_rng_state=False)
                     if self.training and torch.is_grad_enabled() and self.checkpoint_experts else call(inputs))
            z, h = value if need_features else (value, None)
            predictions.append(z.new_zeros(joint.shape).index_copy(0, indices, z))
            states.append(None if h is None else h.new_zeros((x.shape[0], *h.shape[1:])).index_copy(0, indices, h))
        if need_features:
            template = next(h for h in states if h is not None)
            states = [template.new_zeros(template.shape) if h is None else h for h in states]
        return predictions, states


class GeneralistCommunication3D(nn.Module):
    """Same parameters in local / communicating / ungated controls.

    The self-inclusive local control allows only self-attention; the legacy
    local control zeros all messages, exactly as the original LocalV2 does.
    Sender interventions remove
    outgoing edges including that sender's self edge. Empty rows yield zero,
    never softmax(-inf,...,-inf). Routing weights and message gates differ.
    """
    def __init__(self, original, feature_channels, *, local_only=False,
                 output_gate=True, include_self=True):
        super().__init__()
        self.channels = d = original.channels
        self.local_only, self.output_gate, self.include_self = bool(local_only), bool(output_gate), bool(include_self)
        for name in ('project', 'query', 'key', 'value', 'correct'):
            setattr(self, name, getattr(original, name))
        self.project.append(nn.Conv3d(feature_channels, d, 1))
        fifth = nn.Sequential(nn.Conv3d(4*d + 3, d, 3, padding=1), nn.LeakyReLU(), nn.Conv3d(d, 3, 1))
        nn.init.zeros_(fifth[-1].weight); nn.init.zeros_(fifth[-1].bias)
        self.correct.append(fifth)
        self.register_buffer('rho', torch.full((5,), math.atanh(.1)))
        # Headwise, query-dependent post-attention sigmoid gate. Always created
        # so enabling the gate does not alter any other parameter initialization.
        self.gate = nn.Linear(d, 3, bias=False)
        self.last_gate = None

    def forward(self, local, states, mask, *, sender_mask=None):
        b, n = mask.shape
        d = self.channels
        if n != len(self.project) or len(local) != n or len(states) != n or mask.dtype != torch.bool:
            raise ValueError('Expected five local logits/states and a boolean B x 5 expert mask')
        projected = []
        for m, h in enumerate(states):
            indices = torch.where(mask[:, m])[0]
            value = h.new_zeros((b, d, *h.shape[2:]))
            if indices.numel():
                small = self.project[m](h.index_select(0, indices))
                value = small.new_zeros(value.shape).index_copy(0, indices, small)
            projected.append(value)
        pooled = torch.stack([h.mean((2, 3, 4)) for h in projected], 1)
        q = self.query(pooled).reshape(b, n, 3, d).float()
        k = self.key(pooled).reshape(b, n, 3, d).float()
        scores = torch.einsum('bmcd,bncd->bmcn', q, k) / math.sqrt(d)
        edges = mask[:, :, None] & mask[:, None, :]
        identity = torch.eye(n, dtype=torch.bool, device=mask.device)[None]
        if self.local_only:
            edges = edges & identity if self.include_self else torch.zeros_like(edges)
        elif not self.include_self:
            edges = edges & ~identity
        if sender_mask is not None:
            sender_mask = torch.as_tensor(sender_mask, device=mask.device)
            if sender_mask.shape != mask.shape or not ((sender_mask == 0) | (sender_mask == 1)).all():
                raise ValueError('sender_mask must be binary B x 5')
            edges = edges & sender_mask.bool()[:, None, :]
        has_source = edges.any(-1)
        scores = scores.masked_fill(~edges[:, :, None, :], -torch.inf)
        scores = torch.where(has_source[:, :, None, None], scores, torch.zeros_like(scores))
        attention = scores.softmax(-1) * edges[:, :, None, :]
        gates = self.gate(pooled).float().sigmoid() if self.output_gate else torch.ones_like(q[..., 0])
        gates = gates * mask[:, :, None]
        self.last_gate = gates.detach()
        values = [self.value(h).reshape(b, 3, d, *h.shape[2:]) for h in projected]
        probs = [resize_trilinear(z.float(), projected[0].shape[2:]).sigmoid() for z in local]
        refined = []
        for m in range(n):
            indices = torch.where(mask[:, m])[0]
            if not indices.numel():
                refined.append(local[m]); continue
            gate = gates[:, m, :, None, None, None]
            message = sum(v * attention[:, m, :, j, None, None, None, None] for j, v in enumerate(values))
            message = message * gate[:, :, None]
            peer_prediction = sum(p * attention[:, m, :, j, None, None, None] for j, p in enumerate(probs))
            mass = attention[:, m].sum(-1)[:, :, None, None, None]
            # Also gate the disagreement path, so suppressed messages cannot
            # leak back through a second cross-expert input to the decoder.
            disagreement = (peer_prediction - mass * probs[m]) * gate
            state = torch.cat((projected[m], message.flatten(1, 2), disagreement), 1).index_select(0, indices)
            delta = resize_trilinear(self.correct[m](state).float(), local[m].shape[2:])
            residual = delta.new_zeros(local[m].shape).index_copy(0, indices, delta)
            refined.append(local[m].float() + self.rho[m].tanh() * residual)
        return refined, attention
