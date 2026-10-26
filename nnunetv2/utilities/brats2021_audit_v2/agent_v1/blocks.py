"""Sparse modality experts and one-round differentiable agent communication."""
import math
from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..agent_resize import resize_trilinear
from ..semantic_moe.blocks import SemanticModalityExperts3D, EXPERT_WIDTHS
from ..vendor.simmlm_unet import UNet


class FeatureUNet(UNet):
    """Expose one decoder state without changing vendor weights or their keys."""
    def __init__(self, original, feature_level):
        nn.Module.__init__(self)
        for name in ('input_channels', 'n_classes', 'n_stages', 'n_features_per_stage',
                     'kernel_size', 'strides', 'apply_deep_supervision'):
            setattr(self, name, getattr(original, name))
        for name in ('encoder_stages', 'connect_layers', 'decoder_stages', 'seg_layers'):
            setattr(self, name, getattr(original, name))
        self.feature_level = feature_level

    def forward(self, x, return_features=False):
        return self.forward_with_features(x) if return_features else super().forward(x)

    def forward_with_features(self, x):
        encoded = []
        for stage in self.encoder_stages:
            x = stage(x)
            encoded.append(x)
        x, feature = encoded[-1], None
        for i, stage in enumerate(self.decoder_stages):
            x = stage(torch.cat((self.connect_layers[i](x), encoded[-i - 2]), 1))
            if self.n_stages - i - 2 == self.feature_level:
                feature = x
        if feature is None:
            raise RuntimeError('Selected feature level is not a decoder level')
        return self.seg_layers[-1](x), feature


class MissingSemanticExperts3D(SemanticModalityExperts3D):
    def __init__(self, kernel_sizes, strides, patch_size, uniform=False, conditional_beta=False):
        super().__init__(kernel_sizes, strides, patch_size, uniform)
        cumulative, scales = [1, 1, 1], []
        for stride in strides[:-1]:
            cumulative = [a * b for a, b in zip(cumulative, stride)]
            scales.append(math.prod(cumulative) ** (1 / 3))
        self.feature_level = min(range(len(scales)), key=lambda i: abs(math.log2(scales[i]) - 3))
        self.feature_channels = EXPERT_WIDTHS[self.feature_level]
        self.experts = nn.ModuleList([FeatureUNet(e, self.feature_level) for e in self.experts])
        self.beta_condition = None
        if conditional_beta:
            self.beta_condition = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
            nn.init.zeros_(self.beta_condition[-1].weight)
            nn.init.zeros_(self.beta_condition[-1].bias)
        self.set_joint_stage(False)

    def set_joint_stage(self, active):
        super().set_joint_stage(active)
        if getattr(self, 'beta_condition', None) is not None:
            self.beta_condition.requires_grad_(bool(active))

    def weights(self, x, mask):
        if self.uniform or not bool(self.fusion_active.item()):
            return (mask.float() / mask.sum(1, keepdim=True))[:, None, :].expand(-1, 3, -1)
        if tuple(x.shape[2:]) != self.router.patch_size:
            raise ValueError(f'Router expects patch {self.router.patch_size}, got {tuple(x.shape[2:])}')
        scores = self.router.head(self.router.features(x).flatten(1)).reshape(x.shape[0], 3, 4)
        return scores.float().masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)

    def beta(self, x, mask):
        logits = self.beta_logit.float()[None, :].expand(x.shape[0], -1)
        if self.beta_condition is not None:
            condition = torch.cat((mask.float(), x.float().mean((2, 3, 4))), 1)
            logits = logits + self.beta_condition(condition).float()
        return logits.sigmoid()

    def local_predictions(self, x, joint, mask, need_features=False):
        predictions, states = [], []
        for m, expert in enumerate(self.experts):
            indices = torch.where(mask[:, m])[0]
            if indices.numel() == 0:
                predictions.append(joint.new_zeros(joint.shape))
                states.append(None)
                continue
            image = x.index_select(0, indices)[:, m:m + 1]
            call = partial(expert, return_features=True) if need_features else expert
            if self.training and torch.is_grad_enabled() and self.checkpoint_experts:
                value = checkpoint(call, image, use_reentrant=False, preserve_rng_state=False)
            else:
                value = call(image)
            z, h = value if need_features else (value, None)
            predictions.append(z.new_zeros(joint.shape).index_copy(0, indices, z))
            states.append(None if h is None else h.new_zeros((x.shape[0], *h.shape[1:])).index_copy(0, indices, h))
        if need_features:
            template = next(h for h in states if h is not None)
            states = [template.new_zeros(template.shape) if h is None else h for h in states]
        return predictions, states


class ModalityCommunication3D(nn.Module):
    """Observed agents send region-specific messages and correct local logits.

    local_only retains identical parameters but removes cross-agent messages.
    rho starts at zero; correction heads must NOT also be zero initialized.
    """
    def __init__(self, input_channels, message_channels=16, local_only=False):
        super().__init__()
        self.channels, self.local_only = message_channels, bool(local_only)
        self.project = nn.ModuleList([nn.Conv3d(input_channels, message_channels, 1) for _ in range(4)])
        self.query = nn.Linear(message_channels, 3 * message_channels)
        self.key = nn.Linear(message_channels, 3 * message_channels)
        self.value = nn.Conv3d(message_channels, 3 * message_channels, 1)
        self.correct = nn.ModuleList([nn.Sequential(
            nn.Conv3d(4 * message_channels + 3, message_channels, 3, padding=1),
            nn.LeakyReLU(), nn.Conv3d(message_channels, 3, 1)) for _ in range(4)])
        self.rho = nn.Parameter(torch.zeros(4))
        for block in self.correct:
            nn.init.normal_(block[-1].weight, std=.01)
            nn.init.zeros_(block[-1].bias)

    def forward(self, local, states, mask, *, sender_mask=None):
        b, d = mask.shape[0], self.channels
        projected = []
        for m, h in enumerate(states):
            indices = torch.where(mask[:, m])[0]
            if indices.numel():
                value = self.project[m](h.index_select(0, indices))
                projected.append(value.new_zeros((b, d, *h.shape[2:])).index_copy(0, indices, value))
            else:
                projected.append(h.new_zeros((b, d, *h.shape[2:])))
        pooled = torch.stack([h.mean((2, 3, 4)) for h in projected], 1)
        q = self.query(pooled).reshape(b, 4, 3, d).float()
        k = self.key(pooled).reshape(b, 4, 3, d).float()
        scores = torch.einsum('bmcd,bncd->bmcn', q, k) / math.sqrt(d)
        peers = mask[:, :, None] & mask[:, None, :] & ~torch.eye(4, device=mask.device, dtype=torch.bool)[None]
        if sender_mask is not None:
            sender_mask = torch.as_tensor(sender_mask, device=mask.device)
            if sender_mask.shape != mask.shape or not ((sender_mask == 0) | (sender_mask == 1)).all():
                raise ValueError('Diagnostic sender_mask must be binary B x 4')
            peers = peers & sender_mask.bool()[:, None, :]
        if self.local_only:
            peers = torch.zeros_like(peers)
        has_peers = peers.any(-1)
        scores = scores.masked_fill(~peers[:, :, None, :], -torch.inf)
        scores = torch.where(has_peers[:, :, None, None], scores, torch.zeros_like(scores))
        attention = scores.softmax(-1) * peers[:, :, None, :]
        values = [self.value(h).reshape(b, 3, d, *h.shape[2:]) for h in projected]
        small_logits = [resize_trilinear(z.float(), projected[0].shape[2:]).sigmoid() for z in local]
        refined = []
        for m in range(4):
            indices = torch.where(mask[:, m])[0]
            if not indices.numel():
                refined.append(local[m])
                continue
            message = sum(v * attention[:, m, :, n, None, None, None, None] for n, v in enumerate(values))
            peer_prediction = sum(z * attention[:, m, :, n, None, None, None] for n, z in enumerate(small_logits))
            disagreement = (peer_prediction - small_logits[m]) * has_peers[:, m, None, None, None, None]
            state = torch.cat((projected[m], message.flatten(1, 2), disagreement), 1).index_select(0, indices)
            delta = resize_trilinear(self.correct[m](state).float(), local[m].shape[2:])
            residual = delta.new_zeros(local[m].shape).index_copy(0, indices, delta)
            refined.append(local[m].float() + self.rho[m].tanh() * residual)
        return refined, attention
