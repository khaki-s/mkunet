"""Controlled V2 residual initialization and an optional zero-valued null peer."""
import math

import torch
from torch import nn

from ..agent_v1.blocks import ModalityCommunication3D
from ..agent_resize import resize_trilinear


class ModalityCommunicationV2(ModalityCommunication3D):
    """Keep the V1 projections/heads, changing only explicitly selected factors.

    fixed_identity: fixed tanh(rho)=0.1, zero final head; no dead zero*zero gate.
    legacy: retain V1 rho and head initialization for the auxiliary-only ablation.
    A null peer is a reject-message action with zero value, not a missing image.
    """
    def __init__(self, original, *, residual_mode='fixed_identity', null_peer=False):
        nn.Module.__init__(self)
        if residual_mode not in ('fixed_identity', 'legacy'):
            raise ValueError(f'Unknown residual initialization: {residual_mode}')
        self.channels, self.local_only = original.channels, original.local_only
        for name in ('project', 'query', 'key', 'value', 'correct'):
            setattr(self, name, getattr(original, name))
        self.residual_mode, self.null_peer = residual_mode, bool(null_peer)
        if residual_mode == 'legacy':
            self.rho = original.rho
        else:
            self.register_buffer('rho', torch.full((4,), math.atanh(.1)))
            for head in self.correct:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        if self.null_peer:
            self.null_score = nn.Parameter(torch.zeros(3))
        self.last_null_attention = None

    def forward(self, local, states, mask, *, sender_mask=None):
        if not self.null_peer:
            self.last_null_attention = None
            return super().forward(local, states, mask, sender_mask=sender_mask)
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
        scores = scores.masked_fill(~peers[:, :, None, :], -torch.inf)
        null = self.null_score.float()[None, None, :, None].expand(b, 4, 3, 1)
        probabilities = torch.cat((scores, null), -1).softmax(-1)
        attention = probabilities[..., :4] * peers[:, :, None, :]
        null_attention = probabilities[..., 4] * mask[:, :, None]
        self.last_null_attention = null_attention.detach()
        values = [self.value(h).reshape(b, 3, d, *h.shape[2:]) for h in projected]
        small_probs = [resize_trilinear(z.float(), projected[0].shape[2:]).sigmoid() for z in local]
        refined = []
        for m in range(4):
            indices = torch.where(mask[:, m])[0]
            if not indices.numel():
                refined.append(local[m])
                continue
            message = sum(v * attention[:, m, :, n, None, None, None, None] for n, v in enumerate(values))
            peer_prediction = sum(z * attention[:, m, :, n, None, None, None] for n, z in enumerate(small_probs))
            # Null contributes zero disagreement. Subtract only accepted peer mass.
            mass = attention[:, m].sum(-1)[:, :, None, None, None]
            disagreement = peer_prediction - mass * small_probs[m]
            state = torch.cat((projected[m], message.flatten(1, 2), disagreement), 1).index_select(0, indices)
            delta = resize_trilinear(self.correct[m](state).float(), local[m].shape[2:])
            residual = delta.new_zeros(local[m].shape).index_copy(0, indices, delta)
            refined.append(local[m].float() + self.rho[m].tanh() * residual)
        return refined, attention
