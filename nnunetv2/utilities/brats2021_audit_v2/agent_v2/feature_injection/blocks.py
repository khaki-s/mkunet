"""Co-registered expert-axis cross-attention using PyTorch's tested MHA.

The decoder loop is adapted from dynamic_network_architectures.UNetDecoder:
its initialized modules and parameter names are reused, not reinitialized.
LPMoE motivates residual feature injection and MoME motivates query-dependent
expert selection. This 3D, same-voxel attention is a task adaptation, not their
deformable/object-query implementation. See REFERENCES.md.
"""
import torch
from torch import nn
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder


class JointExpertCrossAttention3D(nn.Module):
    """One joint query attends to available expert tokens at each voxel.

    No attention across spatial positions. Three heads are latent feature
    heads; they are not claimed to correspond to WT/TC/ET. Missingness is an
    explicit Bx5 mask with an always-observed Generalist, never inferred from
    zero-valued spatial patches.
    """

    def __init__(self, joint_channels, expert_channels, experts=5,
                 embed_dim=48, heads=3):
        super().__init__()
        self.experts = experts
        self.query_project = nn.Conv3d(joint_channels, embed_dim, 1)
        self.expert_project = nn.ModuleList(
            [nn.Conv3d(expert_channels, embed_dim, 1) for _ in range(experts)])
        self.query_norm = nn.LayerNorm(embed_dim)
        self.expert_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, heads, dropout=0., batch_first=True)
        self.output = nn.Conv3d(embed_dim, joint_channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.last_attention = None
        self.last_relative_update_rms = None

    def forward(self, joint, states, expert_mask, *, capture=False):
        b, _, d, h, w = joint.shape
        if (expert_mask.shape != (b, self.experts)
                or expert_mask.dtype != torch.bool
                or len(states) != self.experts):
            raise ValueError('Feature injection requires boolean B x 5 expert mask and five states')
        if not expert_mask.any(1).all():
            raise ValueError('Feature injection needs at least one available expert')
        if any(s is None or s.shape[0] != b or s.shape[2:] != joint.shape[2:] for s in states):
            raise ValueError('Expert and joint decoder states must be spatially co-registered')
        volume = d * h * w
        query = self.query_project(joint).flatten(2).transpose(1, 2)
        projected = []
        for m, state in enumerate(states):
            indices = torch.where(expert_mask[:, m])[0]
            values = joint.new_zeros((b, query.shape[-1], d, h, w))
            if indices.numel():
                present = self.expert_project[m](state.index_select(0, indices))
                values = present.new_zeros(values.shape).index_copy(0, indices, present)
            projected.append(values.flatten(2).transpose(1, 2))
        # B,V,M,D is a sequence of expert tokens, not a concatenated backbone
        # channel tensor. PyTorch owns the Q/K/V projections and head reduction.
        tokens = torch.stack(projected, dim=2)
        missing = (~expert_mask[:, None]).expand(b, volume, self.experts)
        with torch.autocast(device_type=joint.device.type, enabled=False):
            query = self.query_norm(query.float()).reshape(b * volume, 1, -1)
            tokens = self.expert_norm(tokens.float()).reshape(b * volume, self.experts, -1)
            message, weights = self.attention(
                query, tokens, tokens, key_padding_mask=missing.reshape(b * volume, self.experts),
                need_weights=capture, average_attn_weights=False)
        message = message.reshape(b, volume, -1).transpose(1, 2).reshape(b, -1, d, h, w)
        update = self.output(message).to(joint.dtype)
        if capture:
            # B,H,M,D,H,W; independent from existing B,region,M output routing.
            self.last_attention = (weights.detach().squeeze(2)
                .reshape(b, volume, self.attention.num_heads, self.experts)
                .permute(0, 2, 3, 1).reshape(b, self.attention.num_heads, self.experts, d, h, w))
            base = joint.detach().float().square().mean((1, 2, 3, 4)).sqrt().clamp_min(1e-8)
            self.last_relative_update_rms = update.detach().float().square().mean((1, 2, 3, 4)).sqrt() / base
        else:
            self.last_attention = self.last_relative_update_rms = None
        return joint + update


class ExpertInjectedUNetDecoder(UNetDecoder):
    """One explicit decoder pass; all original parameter paths are preserved."""

    def __init__(self, original, feature_level):
        nn.Module.__init__(self)
        if not isinstance(original, UNetDecoder):
            raise TypeError('Expected dynamic_network_architectures UNetDecoder')
        for name in ('deep_supervision', 'encoder', 'num_classes',
                     'stages', 'transpconvs', 'seg_layers'):
            setattr(self, name, getattr(original, name))
        self.injection_stage = len(self.stages) - 1 - int(feature_level)
        if not 0 <= self.injection_stage < len(self.stages):
            raise ValueError('Expert feature level must correspond to a decoder stage')

    def forward(self, skips, *, injector=None, expert_states=None,
                expert_mask=None, capture=False):
        low = skips[-1]
        outputs = []
        for stage in range(len(self.stages)):
            # This is nnU-Net's original skip connection, unchanged. Expert
            # tensors are never concatenated into this backbone input.
            value = self.stages[stage](torch.cat(
                (self.transpconvs[stage](low), skips[-(stage + 2)]), dim=1))
            if injector is not None and stage == self.injection_stage:
                value = injector(value, expert_states, expert_mask, capture=capture)
            if self.deep_supervision or stage == len(self.stages) - 1:
                outputs.append(self.seg_layers[stage](value))
            low = value
        outputs.reverse()
        return outputs if self.deep_supervision else outputs[0]
