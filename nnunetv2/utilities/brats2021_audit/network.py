"""Reuse initialized nnU-Net encoder/decoder; attach independent ablations."""
import torch
from torch import nn
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from .blocks import BoundedModalityGate3D, ClassConditionalMoE3D, FrequencySkip3D, SegMambaV2Residual3D


class BraTSAuditUNet(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.mode = mode
        channels = self.encoder.output_channels
        self.modality_gate = BoundedModalityGate3D() if mode == 'gate' else nn.Identity()
        self.moe = ClassConditionalMoE3D(channels[1], prior=mode != 'moe_no_prior',
                                       uniform=mode == 'moe_uniform') if mode.startswith('moe') else None
        self.frequency = nn.ModuleList([FrequencySkip3D(c) for c in channels[:2]]) if mode == 'frequency' else nn.ModuleList()
        self.context = SegMambaV2Residual3D(channels[-1]) if mode in ('mamba', 'moe_mamba') else nn.Identity()

    def forward(self, x):
        skips = list(self.encoder(self.modality_gate(x)))
        if self.moe is not None:
            skips[1] = self.moe(skips[1], x)
        for index, block in enumerate(self.frequency):
            skips[index] = block(skips[index])
        skips[-1] = self.context(skips[-1])
        return self.decoder(skips)


def build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                        enable_deep_supervision=True, mode='baseline'):
    allowed = {'baseline', 'gate', 'moe', 'moe_no_prior', 'moe_uniform', 'frequency', 'mamba', 'moe_mamba'}
    if mode not in allowed:
        raise ValueError(f'Unknown mode {mode}')
    if len(configuration_manager.patch_size) != 3 or num_input_channels != 4 or num_output_channels != 3:
        raise ValueError('These ablations require 3D, four MRI channels, and three region heads')
    if configuration_manager.network_arch_class_name.split('.')[-1] != 'PlainConvUNet':
        raise ValueError('Only the audited PlainConvUNet plans are supported')
    # Initialize the baseline FIRST, using exactly nnU-Net's official initializer.
    # Never apply a global initializer after constructing Mamba: dt_bias has
    # special initialization, and identity gates must retain their zero weights.
    backbone = get_network_from_plans(configuration_manager.network_arch_class_name,
                                     configuration_manager.network_arch_init_kwargs,
                                     configuration_manager.network_arch_init_kwargs_req_import,
                                     num_input_channels, num_output_channels, allow_init=True,
                                     deep_supervision=enable_deep_supervision)
    with torch.random.fork_rng(devices=[]):
        return BraTSAuditUNet(backbone, mode)
