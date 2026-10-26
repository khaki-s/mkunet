"""nnU-Net joint backbone plus four supervised SimMLM UNet instances."""
import torch
from torch import nn

from ..baseline.network import build_audit_network
from .blocks import SemanticModalityExperts3D


class BraTSSemanticUNet(nn.Module):
    def __init__(self, backbone, configuration_manager, uniform):
        super().__init__()
        self.encoder, self.decoder = backbone.encoder, backbone.decoder
        self.moe = None  # Old feature-MoE diagnostics do not apply to logit fusion.
        kwargs = configuration_manager.network_arch_init_kwargs
        self.semantic = SemanticModalityExperts3D(kwargs['kernel_sizes'], kwargs['strides'],
                                                   configuration_manager.patch_size, uniform)
        self.register_buffer('initialization_seed', torch.tensor(torch.initial_seed() % (2**63 - 1), dtype=torch.int64))

    def set_epoch(self, epoch, warmup_epochs=50):
        self.semantic.set_joint_stage(epoch >= warmup_epochs)

    def forward(self, x, return_aux=False):
        joint_outputs = self.decoder(self.encoder(x))
        joint = joint_outputs[0] if isinstance(joint_outputs, (list, tuple)) else joint_outputs
        if not return_aux and not bool(self.semantic.fusion_active.item()):
            return joint_outputs
        parts = self.semantic(x, joint)
        output = [parts['final'], *joint_outputs[1:]] if isinstance(joint_outputs, (list, tuple)) else parts['final']
        if return_aux:
            return {'output': output, **parts}
        return output


def build_semantic_network(configuration_manager, num_input_channels, num_output_channels,
                           enable_deep_supervision=True, uniform=False):
    # Build and initialize the original backbone before any extra random draw.
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    # New module construction cannot alter the caller's subsequent RNG stream.
    with torch.random.fork_rng(devices=[]):
        return BraTSSemanticUNet(backbone, configuration_manager, uniform)
