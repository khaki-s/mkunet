"""Add a real fifth expert without replacing or aliasing the joint backbone."""
import torch

from ..network import BraTSAgentV2UNet
from ...baseline.network import build_audit_network
from ...missing_modality import apply_modality_mask
from .blocks import GeneralistExperts3D, GeneralistCommunication3D, EXPERT_NAMES


class BraTSGeneralistAgentUNet(BraTSAgentV2UNet):
    def __init__(self, backbone, configuration_manager, *, correction_enabled=True,
                 output_gate=True, include_self=True, **options):
        if options.get('null_peer', False) or options.get('residual_mode', 'fixed_identity') != 'fixed_identity':
            raise ValueError('Generalist controls require fixed_identity and no null peer')
        super().__init__(backbone, configuration_manager, **options)
        kwargs = configuration_manager.network_arch_init_kwargs
        self.semantic = GeneralistExperts3D(self.semantic, kwargs['kernel_sizes'], kwargs['strides'])
        self.communication = GeneralistCommunication3D(self.communication, self.semantic.feature_channels,
            local_only=options.get('local_only', False), output_gate=output_gate, include_self=include_self)
        self.correction_enabled = bool(correction_enabled)
        self.architecture_version.fill_(3)
        self.expert_names = EXPERT_NAMES
        self.set_epoch(0)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if not getattr(self, 'correction_enabled', True):
            self.communication_active.fill_(False)
            self.communication.requires_grad_(False)

    def forward(self, x, return_aux=False, *, modality_mask=None):
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], 4, device=x.device, dtype=torch.bool)
        x, mask = apply_modality_mask(x, modality_mask)
        joint_outputs = self.decoder(self.encoder(x))
        joint = joint_outputs[0] if isinstance(joint_outputs, (tuple, list)) else joint_outputs
        active = bool(self.semantic.fusion_active)
        if not return_aux and not active:
            return joint_outputs
        communicate = bool(self.communication_active)
        expert_mask = self.semantic.expert_mask(mask)
        local, states = self.semantic.local_predictions(x, joint, mask, communicate)
        refined, attention = self.communication(local, states, expert_mask) if communicate else (local, None)
        weights, beta = self.semantic.weights(x, mask), self.semantic.beta(x, mask)
        moe = self.semantic.fuse(refined, weights)
        blend = beta[:, :, None, None, None]
        final = (1-blend)*joint.float() + blend*moe if active else joint
        output = [final, *joint_outputs[1:]] if isinstance(joint_outputs, (list, tuple)) else final
        if not return_aux:
            return output
        return dict(output=output, joint_outputs=joint_outputs, final=final, joint=joint,
                    local_experts=local, experts=refined, states=states, moe=moe, weights=weights,
                    beta=beta, attention=attention, modality_mask=mask, expert_mask=expert_mask,
                    expert_names=EXPERT_NAMES, active=active,
                    message_gate=self.communication.last_gate if communicate else None)


def build_generalist_network(configuration_manager, num_input_channels=4, num_output_channels=3,
                             enable_deep_supervision=True, **options):
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSGeneralistAgentUNet(backbone, configuration_manager, **options)
