"""Five-expert variants with a new information path into the joint decoder."""
import torch

from ...baseline.network import build_audit_network
from ...missing_modality import apply_modality_mask
from ..generalist.blocks import EXPERT_NAMES
from ..generalist.network import BraTSGeneralistAgentUNet
from .blocks import ExpertInjectedUNetDecoder, JointExpertCrossAttention3D


class BraTSFeatureInjectedGeneralistUNet(BraTSGeneralistAgentUNet):
    def __init__(self, backbone, configuration_manager, **options):
        super().__init__(backbone, configuration_manager, **options)
        level = self.semantic.feature_level
        self.decoder = ExpertInjectedUNetDecoder(self.decoder, level)
        self.feature_injection = JointExpertCrossAttention3D(
            self.encoder.output_channels[level], self.semantic.feature_channels)
        self.register_buffer('feature_injection_active', torch.tensor(False))
        self.architecture_version.fill_(4)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self, 'feature_injection'):
            self.feature_injection_active.fill_(int(epoch) >= 50)
            self.feature_injection.requires_grad_(int(epoch) >= 50)

    def forward(self, x, return_aux=False, *, modality_mask=None, disable_feature_injection=False):
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], 4, device=x.device, dtype=torch.bool)
        x, mask = apply_modality_mask(x, modality_mask)
        skips = self.encoder(x)
        active = bool(self.semantic.fusion_active)
        if not return_aux and not active:
            return self.decoder(skips)
        communicate = bool(self.communication_active)
        inject = bool(self.feature_injection_active) and not disable_feature_injection
        expert_mask = self.semantic.expert_mask(mask)
        # local_predictions uses its 'joint' argument only as a shape/device
        # template. A view provides that template before the single decoder pass.
        local, states = self.semantic.local_predictions(
            x, x[:, :3], mask, need_features=communicate or inject)
        joint_outputs = self.decoder(
            skips, injector=self.feature_injection if inject else None,
            expert_states=states, expert_mask=expert_mask, capture=return_aux)
        joint = joint_outputs[0] if isinstance(joint_outputs, (tuple, list)) else joint_outputs
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
                    message_gate=self.communication.last_gate if communicate else None,
                    feature_attention=self.feature_injection.last_attention if inject else None,
                    feature_update_relative_rms=(
                        self.feature_injection.last_relative_update_rms if inject else None))


def build_feature_injection_network(configuration_manager, num_input_channels=4,
                                    num_output_channels=3, enable_deep_supervision=True,
                                    **options):
    if num_input_channels != 4 or num_output_channels != 3:
        raise ValueError('Feature-injection models require four physical inputs and WT/TC/ET outputs')
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSFeatureInjectedGeneralistUNet(backbone, configuration_manager, **options)
