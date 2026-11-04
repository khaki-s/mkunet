"""Four physical specialists communicate with the actual joint backbone once.

The fifth node has no independent encoder/decoder or trainable image input.
Its logits and decoder state are the same live tensors produced by the joint
forward. Shared tensors are passed as arguments, never registered as modules
twice or detached. Communication/fusion follow the existing LegacyV2 equation.
"""
import torch
from torch import nn

from ...agent_v1.blocks import MissingSemanticExperts3D
from ...baseline.network import build_audit_network
from ...missing_modality import apply_modality_mask
from ..network import BraTSAgentV2UNet
from ..generalist.blocks import GeneralistExperts3D, GeneralistCommunication3D

NODE_NAMES = ('T1', 'T1ce', 'T2', 'FLAIR', 'Joint')


class SharedJointExperts3D(GeneralistExperts3D):
    """Five output nodes backed by exactly four independent expert modules."""
    def __init__(self, original):
        nn.Module.__init__(self)
        for name in ('uniform', 'checkpoint_experts', 'feature_level', 'feature_channels'):
            setattr(self, name, getattr(original, name))
        self.experts = original.experts
        self.router, self.beta_logit = original.router, original.beta_logit
        self.beta_condition = original.beta_condition
        head = self.router.head
        self.router.head = nn.Linear(head.in_features, 15)
        with torch.no_grad():
            self.router.head.weight.zero_()
            self.router.head.bias.zero_()
            self.router.head.weight.reshape(3, 5, -1)[:, :4].copy_(head.weight.reshape(3, 4, -1))
            self.router.head.bias.reshape(3, 5)[:, :4].copy_(head.bias.reshape(3, 4))
        self.register_buffer('fusion_active', original.fusion_active.clone())
        self.set_joint_stage(False)

    def local_predictions(self, x, joint, physical_mask, need_features=False, *, joint_state=None):
        local, states = MissingSemanticExperts3D.local_predictions(
            self, x, joint, physical_mask, need_features)
        if need_features:
            if joint_state is None or joint_state.shape[2:] != states[0].shape[2:]:
                raise ValueError('Joint and specialist communication states need the same spatial grid')
        # Object identity here is intentional. This is not an additional forward.
        return [*local, joint], [*states, joint_state]


class BraTSSharedJointLegacyUNet(BraTSAgentV2UNet):
    def __init__(self, backbone, configuration_manager, *, correction_enabled=True,
                 output_gate=False, include_self=False, **options):
        if not correction_enabled or output_gate or include_self:
            raise ValueError('SharedJointLegacy preserves enabled peer-only, ungated communication')
        if options.get('null_peer', False) or options.get('residual_mode', 'fixed_identity') != 'fixed_identity':
            raise ValueError('SharedJointLegacy requires fixed_identity and no null peer')
        super().__init__(backbone, configuration_manager, **options)
        self.semantic = SharedJointExperts3D(self.semantic)
        self.joint_feature_level = self.semantic.feature_level
        self.joint_feature_stage = len(self.encoder.output_channels) - 2 - self.joint_feature_level
        if not 0 <= self.joint_feature_stage < len(self.decoder.stages):
            raise ValueError('Joint communication state must be an existing decoder level')
        joint_channels = int(self.encoder.output_channels[self.joint_feature_level])
        self.communication = GeneralistCommunication3D(
            self.communication, joint_channels, local_only=options.get('local_only', False),
            output_gate=False, include_self=False)
        self.expert_names = NODE_NAMES
        self.architecture_version.fill_(6)
        self.set_epoch(0)

    def _joint_forward(self, x, need_features):
        if not need_features:
            return self.decoder(self.encoder(x)), None
        captured = []
        handle = self.decoder.stages[self.joint_feature_stage].register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            outputs = self.decoder(self.encoder(x))
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected one joint decoder state from one backbone forward')
        return outputs, captured[0]

    def forward(self, x, return_aux=False, *, modality_mask=None):
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], 4, device=x.device, dtype=torch.bool)
        x, mask = apply_modality_mask(x, modality_mask)
        communicate = bool(self.communication_active)
        joint_outputs, joint_state = self._joint_forward(x, communicate)
        joint = joint_outputs[0] if isinstance(joint_outputs, (list, tuple)) else joint_outputs
        active = bool(self.semantic.fusion_active)
        if not return_aux and not active:
            return joint_outputs
        node_mask = self.semantic.expert_mask(mask)
        local, states = self.semantic.local_predictions(
            x, joint, mask, communicate, joint_state=joint_state)
        refined, attention = self.communication(local, states, node_mask) if communicate else (local, None)
        weights, beta = self.semantic.weights(x, mask), self.semantic.beta(x, mask)
        mixture = self.semantic.fuse(refined, weights)
        blend = beta[:, :, None, None, None]
        final = (1 - blend) * joint.float() + blend * mixture if active else joint
        output = [final, *joint_outputs[1:]] if isinstance(joint_outputs, (list, tuple)) else final
        if not return_aux:
            return output
        return dict(output=output, joint_outputs=joint_outputs, joint=joint, joint_state=joint_state,
                    final=final, local_experts=local, experts=refined, states=states,
                    moe=mixture, weights=weights, beta=beta, attention=attention,
                    modality_mask=mask, expert_mask=node_mask, expert_names=NODE_NAMES,
                    active=active, message_gate=self.communication.last_gate if communicate else None)


def build_shared_joint_legacy_network(configuration_manager, num_input_channels=4, num_output_channels=3,
                                      enable_deep_supervision=True, **options):
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSSharedJointLegacyUNet(backbone, configuration_manager, **options)
