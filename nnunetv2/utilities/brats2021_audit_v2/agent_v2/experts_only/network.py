"""Remove J entirely; retain the four specialists, small G and Legacy messages."""
import torch
from torch import nn

from ...agent_v1.blocks import MissingSemanticExperts3D, ModalityCommunication3D
from ...missing_modality import apply_modality_mask
from ..blocks import ModalityCommunicationV2
from ..generalist.blocks import GeneralistExperts3D, GeneralistCommunication3D, EXPERT_NAMES


class FiveSmallExperts3D(GeneralistExperts3D):
    """Existing expert inputs/router, without a trainable or stored beta."""

    def __init__(self, kernel_sizes, strides, patch_size, *, uniform=False):
        physical = MissingSemanticExperts3D(kernel_sizes, strides, patch_size,
                                             uniform=uniform, conditional_beta=False)
        super().__init__(physical, kernel_sizes, strides)
        del self.beta_logit

    def set_joint_stage(self, active):
        # The inherited name means learned routing is enabled at epoch 50;
        # this module has no joint branch and no beta to enable.
        self.fusion_active.fill_(bool(active))
        self.router.requires_grad_(bool(active) and not self.uniform)

    def beta(self, x, physical_mask):
        # Compatibility for existing routing CSVs: expert fraction=1, J=0.
        # These ones are metadata, never a parameter or a blending operation.
        return torch.ones((x.shape[0], 3), device=x.device, dtype=torch.float32)


class BraTSFiveSmallExpertsLegacyUNet(nn.Module):
    """Exactly five small UNets. All stages output the expert logit mixture.

    0..49: uniform available-expert mixture; no communication.
    50..99: learned masked routing; no communication.
    100..299: same routing plus one peer-only, ungated residual message round.
    Every segmentation branch returns only its highest-resolution prediction.
    """

    def __init__(self, configuration_manager, *, use_experts=True, communication=True,
                 local_only=False, lomix=False, uniform=False, conditional_beta=False,
                 residual_mode='fixed_identity', null_peer=False,
                 correction_enabled=True, output_gate=False, include_self=False):
        super().__init__()
        if (not use_experts or not communication or lomix or conditional_beta or null_peer
                or residual_mode != 'fixed_identity' or not correction_enabled
                or output_gate or include_self or local_only):
            raise ValueError('ExpertsOnlyLegacy requires five experts, learned peer-only ungated '
                             'LegacyV2 communication; no J, LoMix, null peer or conditional beta')
        kwargs = configuration_manager.network_arch_init_kwargs
        self.semantic = FiveSmallExperts3D(kwargs['kernel_sizes'], kwargs['strides'],
                                           configuration_manager.patch_size, uniform=uniform)
        original = ModalityCommunication3D(self.semantic.feature_channels)
        fixed = ModalityCommunicationV2(original, residual_mode='fixed_identity', null_peer=False)
        self.communication = GeneralistCommunication3D(
            fixed, self.semantic.feature_channels, output_gate=False, include_self=False)
        self.expert_names = EXPERT_NAMES
        # These optional-module attributes are required by shared trainer hooks.
        # There is intentionally no encoder, decoder or joint module/alias.
        self.moe = None
        self.lomix = None
        self.deep_supervision = False
        self.register_buffer('initialization_seed', torch.tensor(
            torch.initial_seed() % (2**63 - 1), dtype=torch.int64))
        self.register_buffer('stage_epoch', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('communication_active', torch.tensor(False))
        self.register_buffer('mix_coefficient', torch.tensor(0.))
        self.register_buffer('architecture_version', torch.tensor(7, dtype=torch.int64))
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.stage_epoch.fill_(int(epoch))
        self.semantic.set_joint_stage(int(epoch) >= 50)
        self.communication_active.fill_(int(epoch) >= 100)
        self.communication.requires_grad_(bool(self.communication_active))
        # Legacy's disabled output gate is retained structurally, never trained.
        self.communication.gate.requires_grad_(False)

    def restore_stage(self):
        self.set_epoch(int(self.stage_epoch))

    def forward(self, x, return_aux=False, *, modality_mask=None):
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], 4, device=x.device, dtype=torch.bool)
        x, mask = apply_modality_mask(x, modality_mask)
        # Only a shape template for skipped physical experts; no forward or
        # prediction is computed outside the five small expert modules.
        template = x.new_empty((x.shape[0], 3, *x.shape[2:]))
        communicate = bool(self.communication_active)
        local, states = self.semantic.local_predictions(x, template, mask, communicate)
        expert_mask = self.semantic.expert_mask(mask)
        refined, attention = (self.communication(local, states, expert_mask)
                              if communicate else (local, None))
        weights = self.semantic.weights(x, mask)
        final = self.semantic.fuse(refined, weights)
        if not return_aux:
            return final
        return dict(output=final, final=final, moe=final,
                    local_experts=local, experts=refined, states=states,
                    weights=weights, beta=self.semantic.beta(x, mask),
                    attention=attention, modality_mask=mask, expert_mask=expert_mask,
                    expert_names=EXPERT_NAMES, active=bool(self.semantic.fusion_active),
                    joint_present=False, fusion_kind='five_small_experts_only',
                    message_gate=None)


def build_five_small_experts_legacy_network(configuration_manager, num_input_channels=4,
                                            num_output_channels=3,
                                            enable_deep_supervision=False, **options):
    if num_input_channels != 4 or num_output_channels != 3:
        raise ValueError('Expected four physical MRI slots and three WT/TC/ET heads')
    # nnU-Net passes this factory argument for every architecture. This ablation
    # always has a tensor output; its trainer disables DS in losses/transforms.
    return BraTSFiveSmallExpertsLegacyUNet(configuration_manager, **options)
