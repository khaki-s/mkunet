"""nnU-Net + explicit missingness + optional communication and LoMix."""
import torch
from torch import nn

from .agent_blocks import MissingSemanticExperts3D, ModalityCommunication3D
from .agent_losses import LoMixSupervision3D
from .missing_modality import apply_modality_mask
from .network import build_audit_network


class BraTSModalityAgentUNet(nn.Module):
    def __init__(self, backbone, configuration_manager, *, use_experts=True,
                 communication=False, local_only=False, lomix=False, uniform=False,
                 conditional_beta=False):
        super().__init__()
        if (communication or local_only or uniform or conditional_beta) and not use_experts:
            raise ValueError('Agent options require expert branches')
        if local_only and not communication:
            raise ValueError('local_only requires communication module for matched parameters')
        self.encoder, self.decoder, self.moe = backbone.encoder, backbone.decoder, None
        self.semantic = None
        if use_experts:
            kwargs = configuration_manager.network_arch_init_kwargs
            self.semantic = MissingSemanticExperts3D(kwargs['kernel_sizes'], kwargs['strides'],
                                                     configuration_manager.patch_size, uniform, conditional_beta)
        self.communication = (ModalityCommunication3D(self.semantic.feature_channels, local_only=local_only)
                              if communication else None)
        self.lomix = LoMixSupervision3D() if lomix else None
        self.register_buffer('initialization_seed', torch.tensor(torch.initial_seed() % (2**63 - 1), dtype=torch.int64))
        self.register_buffer('stage_epoch', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('communication_active', torch.tensor(False))
        self.register_buffer('mix_coefficient', torch.tensor(0.))
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.stage_epoch.fill_(int(epoch))
        if self.semantic is not None:
            self.semantic.set_joint_stage(epoch >= 50)
        self.communication_active.fill_(self.communication is not None and epoch >= 100)
        if self.communication is not None:
            self.communication.requires_grad_(bool(self.communication_active.item()))
        self.mix_coefficient.fill_(.1 * min(1., max(0., (epoch - 100) / 19.)) if self.lomix is not None else 0.)
        if self.lomix is not None:
            self.lomix.requires_grad_(epoch >= 100)

    def restore_stage(self):
        """Restore requires_grad flags after state_dict load, preserving saved epoch."""
        self.set_epoch(int(self.stage_epoch.item()))

    def forward(self, x, return_aux=False, *, modality_mask=None):
        # Existing nnU-Net calls with four complete inputs remain valid. All
        # missing-input entry points MUST pass the explicit physical mask.
        if modality_mask is None:
            modality_mask = torch.ones(x.shape[0], 4, dtype=torch.bool, device=x.device)
        x, mask = apply_modality_mask(x, modality_mask)
        joint_outputs = self.decoder(self.encoder(x))
        joint = joint_outputs[0] if isinstance(joint_outputs, (list, tuple)) else joint_outputs
        if self.semantic is None:
            return {'output': joint_outputs, 'joint_outputs': joint_outputs, 'joint': joint,
                    'modality_mask': mask, 'active': False} if return_aux else joint_outputs
        active = bool(self.semantic.fusion_active.item())
        if not return_aux and not active:
            return joint_outputs
        communicate = bool(self.communication_active.item())
        local, states = self.semantic.local_predictions(x, joint, mask, communicate)
        refined, attention = (self.communication(local, states, mask) if communicate else (local, None))
        weights, beta = self.semantic.weights(x, mask), self.semantic.beta(x, mask)
        moe = self.semantic.fuse(refined, weights)
        blend = beta[:, :, None, None, None]
        final = (1 - blend) * joint.float() + blend * moe if active else joint
        output = [final, *joint_outputs[1:]] if isinstance(joint_outputs, (list, tuple)) else final
        if not return_aux:
            return output
        return {'output': output, 'joint_outputs': joint_outputs, 'final': final, 'joint': joint,
                'local_experts': local, 'experts': refined, 'states': states, 'moe': moe, 'weights': weights,
                'beta': beta, 'attention': attention, 'modality_mask': mask, 'active': active}


def build_agent_network(configuration_manager, num_input_channels=4, num_output_channels=3,
                        enable_deep_supervision=True, **options):
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSModalityAgentUNet(backbone, configuration_manager, **options)
