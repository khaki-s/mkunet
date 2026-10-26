"""V2 uses the same joint network, experts, router, stages and sparse input API."""
import torch

from ..agent_v1.network import BraTSModalityAgentUNet
from ..baseline.network import build_audit_network
from .blocks import ModalityCommunicationV2


class BraTSAgentV2UNet(BraTSModalityAgentUNet):
    def __init__(self, backbone, configuration_manager, *, residual_mode='fixed_identity',
                 null_peer=False, **options):
        super().__init__(backbone, configuration_manager, **options)
        if self.communication is None:
            raise ValueError('Agent V2 requires communication=True; use SemanticMissing300 for M0')
        self.communication = ModalityCommunicationV2(self.communication, residual_mode=residual_mode,
                                                     null_peer=null_peer)
        self.register_buffer('architecture_version', torch.tensor(2, dtype=torch.int64))
        self.set_epoch(0)

    def forward(self, x, return_aux=False, *, modality_mask=None):
        result = super().forward(x, return_aux, modality_mask=modality_mask)
        if return_aux:
            result['null_attention'] = (self.communication.last_null_attention
                                        if bool(self.communication_active) else None)
        return result


def build_agent_v2_network(configuration_manager, num_input_channels=4, num_output_channels=3,
                           enable_deep_supervision=True, **options):
    backbone = build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSAgentV2UNet(backbone, configuration_manager, **options)
