"""Preserve the best LegacyV2 training recipe for a fifth-expert size control."""
import hashlib
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300,
)
from .network import build_fullsize_generalist_legacy_network


class BraTSFullSizeGeneralistLegacyTrainer300(nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300):
    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_fullsize_generalist_legacy_network(
            configuration_manager, num_input_channels, num_output_channels,
            enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        registry = (Path(__file__).resolve().parents[4] /
                    'training/nnUNetTrainer/variants/brats2021_audit/'
                    'nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300.py')
        self._v2_source_hashes['fullsize_generalist_trainer_registry'] = hashlib.sha256(
            registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=5, architecture='agent_v2_generalist_fullsize_legacy')
        kwargs = self.configuration_manager.network_arch_init_kwargs
        contract['fullsize_generalist'] = {
            'network_class': self.configuration_manager.network_arch_class_name,
            'features_per_stage': kwargs['features_per_stage'],
            'n_conv_per_stage': kwargs['n_conv_per_stage'],
            'n_conv_per_stage_decoder': kwargs['n_conv_per_stage_decoder'],
            'input_channels': 8, 'joint_input_channels': 4,
            'same_joint_topology_except_input_channels': True,
            'joint_weights_shared_or_copied': False,
            'initializer': 'official PlainConvUNet.initialize via get_network_from_plans',
            'expert_deep_supervision': False,
            'unused_coarse_heads_retained_and_frozen': True,
            'communication_input_channels': self.network.semantic.experts[4].feature_channels,
            'fusion_order': 'expert communication -> expert logit mixture -> joint logit blend',
            'loss_or_schedule_change': False,
            'scope': 'fifth-expert architecture/capacity, not a pure width-only ablation',
        }
        return contract
