"""300 epochs, original GeneralistLegacy loss and sampling, feature path added."""
import hashlib
from pathlib import Path

from ..generalist.trainer import BraTSGeneralistTrainer300
from .network import build_feature_injection_network
from .diagnostics import feature_summary


class BraTSGeneralistFeatureTrainer300(BraTSGeneralistTrainer300):
    output_gate = False
    include_self = False

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_feature_injection_network(
            configuration_manager, num_input_channels, num_output_channels,
            enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        registry = (Path(__file__).resolve().parents[4] /
                    'training/nnUNetTrainer/variants/brats2021_audit/nnUNetTrainer_BraTSAgentGeneralistFeature300.py')
        self._v2_source_hashes['feature_injection_trainer_registry'] = hashlib.sha256(registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=4, architecture='agent_v2_generalist_feature_injection')
        contract['feature_injection'] = dict(
            start_epoch=50, feature_level=self.network.semantic.feature_level,
            decoder_stage=self.network.decoder.injection_stage,
            embed_dim=48, heads=3, attention='torch.nn.MultiheadAttention, per voxel over available experts',
            head_semantics='latent, not region-labelled', dropout=0.,
            residual='fixed coefficient 1; output projection initialized to zero',
            loss_change=False, output_fusion_change=False, decoder_passes=1)
        return contract

    def collect_step_diagnostics(self, parts):
        record = super().collect_step_diagnostics(parts)
        record['feature_injection'] = feature_summary(self.network, parts)
        return record
