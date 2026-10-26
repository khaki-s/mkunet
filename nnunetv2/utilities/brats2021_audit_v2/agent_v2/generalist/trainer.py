"""300-epoch controls, changing only explicitly registered architecture options."""
import hashlib
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from ..trainer import BraTSAgentV2Trainer300
from ..losses import refined_weight
from ...agent_v1.losses import positive_region_loss
from ...diagnostics import preserved_rng, patch_metrics
from ...missing_modality import MASK_CODES, mask_from_code
from .network import build_generalist_network
from .blocks import EXPERT_NAMES
from .losses import observed_expert_loss
from .diagnostics import GeneralistRoutingAccumulator, generalist_summary, generalist_interventions


class BraTSGeneralistTrainer300(BraTSAgentV2Trainer300):
    correction_enabled = True
    output_gate = True
    include_self = True

    @classmethod
    def network_options(cls):
        return dict(super().network_options(), correction_enabled=cls.correction_enabled,
                    output_gate=cls.output_gate, include_self=cls.include_self)

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_generalist_network(configuration_manager, num_input_channels, num_output_channels,
                                         enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        self._agent_routes = GeneralistRoutingAccumulator()
        registry = (Path(__file__).resolve().parents[4] /
                    'training/nnUNetTrainer/variants/brats2021_audit/nnUNetTrainer_BraTSAgentGeneralistV2Loss300.py')
        self._v2_source_hashes['generalist_trainer_registry'] = hashlib.sha256(registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=3, architecture='agent_v2_generalist', generalist={
            'physical_modalities': 4, 'expert_names': list(EXPERT_NAMES), 'generalist_input_channels': 8,
            'availability_channels': 'four broadcast binary physical-mask channels',
            'correction_enabled': self.correction_enabled, 'include_self': self.include_self,
            'headwise_output_gate': self.output_gate,
            'local_control': 'self-only attention' if self.include_self else 'zero messages (original LocalV2)',
            'local_loss': 'mean patients, then mean observed specialists plus generalist',
            'new_loss_coefficients': False, 'shared_joint_backbone': False,
            'provenance': 'SimMLM ICCV2025 Dice+BCE; Gated Attention NeurIPS2025 post-SDPA gate; 3D adaptation'})
        return contract

    def compute_agent_loss(self, parts, target):
        main = self.loss(parts['output'], target)
        highest = target[0] if isinstance(target, (tuple, list)) else target
        expert, per_expert = observed_expert_loss(parts['local_experts'], highest,
                                                 parts['expert_mask'], self.region_aux_loss)
        fusion = self.region_aux_loss(parts['moe'].float(), highest) if parts['active'] else main.new_zeros(())
        loss = main + (.25*expert + .25*fusion if parts['active'] else expert)
        coefficient = (refined_weight(int(self.network.stage_epoch), self.refined_aux_max,
                                      ramp_epochs=self.refined_aux_ramp_epochs)
                       if bool(self.network.communication_active) else 0.)
        refined, per_refined = main.new_zeros(()), [None]*5
        if coefficient > 0:
            refined, per_refined = observed_expert_loss(parts['experts'], highest,
                                                       parts['expert_mask'], positive_region_loss)
            loss = loss + coefficient*refined
        # These metrics also become train_step outputs, which nnU-Net collates
        # at epoch end. Keep expert-name metadata in collect_step_diagnostics;
        # tuple metadata is not supported by the epoch metric collator.
        return loss, dict(main=float(main.detach()), expert_mean=float(expert.detach()),
            expert_losses=per_expert, fusion=float(fusion.detach()), mix=0., mix_losses=[], mix_coefficient=0.,
            refined_aux=float(refined.detach()), refined_aux_coefficient=coefficient,
            refined_aux_by_modality=per_refined)

    def collect_step_diagnostics(self, parts):
        return generalist_summary(self.network, parts)

    def on_validation_epoch_end(self, val_outputs):
        # The inherited four-agent branch probes use a Bx4 mask. Keep nnU-Net's
        # validation aggregation, then run the corresponding Bx5 probes here.
        nnUNetTrainer.on_validation_epoch_end(self, val_outputs)
        if not self._agent_probe_every or self._agent_probe is None or (self.current_epoch+1) % self._agent_probe_every:
            return
        with torch.no_grad(), preserved_rng(self.device):
            x, y = (self._agent_probe[k].to(self.device) for k in ('data', 'target'))
            records = {}
            for code in MASK_CODES:
                with torch.autocast(self.device.type, enabled=self.device.type=='cuda'):
                    parts = self.network(x, True, modality_mask=mask_from_code(code, device=self.device)[None])
                    record = {'final': patch_metrics(parts['final'], y),
                              **generalist_summary(self.network, parts, include_gradients=False)}
                    record['local'] = {name: patch_metrics(z, y) for m, (name, z) in enumerate(zip(EXPERT_NAMES, parts['local_experts']))
                                       if parts['expert_mask'][:, m].any()}
                    if code in (1, 3, 15):
                        record.update(generalist_interventions(self.network, parts, y))
                records[str(code)] = record
            self._write_module_record('GENERALIST_15_MASK_PATCH_PROBE', {
                'masks': records, 'expert_names': EXPERT_NAMES, 'scope': 'fixed validation patch, not test100'})
