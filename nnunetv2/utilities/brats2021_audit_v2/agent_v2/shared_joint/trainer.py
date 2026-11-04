"""Legacy300 recipe with the raw joint branch counted once in local supervision."""
import hashlib
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300,
)
from ...agent_v1.losses import positive_region_loss
from ...diagnostics import preserved_rng, patch_metrics
from ...missing_modality import MASK_CODES, mask_from_code
from ..generalist.losses import observed_expert_loss
from ..losses import refined_weight
from .network import NODE_NAMES, build_shared_joint_legacy_network
from .diagnostics import SharedJointRoutingAccumulator, shared_joint_summary, shared_joint_interventions


class BraTSSharedJointLegacyTrainer300(nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300):
    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_shared_joint_legacy_network(configuration_manager, num_input_channels,
            num_output_channels, enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        self._agent_routes = SharedJointRoutingAccumulator()
        registry = (Path(__file__).resolve().parents[4] /
                    'training/nnUNetTrainer/variants/brats2021_audit/'
                    'nnUNetTrainer_BraTSAgentSharedJointLegacyV2Loss300.py')
        self._v2_source_hashes['shared_joint_trainer_registry'] = hashlib.sha256(registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=6, architecture='agent_v2_shared_joint_legacy')
        contract['generalist'] = {
            'physical_modalities': 4, 'expert_names': list(NODE_NAMES),
            'independent_expert_networks': 4, 'shared_joint_backbone': True,
            'node_5': 'same live joint logits and decoder state; no independent fifth network',
            'generalist_input_channels': 4, 'availability_channels': 'no extra MRI channels; explicit mask is retained by data/routing API',
            'correction_enabled': True, 'include_self': False, 'headwise_output_gate': False,
            'local_loss': 'mean patients, mean observed FOUR specialists; raw joint already covered by main',
            'refined_loss': 'mean patients, mean observed FIVE communication outputs including refined Joint',
            'coefficients_changed': False, 'communication_rounds': 1,
            'joint_forward_calls_per_network_forward': 1,
            'fusion': '(1-beta)*raw_joint + beta*weighted_five_communicated_nodes',
            'joint_node_detached': False, 'joint_registered_as_extra_expert': False,
            'official_decoder_encoder_aliases': 'inherited unchanged; optimizer parameter objects remain unique',
            'provenance': 'reuse LegacyV2 QKV and loss code; new shared-node architecture adaptation',
        }
        return contract

    def compute_agent_loss(self, parts, target):
        main = self.loss(parts['output'], target)
        highest = target[0] if isinstance(target, (list, tuple)) else target
        # Raw Joint is NOT independently trained a second time as expert five.
        expert, local_losses = observed_expert_loss(parts['local_experts'][:4], highest,
                                                    parts['modality_mask'], self.region_aux_loss)
        fusion = self.region_aux_loss(parts['moe'].float(), highest) if parts['active'] else main.new_zeros(())
        loss = main + (.25 * expert + .25 * fusion if parts['active'] else expert)
        coefficient = (refined_weight(int(self.network.stage_epoch), self.refined_aux_max,
                                      ramp_epochs=self.refined_aux_ramp_epochs)
                       if bool(self.network.communication_active) else 0.)
        refined, refined_losses = main.new_zeros(()), [None] * 5
        if coefficient > 0:
            refined, refined_losses = observed_expert_loss(parts['experts'], highest,
                                                           parts['expert_mask'], positive_region_loss)
            loss = loss + coefficient * refined
        return loss, dict(main=float(main.detach()), expert_mean=float(expert.detach()),
            expert_losses=[*local_losses, None], fusion=float(fusion.detach()), mix=0.,
            mix_losses=[], mix_coefficient=0., refined_aux=float(refined.detach()),
            refined_aux_coefficient=coefficient, refined_aux_by_modality=refined_losses)

    def collect_step_diagnostics(self, parts):
        return shared_joint_summary(self.network, parts)

    def on_validation_epoch_end(self, val_outputs):
        nnUNetTrainer.on_validation_epoch_end(self, val_outputs)
        if not self._agent_probe_every or self._agent_probe is None or (self.current_epoch + 1) % self._agent_probe_every:
            return
        with torch.no_grad(), preserved_rng(self.device):
            x, y = (self._agent_probe[key].to(self.device) for key in ('data', 'target'))
            records = {}
            for code in MASK_CODES:
                with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                    parts = self.network(x, True, modality_mask=mask_from_code(code, device=self.device)[None])
                    record = {'final': patch_metrics(parts['final'], y),
                              **shared_joint_summary(self.network, parts, include_gradients=False)}
                    record['local'] = {name: patch_metrics(z, y) for index, (name, z) in
                                       enumerate(zip(NODE_NAMES, parts['local_experts']))
                                       if parts['expert_mask'][:, index].any()}
                    if code in (1, 3, 15):
                        record.update(shared_joint_interventions(self.network, parts, y))
                records[str(code)] = record
            self._write_module_record('SHARED_JOINT_15_MASK_PATCH_PROBE', {
                'masks': records, 'expert_names': NODE_NAMES, 'scope': 'fixed validation patch, not test100'})
