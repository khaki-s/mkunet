"""V2 keeps the 300-epoch V1 training distribution and optimizer unchanged."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..agent_v1.trainer import BraTSModalityAgentTrainer300
from ..diagnostics import preserved_rng
from ..missing_modality import mask_from_code
from .network import build_agent_v2_network
from .losses import refined_auxiliary_loss, refined_weight
from .diagnostics import residual_summary, communication_gradient_summary, no_peer_intervention


class BraTSAgentV2Trainer300(BraTSModalityAgentTrainer300):
    communication = True
    residual_mode = 'fixed_identity'
    null_peer = False
    refined_aux_max = .1
    refined_aux_ramp_epochs = 20

    @classmethod
    def network_options(cls):
        return {**super().network_options(), 'residual_mode': cls.residual_mode, 'null_peer': cls.null_peer}

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_agent_v2_network(configuration_manager, num_input_channels, num_output_channels,
                                      enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get('BRATS_AGENT_INITIAL_WEIGHTS'):
            raise ValueError('V2 controlled ablations start from scratch; unset BRATS_AGENT_INITIAL_WEIGHTS. '
                             'Pretrained adaptation needs a separate budget and migration contract.')
        super().initialize()
        source = Path(__file__).resolve().parents[1]
        self._v2_source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(source.rglob('*.py'))}
        registry = source.parents[1] / 'training/nnUNetTrainer/variants/brats2021_audit/nnUNetTrainer_BraTSAgentV2Loss300.py'
        self._v2_source_hashes['trainer_registry'] = hashlib.sha256(registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=2, architecture='agent_v2', v2={
            'residual_mode': self.residual_mode, 'fixed_residual_scale': .1 if self.residual_mode=='fixed_identity' else None,
            'null_peer': self.null_peer, 'refined_aux_max': self.refined_aux_max,
            'refined_aux_start_epoch': 100, 'refined_aux_ramp_epochs': self.refined_aux_ramp_epochs,
            'loss_normalization': 'mean patients, mean present agents; BCE+(1-Dice)',
            'communication_rounds': 1, 'optimizer_change': False,
            'implementation_sha256': hashlib.sha256(json.dumps(self._v2_source_hashes,sort_keys=True).encode()).hexdigest()})
        return contract

    def compute_agent_loss(self, parts, target):
        loss, components = super().compute_agent_loss(parts, target)
        highest = target[0] if isinstance(target, (tuple,list)) else target
        coefficient = refined_weight(int(self.network.stage_epoch), self.refined_aux_max,
                                     ramp_epochs=self.refined_aux_ramp_epochs)
        coefficient = coefficient if bool(self.network.communication_active) else 0.
        refined = loss.new_zeros(())
        per_modality = [None]*4
        if coefficient > 0:
            refined, per_modality = refined_auxiliary_loss(parts['experts'], highest, parts['modality_mask'])
            loss = loss + coefficient*refined
        components.update(refined_aux=float(refined.detach()), refined_aux_coefficient=coefficient,
                          refined_aux_by_modality=per_modality)
        return loss, components

    def collect_step_diagnostics(self, parts):
        result = super().collect_step_diagnostics(parts)
        result['v2_residual'] = residual_summary(parts)
        result['v2_communication_gradients'] = communication_gradient_summary(self.network.communication)
        result['residual_scale_trainable'] = self.residual_mode == 'legacy'
        return result

    def on_train_start(self):
        super().on_train_start()
        (Path(self.output_folder)/'agent_v2_source_hashes.json').write_text(json.dumps(self._v2_source_hashes,indent=2)+'\n')

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._write_module_record('AGENT_V2_LOSS', {k: float(np.mean([r[k] for r in train_outputs]))
                                                   for k in ('refined_aux','refined_aux_coefficient')})

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        if (not self._agent_probe_every or self._agent_probe is None
                or (self.current_epoch+1) % self._agent_probe_every
                or not bool(self.network.communication_active)):
            return
        with torch.no_grad(), preserved_rng(self.device):
            x, y = (self._agent_probe[k].to(self.device) for k in ('data','target'))
            records = {}
            # Representative k=1,2,4 probes supplement the inherited all15 probe.
            for code in (1,3,15):
                with torch.autocast(self.device.type, enabled=self.device.type=='cuda'):
                    parts = self.network(x, True, modality_mask=mask_from_code(code, device=self.device)[None])
                    records[str(code)] = {'residual': residual_summary(parts),
                                          'no_peer': no_peer_intervention(self.network, parts, y)}
            self._write_module_record('AGENT_V2_PEER_PROBE', {'masks':records,
                'scope':'fixed validation patch, no-peer intervention; not test100 or full-case accuracy'})
