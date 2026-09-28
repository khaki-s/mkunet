"""300-epoch modality-dropout/agent ablations with explicit phase and resume."""
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .agent_diagnostics import RoutingAccumulator, gradient_summary, output_summary, branch_interventions
from .agent_losses import present_expert_loss
from .agent_network import build_agent_network
from .diagnostics import patch_metrics, preserved_rng
from .missing_modality import MASK_CODES, mask_from_code, sample_masks, validate_mask
from .reproducibility import configure_determinism
from .trainer import BraTSAuditTrainer300


class BraTSModalityAgentTrainer300(BraTSAuditTrainer300):
    mode = 'modality_agent'
    exclusive_loss = True
    use_experts = True
    communication = False
    local_only = False
    lomix = False
    uniform_fusion = False
    conditional_beta = False
    mask_policy = 'cardinality'

    @classmethod
    def network_options(cls):
        return dict(use_experts=cls.use_experts, communication=cls.communication, local_only=cls.local_only,
                    lomix=cls.lomix, uniform=cls.uniform_fusion, conditional_beta=cls.conditional_beta)

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_agent_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        loss = self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss
        self.region_aux_loss = loss.base_loss if self.exclusive_loss else loss
        self._agent_steps = self.current_epoch * self.num_iterations_per_epoch
        self._agent_routes = RoutingAccumulator()
        self._agent_probe = None
        self._agent_diag_every = int(os.environ.get('BRATS_DIAG_EVERY', '50'))
        self._agent_probe_every = int(os.environ.get('BRATS_ABLATE_EVERY', '25'))
        if min(self._agent_diag_every, self._agent_probe_every) < 0:
            raise ValueError('Diagnostic intervals must be nonnegative')
        self.network.set_epoch(self.current_epoch)
        self._agent_initialization = {'kind': 'from_scratch'}
        encode = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        split_file = Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'
        self._agent_resume_signature = {
            'fold': self.fold, 'batch_size': self.batch_size,
            'patch_size': list(self.configuration_manager.patch_size),
            'plans_sha256': encode(self.plans_manager.plans), 'dataset_json_sha256': encode(self.dataset_json),
            'splits_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest() if split_file.exists() else None}
        weights_path = os.environ.get('BRATS_AGENT_INITIAL_WEIGHTS')
        if weights_path:
            state = torch.load(weights_path, map_location=self.device, weights_only=False)
            metadata = state.get('migration', {})
            if metadata.get('kind') != 'semantic_weights_only' or metadata.get('target_trainer') != type(self).__name__:
                raise ValueError('Initial weights must be an explicit migration for this trainer')
            if metadata.get('seed') != self.audit_seed or metadata.get('target_options') != self.network_options():
                raise ValueError('Migration seed or target options differ')
            self.network.load_state_dict(state['network_weights'], strict=True)
            self.network.set_epoch(self.current_epoch)
            self._agent_initialization = metadata

    def training_contract(self):
        return {'version': 1, 'seed': self.audit_seed, 'data_and_plans': self._agent_resume_signature, 'mask_policy': self.mask_policy,
                'mask_rng': 'sha256_per_logical_sample_v1', 'epochs': 300,
                'iterations_per_epoch': self.num_iterations_per_epoch, 'options': self.network_options(),
                'exclusive_loss': self.exclusive_loss, 'initialization': self._agent_initialization, 'phase_epochs': [0, 50, 100],
                'online_validation': 'complete-input patches; all15 probe is diagnostic only',
                'actual_validation': 'all15 controlled_preprocessed, fixed checkpoint'}

    def on_train_start(self):
        super().on_train_start()
        self.network.set_epoch(self.current_epoch)
        if self._agent_steps != self.current_epoch * self.num_iterations_per_epoch:
            raise ValueError('Only epoch-boundary resume is supported by the nnU-Net data stream')
        (Path(self.output_folder) / 'agent_experiment.json').write_text(
            json.dumps(self.training_contract(), indent=2) + '\n')
        probe = Path(self.output_folder) / 'agent_validation_probe.pt'
        if self._agent_probe_every and probe.exists():
            self._agent_probe = torch.load(probe, map_location='cpu', weights_only=False)
            if self._agent_probe['seed'] != self.audit_seed:
                raise ValueError('Diagnostic probe seed mismatch')

    def on_train_epoch_start(self):
        self.network.set_epoch(self.current_epoch)
        super().on_train_epoch_start()
        self._write_module_record('AGENT_PHASE', {'stage_epoch': int(self.network.stage_epoch),
            'dropout_active': self.current_epoch >= 50, 'communication_active': bool(self.network.communication_active),
            'mix_coefficient': float(self.network.mix_coefficient), 'next_mask_step': self._agent_steps})

    def compute_agent_loss(self, parts, target):
        main = self.loss(parts['output'], target)
        highest = target[0] if isinstance(target, (list, tuple)) else target
        expert_loss, fusion, mix = main.new_zeros(()), main.new_zeros(()), main.new_zeros(())
        experts, mix_losses = [None] * 4, []
        if self.use_experts:
            expert_loss, experts = present_expert_loss(parts['local_experts'], highest,
                                                      parts['modality_mask'], self.region_aux_loss)
            if parts['active']:
                fusion = self.region_aux_loss(parts['moe'].float(), highest)
        coefficient = float(self.network.mix_coefficient)
        if coefficient > 0:
            mix, components = self.network.lomix(parts['joint_outputs'], highest)
            mix_losses = components.cpu().tolist()
        loss = main + (.25 * expert_loss + .25 * fusion if parts['active'] else expert_loss) + coefficient * mix
        return loss, {'main': float(main.detach()), 'expert_mean': float(expert_loss.detach()),
                      'expert_losses': experts, 'fusion': float(fusion.detach()),
                      'mix': float(mix.detach()), 'mix_losses': mix_losses, 'mix_coefficient': coefficient}

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, (list, tuple)) else target.to(self.device, non_blocking=True)
        physical = batch.get('modality_mask')
        physical = (torch.ones(data.shape[0], 4, dtype=torch.bool, device=self.device) if physical is None
                    else validate_mask(physical, data.shape[0], device=self.device))
        mask = (physical if self.current_epoch < 50 else sample_masks(self.audit_seed, self._agent_steps,
                data.shape[0], policy=self.mask_policy, available=physical, device=self.device))
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            parts = self.network(data, return_aux=True, modality_mask=mask)
            loss, losses = self.compute_agent_loss(parts, target)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f'Nonfinite agent loss at step {self._agent_steps}')
        old_scale = self.grad_scaler.get_scale() if self.grad_scaler is not None else None
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        sampled = self._agent_diag_every and self._agent_steps % self._agent_diag_every == 0
        diagnostic = {'gradients': gradient_summary(self.network, mask), **output_summary(parts)} if sampled else None
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        if sampled:
            if self.network.lomix is not None:
                diagnostic['lomix_weights'] = {'alpha': [a.detach().float().softmax(-1).cpu().tolist() for a in self.network.lomix.alpha],
                                              'omega': self.network.lomix.loss_weights().detach().cpu().tolist()}
            if self.network.communication is not None:
                diagnostic['communication_rho'] = self.network.communication.rho.detach().tanh().cpu().tolist()
            diagnostic.update(losses=losses, logical_step=self._agent_steps,
                              amp_step_skipped=old_scale is not None and self.grad_scaler.get_scale() < old_scale)
            self._write_module_record('AGENT_STEP', diagnostic)
        self._agent_routes.add(parts)
        self._agent_steps += 1
        return {'loss': loss.detach().cpu().numpy(), **losses}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._write_module_record('AGENT_ROUTING_BY_MASK', {'masks': self._agent_routes.finish()})
        self._write_module_record('AGENT_LOSS', {k: float(np.mean([r[k] for r in train_outputs]))
                                                for k in ('main', 'expert_mean', 'fusion', 'mix', 'mix_coefficient')})

    def validation_step(self, batch):
        if self._agent_probe_every and self._agent_probe is None:
            target = batch['target'][0] if isinstance(batch['target'], (list, tuple)) else batch['target']
            self._agent_probe = {'data': batch['data'][:1].detach().cpu().clone(),
                                 'target': target[:1].detach().cpu().clone(), 'seed': self.audit_seed}
            torch.save(self._agent_probe, Path(self.output_folder) / 'agent_validation_probe.pt')
        # Keep the inherited green curve's complete-input meaning explicit.
        return nnUNetTrainer.validation_step(self, batch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        if not self._agent_probe_every or self._agent_probe is None or (self.current_epoch + 1) % self._agent_probe_every:
            return
        with torch.no_grad(), preserved_rng(self.device):
            x, y = (self._agent_probe[k].to(self.device) for k in ('data', 'target'))
            records = {}
            for code in MASK_CODES:
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    parts = self.network(x, True, modality_mask=mask_from_code(code, device=self.device)[None])
                final = parts['output'][0] if isinstance(parts['output'], (list, tuple)) else parts['output']
                record = {'final': patch_metrics(final, y), **output_summary(parts)}
                if self.use_experts:
                    record['local'] = {str(m): patch_metrics(z, y) for m, z in enumerate(parts['local_experts']) if (code >> m) & 1}
                    record['refined'] = {str(m): patch_metrics(z, y) for m, z in enumerate(parts['experts']) if (code >> m) & 1}
                if self.use_experts and code == 15:
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                        record['branch_interventions'] = branch_interventions(self.network, parts, y)
                records[str(code)] = record
            self._write_module_record('AGENT_15_MASK_PATCH_PROBE', {'masks': records, 'scope': 'fixed validation patch, not full-case metrics'})

    def save_checkpoint(self, filename):
        if self.disable_checkpointing or self.local_rank != 0:
            return
        state = {'network_weights': self.network.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
                 'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                 'logging': self.logger.get_checkpoint(), '_best_ema': self._best_ema,
                 'current_epoch': self.current_epoch + 1, 'init_args': self.my_init_kwargs,
                 'trainer_name': type(self).__name__, 'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                 'agent_training': self.training_contract(), 'next_mask_step': self._agent_steps}
        path = Path(filename)
        torch.save(state, str(path) + '.tmp')
        os.replace(str(path) + '.tmp', path)

    def load_checkpoint(self, filename_or_checkpoint):
        if not self.was_initialized:
            self.initialize()
        state = (torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
                 if isinstance(filename_or_checkpoint, (str, Path)) else filename_or_checkpoint)
        self._agent_initialization = state.get('agent_training', {}).get('initialization', {'kind': 'from_scratch'})
        if state['trainer_name'] != type(self).__name__ or state.get('agent_training') != self.training_contract():
            raise ValueError('Trainer/seed/sampling contract differs; use explicit migration, not --c')
        weights = {k.removeprefix('module.'): v for k, v in state['network_weights'].items()}
        if int(weights['initialization_seed']) != self.audit_seed:
            raise ValueError('Initialization seed mismatch')
        self.network.load_state_dict(weights, strict=True)
        self.network.restore_stage()
        self.optimizer.load_state_dict(state['optimizer_state'])
        if self.grad_scaler is not None and state.get('grad_scaler_state') is not None:
            self.grad_scaler.load_state_dict(state['grad_scaler_state'])
        self.my_init_kwargs, self.current_epoch = state['init_args'], state['current_epoch']
        self.logger.load_checkpoint(state['logging'])
        self._best_ema = state['_best_ema']
        self.inference_allowed_mirroring_axes = state.get('inference_allowed_mirroring_axes')
        self._agent_steps = int(state['next_mask_step'])
        if self._agent_steps != self.current_epoch * self.num_iterations_per_epoch:
            raise ValueError('Checkpoint is not at an epoch boundary; cannot align augmentation and mask streams')
        self.lr_scheduler.step(self.current_epoch)

    def perform_actual_validation(self, save_probabilities=False):
        from .agent_evaluate import evaluate_preprocessed
        configure_determinism()
        _, keys = self.do_split()
        old_ds, old_training = self.network.decoder.deep_supervision, self.network.training
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        try:
            return evaluate_preprocessed(self.network, self.plans_manager, self.configuration_manager,
                self.dataset_json, self.dataset_class, self.preprocessed_dataset_folder, keys,
                Path(self.preprocessed_dataset_folder_base) / 'gt_segmentations',
                Path(self.output_folder) / 'validation_15m', device=self.device,
                mirror_axes=self.inference_allowed_mirroring_axes, save_probabilities=save_probabilities,
                provenance={'trainer': type(self).__name__, 'stage_epoch': int(self.network.stage_epoch)})
        finally:
            self.set_deep_supervision_enabled(old_ds)
            self.network.train(old_training)
