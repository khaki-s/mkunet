"""300 epochs: 50 independent-expert warmup + 250 semantic fusion.

Final and original low-resolution outputs retain ExclusiveHierarchyLoss with
the original deep-supervision weights. Auxiliary outputs use region Dice+BCE.
The uniform control differs only in fixing router weights to 1/4.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from ..diagnostics import preserved_rng
from ..reproducibility import state_digest
from .blocks import EXPERT_WIDTHS, MODALITIES
from .diagnostics import SemanticDiagnostics, probe_digest, routing_summary, semantic_probe
from .network import build_semantic_network
from ..baseline.trainer import BraTSAuditTrainer300


class BraTSSemanticMoETrainer300(BraTSAuditTrainer300):
    mode = 'semantic_moe'
    exclusive_loss = True
    uniform_fusion = False
    warmup_epochs = 50
    expert_aux_weight = .25
    fusion_aux_weight = .25

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_semantic_network(configuration_manager, num_input_channels, num_output_channels,
                                       enable_deep_supervision, cls.uniform_fusion)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        # Reuse the very same region-loss definition inside ExclusiveHierarchyLoss.
        final_loss = self.loss.loss if isinstance(self.loss, DeepSupervisionWrapper) else self.loss
        self.region_aux_loss = final_loss.base_loss
        self.network.set_epoch(self.current_epoch, self.warmup_epochs)
        self._semantic_monitor = None
        self._semantic_steps = 0
        self._route_batches = []
        self._semantic_probe = None
        self._semantic_diag_every = int(os.environ.get('BRATS_DIAG_EVERY', '50'))
        self._semantic_ablate_every = int(os.environ.get('BRATS_ABLATE_EVERY', '25'))
        if min(self._semantic_diag_every, self._semantic_ablate_every) < 0:
            raise ValueError('Diagnostic intervals must be nonnegative; 0 disables')
        self._expert_initial_sha256 = [state_digest(expert) for expert in self.network.semantic.experts]

    def on_train_start(self):
        super().on_train_start()
        self.network.set_epoch(self.current_epoch, self.warmup_epochs)
        self._semantic_steps = self.current_epoch * self.num_iterations_per_epoch
        source = Path(__file__).resolve().parents[1] / 'vendor/simmlm_provenance.json'
        record = {'trainer': type(self).__name__, 'epochs': self.num_epochs, 'warmup_epochs': self.warmup_epochs,
                  'seed': self.audit_seed, 'backbone_initial_sha256': self.backbone_initial_sha256,
                  'expert_initial_sha256': self._expert_initial_sha256, 'expert_widths': EXPERT_WIDTHS,
                  'region_order': ['WT', 'TC', 'ET'], 'modality_order': MODALITIES,
                  'uniform_fusion': self.uniform_fusion, 'warmup_expert_loss_weight': 1.,
                  'joint_expert_loss_weight': self.expert_aux_weight,
                  'joint_fusion_loss_weight': self.fusion_aux_weight,
                  'expert_activation_checkpointing': self.network.semantic.checkpoint_experts,
                  'source': json.loads(source.read_text()), 'data_randomness': self.audit_data_rng}
        (Path(self.output_folder) / 'semantic_experiment.json').write_text(json.dumps(record, indent=2) + '\n')
        self.print_to_log_file('SEMANTIC_EXPERIMENT', record)
        probe_path = Path(self.output_folder) / 'semantic_validation_probe.pt'
        if self._semantic_ablate_every and probe_path.exists():
            self._semantic_probe = torch.load(probe_path, map_location='cpu', weights_only=False)
            if self._semantic_probe['seed'] != self.audit_seed:
                raise ValueError('Saved diagnostic patch seed differs from BRATS_AUDIT_SEED')

    def on_train_epoch_start(self):
        self.network.set_epoch(self.current_epoch, self.warmup_epochs)
        super().on_train_epoch_start()
        self._write_module_record('SEMANTIC_PHASE', {
            'phase': 'joint_fusion' if self.current_epoch >= self.warmup_epochs else 'independent_warmup',
            'router_trainable': self.current_epoch >= self.warmup_epochs and not self.uniform_fusion,
            'expert_aux_weight': self.expert_aux_weight if self.current_epoch >= self.warmup_epochs else 1.,
            'fusion_aux_weight': self.fusion_aux_weight if self.current_epoch >= self.warmup_epochs else 0.})

    def compute_semantic_loss(self, parts, target):
        highest_target = target[0] if isinstance(target, (list, tuple)) else target
        main = self.loss(parts['output'], target)
        experts = torch.stack([self.region_aux_loss(z.float(), highest_target) for z in parts['experts']])
        expert_mean = experts.mean()
        fusion = self.region_aux_loss(parts['moe'].float(), highest_target) if parts['active'] else main.new_zeros(())
        total = main + (self.expert_aux_weight * expert_mean + self.fusion_aux_weight * fusion
                        if parts['active'] else expert_mean)
        return total, main, experts, fusion

    def train_step(self, batch):
        sampled = self._semantic_diag_every > 0 and self._semantic_steps % self._semantic_diag_every == 0
        if sampled and self._semantic_monitor is None:
            self._semantic_monitor = SemanticDiagnostics(self.network)
        monitor = self._semantic_monitor
        if sampled:
            monitor.begin_step()
        old_scale = self.grad_scaler.get_scale() if sampled and self.grad_scaler is not None else None
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, (list, tuple)) else target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                parts = self.network(data, return_aux=True)
                loss, main, experts, fusion = self.compute_semantic_loss(parts, target)
            if sampled:
                monitor.observe_output(parts)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()
            if sampled:
                new_scale = self.grad_scaler.get_scale() if self.grad_scaler is not None else None
                monitor.end_step(old_scale is not None and new_scale < old_scale)
        except BaseException:
            if sampled:
                monitor.cancel_step()
            raise
        self._route_batches.append(parts['weights'].detach())
        self._semantic_steps += 1
        return {'loss': loss.detach().cpu().numpy(), 'main_loss': main.detach().cpu().numpy(),
                'expert_losses': experts.detach().cpu().numpy(), 'fusion_loss': fusion.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._write_module_record('SEMANTIC_LOSS', {
            'main_exclusive_ds': float(np.mean([x['main_loss'] for x in train_outputs])),
            'expert_region_losses_T1_T1ce_T2_FLAIR': np.mean([x['expert_losses'] for x in train_outputs], axis=0).tolist(),
            'fusion_region_loss_unweighted': float(np.mean([x['fusion_loss'] for x in train_outputs])),
            'note': 'train total includes auxiliary losses; validation loss uses only final ExclusiveLoss/DS'})
        if self._route_batches:
            self._write_module_record('SEMANTIC_ROUTING', routing_summary(torch.cat(self._route_batches), self.network.semantic))
            self._route_batches.clear()
        if self._semantic_monitor is not None:
            summary = self._semantic_monitor.epoch_summary()
            if summary:
                self._write_module_record('MODULE_GRAD_FEATURE', {'modules': summary,
                    'expected_frozen_parameters': 'unused expert DS heads; router and beta during warmup; uniform-control router'})

    def validation_step(self, batch):
        if self._semantic_ablate_every and self._semantic_probe is None:
            target = batch['target'][0] if isinstance(batch['target'], (list, tuple)) else batch['target']
            self._semantic_probe = {'data': batch['data'].detach().cpu().clone(),
                                    'target': target.detach().cpu().clone(), 'seed': self.audit_seed,
                                    'keys': [str(k) for k in batch.get('keys', [])]}
            torch.save(self._semantic_probe, Path(self.output_folder) / 'semantic_validation_probe.pt')
        # Keep nnU-Net's online TP/FP/FN and validation loss unchanged.
        return nnUNetTrainer.validation_step(self, batch)

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        probe_due = self._semantic_ablate_every and (self.current_epoch in (0, self.warmup_epochs)
                    or (self.current_epoch + 1) % self._semantic_ablate_every == 0)
        if probe_due and self._semantic_probe is not None:
            if self.network.training:
                raise RuntimeError('Semantic contribution probe must run in eval mode')
            with torch.no_grad(), preserved_rng(self.device):
                data = self._semantic_probe['data'].to(self.device, non_blocking=True)
                target = self._semantic_probe['target'].to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                    parts = self.network(data, return_aux=True)
                payload = semantic_probe(self.network, parts, target)
            self._write_module_record('SEMANTIC_CONTRIBUTION', {
                'probe_sha256': probe_digest(self._semantic_probe), 'case_keys': self._semantic_probe['keys'], **payload})

    def load_checkpoint(self, filename_or_checkpoint):
        # Local nnUNetTrainer's dict branch leaves `checkpoint` undefined. Handle
        # both supported inputs here, without modifying upstream trainer code.
        if not self.was_initialized:
            self.initialize()
        checkpoint = (torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
                      if isinstance(filename_or_checkpoint, (str, Path)) else filename_or_checkpoint)
        if checkpoint['trainer_name'] != type(self).__name__:
            raise ValueError('Use matching semantic trainer for resume; learned/uniform are separate experiments')
        weights = {k.removeprefix('module.'): v for k, v in checkpoint['network_weights'].items()}
        if int(weights['initialization_seed'].item()) != self.audit_seed:
            raise ValueError('Resume must use the original BRATS_AUDIT_SEED')
        self.network.load_state_dict(weights)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
            self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', self.inference_allowed_mirroring_axes)
        # Preserve saved prediction phase for --val and checkpoint_best at the
        # warmup boundary. on_train_epoch_start enables the next training phase.
        self.network.semantic.set_joint_stage(bool(self.network.semantic.fusion_active.item()))
        self.lr_scheduler.step(self.current_epoch)
        self._semantic_steps = self.current_epoch * self.num_iterations_per_epoch

    def on_train_end(self):
        try:
            return super().on_train_end()
        finally:
            if self._semantic_monitor is not None:
                self._semantic_monitor.close()
