"""300-epoch controlled ablations. Network, blocks and trainer are separated.

Discoverable public subclasses live in variants/brats2021_audit. No legacy
trainer, optimizer mixin, custom augmentation, or checkpoint is inherited.
"""
import json
import os
import numpy as np
from pathlib import Path

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from .blocks import ExclusiveHierarchyLoss
from .network import build_audit_network
from ..sampling import ExclusiveClassDataLoader, validate_index, digest
from ..diagnostics import ModuleDiagnosticsMixin, finite_number
from ..reproducibility import seed_experiment, seeded_augmenter, state_digest, configure_determinism


class BraTSAuditTrainer300(ModuleDiagnosticsMixin, nnUNetTrainer):
    mode = 'baseline'
    class_sampling = False
    exclusive_loss = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.audit_seed = int(os.environ.get('BRATS_AUDIT_SEED', '0'))
        if configuration != '3d_fullres' or self.is_cascaded or self.is_ddp:
            raise ValueError('First ablations support single-GPU 3d_fullres only')
        channels = [str(dataset_json['channel_names'][str(i)]).lower() for i in range(4)]
        if channels != ['t1', 't1ce', 't2', 'flair']:
            raise ValueError('Required modality order: T1/T1ce/T2/FLAIR')
        regions = [tuple(r) if isinstance(r, (list, tuple)) else (r,)
                   for r in self.label_manager.foreground_regions]
        if regions != [(1, 2, 3), (2, 3), (3,)] or self.label_manager.has_ignore_label:
            raise ValueError('Required regions WT=(1,2,3), TC=(2,3), ET=(3), without ignore label')
        if list(dataset_json.get('regions_class_order', [])) != [1, 2, 3]:
            raise ValueError('Required export overwrite order: ED=1, NCR=2, ET=3')

    def _do_i_compile(self):
        return False

    def initialize(self):
        if self.was_initialized:
            return
        seed_experiment(self.audit_seed)
        super().initialize()
        self.backbone_initial_sha256 = {
            'encoder': state_digest(self.network.encoder),
            'decoder': state_digest(self.network.decoder)}
        self.print_to_log_file('BACKBONE_INITIALIZATION', {
            'seed': self.audit_seed, 'sha256': self.backbone_initial_sha256,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'note': 'Compare hashes BEFORE any training or checkpoint loading.'})

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_audit_network(configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision, cls.mode)

    def _build_loss(self):
        loss = super()._build_loss()
        if self.exclusive_loss:
            if isinstance(loss, DeepSupervisionWrapper):
                loss.loss = ExclusiveHierarchyLoss(loss.loss)
            else:
                loss = ExclusiveHierarchyLoss(loss)
        return loss

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch = self.configuration_manager.patch_size
        scales = self._get_deep_supervision_scales()
        rotation, dummy2d, initial_patch, mirrors = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        train_transform = self.get_training_transforms(
            patch, rotation, scales, mirrors, dummy2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=False, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions, ignore_label=None)
        val_transform = self.get_validation_transforms(
            scales, is_cascaded=False, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions, ignore_label=None)
        train, val = self.get_tr_and_val_datasets()
        shared = dict(oversample_foreground_percent=self.oversample_foreground_percent,
                      sampling_probabilities=None, pad_sides=None,
                      probabilistic_oversampling=self.probabilistic_oversampling)
        if self.class_sampling:
            index = Path(self.preprocessed_dataset_folder).parent / f'brats2021_audit_sampling_fold{self.fold}'
            manifest = validate_index(index, self.preprocessed_dataset_folder, train.identifiers)
            self.print_to_log_file('Training-only class sampling index:', str(index))
            dl_train = ExclusiveClassDataLoader(train, self.batch_size, initial_patch, patch, self.label_manager,
                                                transforms=train_transform, index_folder=index, manifest=manifest, **shared)
        else:
            dl_train = nnUNetDataLoader(train, self.batch_size, initial_patch, patch, self.label_manager,
                                        transforms=train_transform, **shared)
        dl_val = nnUNetDataLoader(val, self.batch_size, patch, patch, self.label_manager,
                                 transforms=val_transform, **shared)
        workers = get_allowed_n_proc_DA()
        val_workers = max(1, workers // 2) if workers else 0
        self.audit_data_rng = {'policy': 'seed_per_logical_batch_v1', 'seed': self.audit_seed,
                               'train_workers': workers, 'val_workers': val_workers,
                               'train_offset': self.current_epoch * self.num_iterations_per_epoch,
                               'val_offset': self.current_epoch * self.num_val_iterations_per_epoch}
        self.print_to_log_file('DATA_RANDOMNESS', self.audit_data_rng)
        train_aug = seeded_augmenter(dl_train, self.audit_seed, 0, workers,
                                      self.audit_data_rng['train_offset'], self.device.type == 'cuda')
        val_aug = seeded_augmenter(dl_val, self.audit_seed, 1, val_workers,
                                    self.audit_data_rng['val_offset'], self.device.type == 'cuda')
        # Do not consume a hidden first batch: offsets also identify resume batches.
        return train_aug, val_aug

    def on_train_start(self):
        # CLI resets cuDNN flags after loading a checkpoint. Reapply without
        # resetting RNG or overwriting resumed weights.
        configure_determinism()
        super().on_train_start()
        root = Path(self.preprocessed_dataset_folder).parent
        sources = Path(__file__).resolve().parents[1]
        record = {'trainer': type(self).__name__, 'mode': self.mode, 'class_sampling': self.class_sampling,
                  'exclusive_loss': self.exclusive_loss, 'epochs': self.num_epochs, 'seed': self.audit_seed,
                  'torch': torch.__version__, 'patch_size': list(self.configuration_manager.patch_size),
                  'batch_size': self.batch_size,
                  'backbone_initial_sha256': self.backbone_initial_sha256,
                  'data_randomness': self.audit_data_rng,
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'data_hashes': {p: digest(root / p) for p in
                                  ('splits_final.json', 'dataset.json', 'nnUNetPlans.json')},
                  'code_hashes': {str(p.relative_to(sources)): digest(p) for p in sorted(sources.rglob('*.py'))},
                  'mamba_source': json.loads((sources / 'vendor/provenance.json').read_text())}
        (Path(self.output_folder) / 'audit_experiment.json').write_text(json.dumps(record, indent=2) + '\n')
        self.print_to_log_file('BraTS audit experiment:', record)

    def perform_actual_validation(self, save_probabilities=False):
        configure_determinism()
        return super().perform_actual_validation(save_probabilities)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        moe = self.network.moe
        if moe is not None and moe.route_count.item() > 0:
            weights = (moe.route_sum / moe.route_count).cpu().tolist()
            self._write_module_record('MODULE_ROUTING', {
                'module': 'moe', 'queries': ['q0', 'q1', 'q2'],
                'query_semantics': 'latent queries; no guaranteed ED/NCR/ET correspondence',
                'experts': ['T1', 'T1ce', 'T2', 'FLAIR', 'joint'],
                'scope': 'mean over training patches, including background; not causal importance',
                'training_batches': int(moe.route_count.item()),
                'mean_weights': [[finite_number(w) for w in row] for row in weights]})
            moe.route_sum.zero_()
            moe.route_count.zero_()
