"""Standalone five-expert ablation with the Legacy300 optimizer and schedule."""
import hashlib
import json
import os
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from ...diagnostics import preserved_rng, patch_metrics
from ...missing_modality import MASK_CODES, mask_from_code
from ...reproducibility import seed_experiment, state_digest, configure_determinism
from ..generalist.blocks import EXPERT_NAMES
from ..generalist.trainer import BraTSGeneralistTrainer300
from .diagnostics import ExpertsOnlyRoutingAccumulator, experts_only_summary
from .network import build_five_small_experts_legacy_network


class BraTSFiveSmallExpertsLegacyTrainer300(BraTSGeneralistTrainer300):
    output_gate = False
    include_self = False
    mode = 'five_small_experts_only_legacy_v2'

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=False):
        return build_five_small_experts_legacy_network(
            configuration_manager, num_input_channels, num_output_channels,
            enable_deep_supervision, **cls.network_options())

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get('BRATS_AGENT_INITIAL_WEIGHTS'):
            raise ValueError('ExpertsOnlyLegacy starts from scratch; unset BRATS_AGENT_INITIAL_WEIGHTS')
        seed_experiment(self.audit_seed)
        # The ancestor initializer fingerprints encoder/decoder as a joint
        # backbone. Deliberately bypass it, while reusing its loss construction,
        # seeded data loaders, train_step, optimizer, checkpoint and resume code.
        nnUNetTrainer.initialize(self)
        self.backbone_initial_sha256 = {}
        self.expert_initial_sha256 = {name: state_digest(module)
            for name, module in zip(EXPERT_NAMES, self.network.semantic.experts)}
        self.print_to_log_file('FIVE_EXPERT_INITIALIZATION', {
            'seed': self.audit_seed, 'joint_backbone_present': False,
            'expert_sha256': self.expert_initial_sha256,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()})
        self.region_aux_loss = self.loss.base_loss
        self._agent_steps = self.current_epoch * self.num_iterations_per_epoch
        self._agent_routes = ExpertsOnlyRoutingAccumulator()
        self._agent_probe = None
        self._agent_diag_every = int(os.environ.get('BRATS_DIAG_EVERY', '50'))
        self._agent_probe_every = int(os.environ.get('BRATS_ABLATE_EVERY', '25'))
        if min(self._agent_diag_every, self._agent_probe_every) < 0:
            raise ValueError('Diagnostic intervals must be nonnegative')
        self.network.set_epoch(self.current_epoch)
        self._agent_initialization = {'kind': 'from_scratch'}
        encode = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        splits = Path(self.preprocessed_dataset_folder_base) / 'splits_final.json'
        self._agent_resume_signature = {
            'fold': self.fold, 'batch_size': self.batch_size,
            'patch_size': list(self.configuration_manager.patch_size),
            'plans_sha256': encode(self.plans_manager.plans),
            'dataset_json_sha256': encode(self.dataset_json),
            'splits_sha256': hashlib.sha256(splits.read_bytes()).hexdigest() if splits.exists() else None}
        source = Path(__file__).resolve().parents[2]
        self._v2_source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(source.rglob('*.py'))}
        registry = source.parents[1] / ('training/nnUNetTrainer/variants/brats2021_audit/'
            'nnUNetTrainer_BraTSAgentGeneralistExpertsOnlyLegacyV2Loss300.py')
        self._v2_source_hashes['experts_only_trainer_registry'] = hashlib.sha256(registry.read_bytes()).hexdigest()

    def training_contract(self):
        contract = super().training_contract()
        contract.update(version=7, architecture='agent_v2_five_small_experts_only_legacy')
        contract['generalist'].update(shared_joint_backbone=False,
            joint_backbone_present=False, independent_expert_networks=5,
            generalist_input_channels=8, beta_trainable=False,
            beta_kind='not used in forward; constant 1 in compatibility routing records')
        contract['experts_only'] = {
            'networks': ['T1', 'T1ce', 'T2', 'FLAIR', 'Generalist'],
            'widths': [8, 16, 32, 64, 80, 80],
            'joint_encoder_decoder_constructed': False, 'joint_bypass': False,
            'forward': 'sum_m w_cm * refined_expert_mc; no beta blending',
            'deep_supervision': False, 'output': 'single highest-resolution Bx3xDxHxW tensor',
            'warmup_output': 'uniform mean of observed physical specialists and Generalist',
            'main_loss': 'region BCE-Dice + 0.1 exclusive + 0.05 hierarchy, highest resolution only',
            'local_and_refined_loss': 'unchanged patient-first observed five-expert means',
            'loss_coefficients': 'inherited Legacy: main+local before50; main+.25local+.25fusion thereafter; refined ramp100..119 to.1',
            'main_and_fusion_same_output': True,
            'main_and_fusion_collapse': 'after50: 1.25*region(final)+.1exclusive(final)+.05hierarchy(final)+.25local+lambda*refined',
            'optimizer_and_mask_sampling': 'unchanged Legacy300',
            'initial_tensors_identical_to_six_network_legacy': False,
            'initialization': 'same expert constructors/default initializers, new seeded from-scratch draw without constructing J',
            'scope': 'J-removal/capacity ablation; also removes J coarse-scale supervision; not a deep-supervision-matched causal control',
        }
        return contract

    def set_deep_supervision_enabled(self, enabled):
        if enabled:
            raise ValueError('ExpertsOnlyLegacy has only highest-resolution supervision')
        self.network.deep_supervision = False

    def collect_step_diagnostics(self, parts):
        return experts_only_summary(self.network, parts)

    def on_validation_epoch_end(self, val_outputs):
        nnUNetTrainer.on_validation_epoch_end(self, val_outputs)
        if (not self._agent_probe_every or self._agent_probe is None
                or (self.current_epoch + 1) % self._agent_probe_every):
            return
        with torch.no_grad(), preserved_rng(self.device):
            x, y = (self._agent_probe[k].to(self.device) for k in ('data', 'target'))
            records = {}
            for code in MASK_CODES:
                with torch.autocast(self.device.type, enabled=self.device.type == 'cuda'):
                    parts = self.network(x, True, modality_mask=mask_from_code(code, device=self.device)[None])
                    record = {'final': patch_metrics(parts['final'], y),
                              **experts_only_summary(self.network, parts, include_gradients=False)}
                    record['local'] = {name: patch_metrics(z, y) for m, (name, z) in
                                      enumerate(zip(EXPERT_NAMES, parts['local_experts']))
                                      if parts['expert_mask'][:, m].any()}
                records[str(code)] = record
            self._write_module_record('EXPERTS_ONLY_15_MASK_PATCH_PROBE', {
                'masks': records, 'expert_names': EXPERT_NAMES,
                'joint_present': False, 'scope': 'fixed validation patch, not test100'})

    def perform_actual_validation(self, save_probabilities=False):
        from ...agent_evaluate import evaluate_preprocessed
        configure_determinism()
        _, keys = self.do_split()
        was_training = self.network.training
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        try:
            return evaluate_preprocessed(self.network, self.plans_manager, self.configuration_manager,
                self.dataset_json, self.dataset_class, self.preprocessed_dataset_folder, keys,
                Path(self.preprocessed_dataset_folder_base) / 'gt_segmentations',
                Path(self.output_folder) / 'validation_15m', device=self.device,
                mirror_axes=self.inference_allowed_mirroring_axes,
                save_probabilities=save_probabilities,
                provenance={'trainer': type(self).__name__, 'stage_epoch': int(self.network.stage_epoch),
                            'joint_present': False})
        finally:
            self.network.train(was_training)
