import pydoc

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.fadsr_unet import FADSR_PlainConvUNet
from nnunetv2.utilities.FADSR.diagnostics import (
    BranchDiagnostics, split_no_decay_params,
)


class nnUNetTrainer_FADSR(nnUNetTrainer):
    # subclasses set this to rewire FADSR without touching the plans
    fadsr_arch_overrides: dict = {}

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
        for ri in configuration_manager.network_arch_init_kwargs_req_import:
            if arch_kwargs[ri] is not None:
                arch_kwargs[ri] = pydoc.locate(arch_kwargs[ri])
        arch_kwargs.update(cls.fadsr_arch_overrides)
        return FADSR_PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            # lets FADSR size its cascade per skip instead of always building 3
            patch_size=configuration_manager.patch_size,
            **arch_kwargs
        )


class BranchDiagnosticsMixin:
    """Log per-branch contribution and gradient flow once per epoch.

    Answers the question the Dice column cannot: is the module doing anything?
    Everything added here is `y = x + scale * branch(x)`, so a small `scale`
    makes the branch invisible regardless of how it is designed internally.
    """

    enable_branch_diagnostics: bool = True
    diag_grad_every: int = 200

    def on_train_start(self):
        super().on_train_start()
        self._diag = None
        if not self.enable_branch_diagnostics:
            return
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, '_orig_mod', net)      # torch.compile unwrap
        self._diag = BranchDiagnostics(net, grad_every=self.diag_grad_every)
        self.print_to_log_file(
            f'BranchDiagnostics: tracking {len(self._diag._targets)} gated branches')

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(batch)
        # nnUNetTrainer.train_step zero_grads at the TOP, so grads are still
        # live here, and already unscaled by grad_scaler.unscale_() before the
        # optimizer step. Post-clip, which preserves ratios.
        if getattr(self, '_diag', None) is not None:
            self._diag.sample_grads()
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if getattr(self, '_diag', None) is not None:
            self.print_to_log_file(self._diag.format_report(), also_print_to_console=False)
            self._diag.reset()


class NoDecayGatesMixin:
    """Keep weight decay off gates and LayerScale.

    nnUNetTrainer.configure_optimizers hands self.network.parameters() to SGD, so
    FADSR.no_weight_decay() is never consulted.
    """

    def configure_optimizers(self):
        decay, no_decay = split_no_decay_params(self.network)
        optimizer = torch.optim.SGD(
            [{'params': decay, 'weight_decay': self.weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            self.initial_lr, momentum=0.99, nesterov=True)
        self.print_to_log_file(
            f'optimizer: {len(decay)} tensors with wd={self.weight_decay}, '
            f'{len(no_decay)} gate/LayerScale tensors with wd=0')
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


class nnUNetTrainer_FADSR_99epochs(nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 99


class nnUNetTrainer_FADSR_100epochs(nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_FADSR_diag_100epochs(BranchDiagnosticsMixin,
                                         nnUNetTrainer_FADSR_100epochs):
    """Current best architecture, unchanged. Only adds logging.

    Run this first: it reports what the existing 91.24/91.62/88.87 network is
    actually doing internally, with zero risk to the result.
    """
    pass


class nnUNetTrainer_FADSR_base_100epochs(BranchDiagnosticsMixin,
                                         NoDecayGatesMixin,
                                         nnUNetTrainer_FADSR_100epochs):
    """Direction 1: restore WTConv's parallel base branch, open the gates.

    The db2 padding fix and the resolution-capped cascade are unconditional in
    blocks.py now, so they apply here and to nnUNetTrainer_FADSR_100epochs alike.
    What this trainer adds on top:

    - use_base_branch: parallel full-res depthwise, zero-init weight with
      base_scale=1.0 -> identity at init but full-strength gradient from step 0
      (RepLKNet's parallel small-kernel path; WTConv's `base_conv`).
    - mixer_scale_init=0.1: WTConv's value for the wavelet path
      (FADSR/wtconv2d.py:30). At 0.0 the branch weights get exactly zero
      gradient at step 0; measured init perturbation at 0.1 is 2.2-2.5%.
    - bottleneck_layer_scale_init=1e-2: the trained gamma was 7.2e-5, i.e. ASPP
      +CBAM moved the bottleneck by ~0.007%.
    """
    fadsr_arch_overrides = {
        'fadsr_use_base_branch': True,
        'fadsr_base_kernel': 5,
        'fadsr_mixer_scale_init': 0.1,
        'bottleneck_layer_scale_init': 1e-2,
    }