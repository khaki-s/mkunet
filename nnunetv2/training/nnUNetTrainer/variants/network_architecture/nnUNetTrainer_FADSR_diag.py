"""Diagnostics-only trainers for the dd43a60 architecture.

Nothing here changes the network. `nnUNetTrainer_FADSR_diag_100epochs` builds the
exact same graph as `nnUNetTrainer_FADSR_100epochs` and only adds per-epoch
logging, so it is safe to run against the 91.24/91.62/88.87 baseline.

What gets logged, and why it is the number that matters: every module added to
this network has the form

    y = x + scale * branch(x)

so its influence is `scale * ||branch(x)|| / ||x||`, NOT its parameter count.
Measured on the trained dd43a60 checkpoint, that came out at

    FADSR mixer gates   |scale| ~ 0.009  -> contrib < 1%
    BottleneckASPPCBAM  |gamma| = 7.2e-5 -> contrib ~ 0.007%

i.e. both modules were nearly inert and the Dice differences between variants
were mostly not attributable to them. This logs it during training so that shows
up on epoch 1 instead of after 100 epochs.

  contrib%     = ||y - x|| / ||x||, on real batches (forward hook)
  gradW/gradS  = ||dL/d(branch weights)|| / ||dL/d(scale)||
                 At scale=0 this is 0 BY CONSTRUCTION: dL/dW = dL/dy * scale * (...)
                 = 0, so only `scale` moves and the branch waits for it to leave 0.
"""

import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainer_FADSR import (
    nnUNetTrainer_FADSR_100epochs,
)
from nnunetv2.utilities.FADSR.diagnostics import (
    BranchDiagnostics, split_no_decay_params,
)


class BranchDiagnosticsMixin:
    """Log per-branch contribution and gradient flow once per epoch."""

    enable_branch_diagnostics: bool = True
    diag_grad_every: int = 50

    def on_train_start(self):
        super().on_train_start()
        self._diag = None
        if not self.enable_branch_diagnostics:
            return
        net = self.network.module if self.is_ddp else self.network
        net = getattr(net, '_orig_mod', net)          # torch.compile unwrap
        self._diag = BranchDiagnostics(net, grad_every=self.diag_grad_every)
        self.print_to_log_file(
            f'BranchDiagnostics: tracking {len(self._diag._targets)} gated branches')

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(batch)
        # nnUNetTrainer.train_step zero_grads at the TOP of the step, so grads are
        # still live here, and already unscaled by grad_scaler.unscale_() before
        # the optimizer step. Post-clip, which preserves the ratio.
        if getattr(self, '_diag', None) is not None:
            self._diag.sample_grads()
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if getattr(self, '_diag', None) is not None:
            self.print_to_log_file(self._diag.format_report(),
                                   also_print_to_console=False)
            self._diag.reset()


class NoDecayGatesMixin:
    """Keep weight decay off gates and LayerScale.

    nnUNetTrainer.configure_optimizers hands self.network.parameters() straight to
    SGD, so FADSR.no_weight_decay() is defined but never consulted -- and it only
    matches 'scale', missing every LayerScale `gamma`. At wd=3e-5 with momentum
    0.99 a gate is being shrunk while it is still trying to climb out of 0.
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


class nnUNetTrainer_FADSR_diag_100epochs(BranchDiagnosticsMixin,
                                         nnUNetTrainer_FADSR_100epochs):
    """dd43a60 architecture, byte-identical. Logging only, zero risk."""
    pass
