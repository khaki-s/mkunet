from typing import Sequence

import torch
from torch import nn

from monai.losses import TverskyLoss


class MonaiRegionTverskyBCE(nn.Module):
    """Same call signature as DC_and_BCE_loss, so DeepSupervisionWrapper is unchanged."""

    def __init__(self, alphas: Sequence[float], betas: Sequence[float],
                 batch_dice: bool = True, smooth: float = 1e-5,
                 weight_ce: float = 1.0, weight_tversky: float = 1.0,
                 bce_kwargs: dict = None):
        """
        :param alphas: FP weight per region, ordered [WT, TC, ET].
        :param betas:  FN weight per region. beta > alpha favours recall.
        :param batch_dice: aggregate TP/FP/FN over the batch before the ratio
                           (monai's `batch`), matching nnU-Net's batch_dice.
        """
        super().__init__()
        assert len(alphas) == len(betas), 'alphas and betas must be the same length'
        self.n_regions = len(alphas)
        self.weight_ce = weight_ce
        self.weight_tversky = weight_tversky

        self.ce = nn.BCEWithLogitsLoss(**(bce_kwargs or {}))
        # one per region because monai takes scalar alpha/beta
        self.tversky = nn.ModuleList([
            TverskyLoss(include_background=True, sigmoid=True, batch=batch_dice,
                        alpha=float(a), beta=float(b),
                        smooth_nr=smooth, smooth_dr=smooth)
            for a, b in zip(alphas, betas)
        ])

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        assert net_output.shape[1] == self.n_regions, \
            f'expected {self.n_regions} region channels, got {net_output.shape[1]}'
        assert net_output.shape == target.shape, \
            f'net_output {tuple(net_output.shape)} vs target {tuple(target.shape)}'

        target = target.float()

        # per-region so each can carry its own alpha/beta
        tv = net_output.new_zeros(())
        for c, loss_c in enumerate(self.tversky):
            # monai returns 1 - index (range [0,1]); nnU-Net's dice returns -dc
            # (range [-1,0]). Shift to nnU-Net's convention so train_loss stays
            # comparable with earlier runs. Constant, so gradients are unchanged.
            tv = tv + (loss_c(net_output[:, c:c + 1], target[:, c:c + 1]) - 1.0)
        tv = tv / self.n_regions          # mean, to match Dice's dc.mean()

        ce = self.ce(net_output, target)
        return self.weight_ce * ce + self.weight_tversky * tv
