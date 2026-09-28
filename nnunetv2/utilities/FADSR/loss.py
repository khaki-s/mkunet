# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, FocalLoss

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class Loss(nn.Module):
    def __init__(self, focal):
        super(Loss, self).__init__()
        if focal:
            self.loss_fn = DiceFocalLoss(
                include_background=False, softmax=True, to_onehot_y=True, batch=True, gamma=2.0
            )
        else:
            self.loss_fn = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, batch=True)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


class LossBraTS(nn.Module):
    def __init__(self, focal):
        super(LossBraTS, self).__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False) if focal else nn.BCEWithLogitsLoss()

    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y.float())

    def forward(self, p, y):
        y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
        p_wt, p_tc, p_et = p[:, 0].unsqueeze(1), p[:, 1].unsqueeze(1), p[:, 2].unsqueeze(1)
        l_wt, l_tc, l_et = self._loss(p_wt, y_wt), self._loss(p_tc, y_tc), self._loss(p_et, y_et)
        return l_wt + l_tc + l_et


class WeightedRegionSoftDiceLoss(MemoryEfficientSoftDiceLoss):
    def __init__(self, *args, region_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_weights = region_weights

    def forward(self, x, y, loss_mask=None):
        if self.region_weights is None:
            return super().forward(x, y, loss_mask)

        # Reproduce the parent up to the per-channel dice, then weight it. Kept
        # in sync with MemoryEfficientSoftDiceLoss; do_bg is True for regions.
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = (y_onehot.sum(axes, dtype=torch.float32) if loss_mask is None
                      else (y_onehot * loss_mask).sum(axes, dtype=torch.float32))

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                from nnunetv2.utilities.ddp_allgather import AllGatherGrad
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        dc = (2 * intersect + self.smooth) / (
            sum_gt + sum_pred + float(self.smooth)).clamp_min(1e-8)

        w = torch.as_tensor(self.region_weights, dtype=dc.dtype, device=dc.device)
        if dc.ndim == 2:                     # (B, C) when batch_dice is False
            w = w.view(1, -1)
        dc = (dc * w).sum(-1) / w.sum()
        return -dc.mean()


class DC_and_BCE_loss_weighted(nn.Module):
    """DC_and_BCE_loss with per-region weights on both the BCE and dice terms."""

    def __init__(self, bce_kwargs, soft_dice_kwargs, region_weights,
                 weight_ce=1, weight_dice=1, use_ignore_label=False,
                 dice_class=WeightedRegionSoftDiceLoss):
        super().__init__()
        bce_kwargs = dict(bce_kwargs)
        # per-channel weighting needs the unreduced map in both cases
        bce_kwargs['reduction'] = 'none'
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label
        self.region_weights = list(region_weights)
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid,
                             region_weights=self.region_weights,
                             **soft_dice_kwargs)

    def forward(self, net_output, target):
        if self.use_ignore_label:
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()

        w = torch.as_tensor(self.region_weights, dtype=net_output.dtype,
                            device=net_output.device)
        w = w.view(1, -1, *([1] * (net_output.ndim - 2))) / w.mean()

        ce_map = self.ce(net_output, target_regions) * w
        if mask is not None:
            ce_loss = (ce_map * mask).sum() / torch.clip(
                mask.sum() * ce_map.shape[1], min=1e-8)
        else:
            ce_loss = ce_map.mean()

        return self.weight_ce * ce_loss + self.weight_dice * dc_loss
