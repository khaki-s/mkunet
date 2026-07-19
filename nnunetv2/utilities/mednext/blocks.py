import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ImportError:
    try:
        from timm.models.layers import DropPath
    except ImportError:
        class DropPath(nn.Module):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return x


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, C = x.shape[0], x.shape[1]
    spatial = x.shape[2:]
    assert C % groups == 0, (
        f"channel_shuffle: num_channels={C} not divisible by groups={groups}"
    )
    channels_per_group = C // groups
    # (B, groups, channels_per_group, *spatial) -> swap first two channel axes
    x = x.view(B, groups, channels_per_group, *spatial)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(B, C, *spatial)
    return x


def act_layer(act: str = 'gelu') -> nn.Module:
    act = act.lower()
    if act == 'gelu':
        return nn.GELU()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    if act == 'relu6':
        return nn.ReLU6(inplace=True)
    if act == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    raise NotImplementedError(f'activation layer [{act}] is not found')


def _gn_groups(num_channels: int, prefer=(32, 16, 8, 4, 2, 1)) -> int:
    for g in prefer:
        if num_channels % g == 0:
            return g
    return 1


def _norm_layer(norm_type: str, num_channels: int, dim: str) -> nn.Module:

    if norm_type in ('group', 'gn'):
        return nn.GroupNorm(num_groups=_gn_groups(num_channels),
                            num_channels=num_channels)
    if norm_type in ('batch', 'bn'):
        return nn.BatchNorm2d(num_channels) if dim == '2d' else nn.BatchNorm3d(num_channels)
    if norm_type in ('instance', 'in'):
        return (nn.InstanceNorm2d(num_channels) if dim == '2d'
                else nn.InstanceNorm3d(num_channels))
    raise NotImplementedError(f'norm_type [{norm_type}] is not supported')


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # gamma
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # beta
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        # channels_first: normalise across the channel axis only.
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
        x = self.weight.view(shape) * x + self.bias.view(shape)
        return x


class MedNeXtBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 2,
                 kernel_sizes: Sequence[int] = (1, 3, 5),
                 do_res: bool = True,
                 norm_type: str = 'batch',
                 dim: str = '2d',
                 add: bool = True,
                 act: str = 'relu6',
                 layer_scale_init: float = 1e-6,
                 drop_path_rate: float = 0.0,
                 **kwargs):

        super().__init__()

        assert dim in ['2d', '3d']
        self.dim = dim
        conv = nn.Conv2d if dim == '2d' else nn.Conv3d

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.n_scales = len(self.kernel_sizes)
        self.add = add

        # Residual only when shapes match (blocks in mednext_unet are ch->ch).
        self.do_res = do_res and (in_channels == out_channels)

        # Expansion factor (t in MobileNetV2). ex_c is the working width.
        self.exp_r = exp_r
        self.ex_c = int(in_channels * exp_r)

        # ---- 1x1 Expand + Norm + Act ----
        self.expand_pw = conv(in_channels, self.ex_c, kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.expand_norm = _norm_layer(norm_type, self.ex_c, dim)
        self.act = act_layer(act)

        # ---- Parallel depthwise branches: DW{k} + Norm + Act ----
        # Each branch normalises internally, per the spec.
        self.dw_branches = nn.ModuleList([
            nn.Sequential(
                conv(self.ex_c, self.ex_c, kernel_size=k, stride=1,
                     padding=k // 2, groups=self.ex_c, bias=False),
                _norm_layer(norm_type, self.ex_c, dim),
                act_layer(act),
            )
            for k in self.kernel_sizes
        ])

        # Channels feeding the projection.
        self.combined_channels = self.ex_c if self.add else self.ex_c * self.n_scales

        # Channel shuffle groups: gcd of combined & output, as in MKUNet.
        self.shuffle_groups = gcd(self.combined_channels, out_channels)
        if self.combined_channels % self.shuffle_groups != 0:
            self.shuffle_groups = 1  # safety net; would otherwise assert-fail

        # ---- 1x1 Project + Norm (no activation -> residual stays linear) ----
        self.project_pw = conv(self.combined_channels, out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.project_norm = _norm_layer(norm_type, out_channels, dim)

        # ---- LayerScale (ConvNeXt gamma) ----
        if layer_scale_init > 0.0 and self.do_res:
            ls_shape = (1, out_channels) + ((1, 1) if dim == '2d' else (1, 1, 1))
            self.layer_scale = nn.Parameter(torch.full(ls_shape, layer_scale_init))
        else:
            self.layer_scale = None

        # ---- DropPath (stochastic depth) ----
        self.drop_path = (DropPath(drop_path_rate)
                          if drop_path_rate > 0.0 and self.do_res
                          else nn.Identity())

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                # efficientnet-like fan-out init
                fan_out = 1
                for k in m.kernel_size:
                    fan_out *= k
                fan_out *= m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, dummy_tensor=None):
        residual = x

        # 1x1 Expand + Norm + Act
        y = self.act(self.expand_norm(self.expand_pw(x)))

        # Parallel multi-kernel depthwise branches (each internally normed)
        if self.add:
            # Sum -> balanced multi-scale features
            y = sum(branch(y) for branch in self.dw_branches)
        else:
            y = torch.cat([branch(y) for branch in self.dw_branches], dim=1)

        # Channel shuffle to mix across the parallel-branch groups
        if self.shuffle_groups > 1:
            y = channel_shuffle(y, self.shuffle_groups)

        # 1x1 Project + Norm
        y = self.project_norm(self.project_pw(y))

        # LayerScale
        if self.layer_scale is not None:
            y = self.layer_scale * y

        # DropPath + Residual
        if self.do_res:
            y = residual + self.drop_path(y)
        return y


if __name__ == "__main__":

    # 2D
    net2d = MedNeXtBlock(in_channels=32, out_channels=32, exp_r=2,
                         kernel_sizes=(1, 3, 5), dim='2d',
                         norm_type='batch', layer_scale_init=1e-6,
                         drop_path_rate=0.1).cuda()
    with torch.no_grad():
        x = torch.zeros((2, 32, 64, 64)).cuda()
        print(net2d(x).shape)  # (2, 32, 64, 64)

    # 3D
    net3d = MedNeXtBlock(in_channels=32, out_channels=32, exp_r=2,
                         kernel_sizes=(1, 3), dim='3d',
                         norm_type='group', layer_scale_init=1e-6)
    with torch.no_grad():
        x = torch.zeros((2, 32, 16, 16, 16)).cuda()
        print(net3d(x).shape)  # (2, 32, 16, 16, 16)
