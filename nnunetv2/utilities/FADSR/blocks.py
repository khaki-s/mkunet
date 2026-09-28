from itertools import product
from math import gcd, prod, sqrt

import torch
from torch import nn
from torch.nn import functional as F


class MedNeXtContextBlock3D(nn.Module):
    """MedNeXt-inspired depthwise inverted bottleneck, with an outer LayerScale.

    Official reference: MIC-DKFZ/MedNeXt, mednextv1/blocks.py:MedNeXtBlock.
    No downsampling, GRN, or replacement of the original encoder convolution.
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 expansion: int = 2, layer_scale: float = 1e-2):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1 or expansion < 1:
            raise ValueError('kernel_size must be positive odd; expansion must be positive')
        self.channels, self.expansion = channels, expansion
        self.depthwise = nn.Conv3d(channels, channels, kernel_size,
                                   padding=kernel_size // 2, groups=channels)
        self.norm = nn.GroupNorm(channels, channels)
        self.expand = nn.Conv3d(channels, expansion * channels, 1)
        self.act = nn.GELU()
        self.project = nn.Conv3d(expansion * channels, channels, 1)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), layer_scale))

    def forward(self, x):
        residual = self.project(self.act(self.expand(self.norm(self.depthwise(x)))))
        return x + self.gamma * residual

    def compute_conv_feature_map_size(self, spatial_size):
        return (2 + self.expansion) * self.channels * prod(spatial_size)


class EMCADMultiScaleBlock3D(nn.Module):
    """3D adaptation of EMCAD's additive, parallel MSCB (not full EMCAD).

    Official reference: SLDGroup/EMCAD, lib/decoders.py:MSDC/MSCB.
    Adaptations: Conv3d, InstanceNorm3d, small residual LayerScale. No attention
    stack, new upsampler, pretrained encoder, or altered supervision.
    """

    def __init__(self, channels: int, kernel_sizes=(1, 3, 5),
                 expansion: int = 2, layer_scale: float = 1e-2):
        super().__init__()
        if expansion < 1 or not kernel_sizes or any(k < 1 or k % 2 != 1 for k in kernel_sizes):
            raise ValueError('positive expansion and positive odd kernel sizes required')
        self.channels, self.expansion = channels, expansion
        hidden = channels * expansion
        self.shuffle_groups = gcd(hidden, channels)
        self.expand = nn.Sequential(
            nn.Conv3d(channels, hidden, 1, bias=False),
            nn.InstanceNorm3d(hidden, eps=1e-5, affine=True), nn.ReLU6(inplace=True))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(hidden, hidden, k, padding=k // 2, groups=hidden, bias=False),
                nn.InstanceNorm3d(hidden, eps=1e-5, affine=True), nn.ReLU6(inplace=True))
            for k in kernel_sizes
        ])
        self.project = nn.Sequential(
            nn.Conv3d(hidden, channels, 1, bias=False),
            nn.InstanceNorm3d(channels, eps=1e-5, affine=True))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), layer_scale))

    def forward(self, x):
        expanded = self.expand(x)
        combined = self.branches[0](expanded)
        for branch in self.branches[1:]:
            combined = combined + branch(expanded)
        b, c, d, h, w = combined.shape
        shuffled = combined.reshape(b, self.shuffle_groups, c // self.shuffle_groups, d, h, w)
        shuffled = shuffled.transpose(1, 2).contiguous().reshape(b, c, d, h, w)
        return x + self.gamma * self.project(shuffled)

    def compute_conv_feature_map_size(self, spatial_size):
        return (self.expansion * (1 + len(self.branches)) + 1) * self.channels * prod(spatial_size)


class HaarWavelet3D(nn.Module):
    """One-level orthonormal Haar transform with explicit odd-size padding.

    Channel-major coefficient order: LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH.
    Fixed filters are buffers, not optimizer parameters. Analysis/synthesis use
    the same orthonormal bank and crop back to the original rectangular shape.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        lo = torch.tensor([1., 1.]) / sqrt(2.)
        hi = torch.tensor([1., -1.]) / sqrt(2.)
        bank = torch.stack([torch.einsum('i,j,k->ijk', a, b, c)
                            for a, b, c in product((lo, hi), repeat=3)])
        self.register_buffer('filters', bank[:, None].repeat(channels, 1, 1, 1, 1))

    def analysis(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError('Expected B,C,D,H,W with the configured channel count')
        spatial = x.shape[2:]
        padding = tuple(v for size in reversed(spatial) for v in (0, size % 2))
        padded = F.pad(x, padding) if any(padding) else x
        coeffs = F.conv3d(padded, self.filters.to(dtype=x.dtype), stride=2, groups=self.channels)
        return coeffs, spatial

    def synthesis(self, coeffs, original_spatial):
        x = F.conv_transpose3d(coeffs, self.filters.to(dtype=coeffs.dtype),
                               stride=2, groups=self.channels)
        d, h, w = original_spatial
        return x[:, :, :d, :h, :w]


class WTConvResidualBlock3D(nn.Module):
    """One-level, true depthwise 3D WTConv-inspired skip residual.

    Official reference: BGU-CS-VIL/WTConv, wtconv/wtconv2d.py:WTConv2d.
    Adaptations: 8 Haar bands, one level, outer identity/LayerScale, stride 1.
    The original skip retains all information; no forced HF gain, shrinkage,
    dense SC-by-SC channel mixing, or deepest-stage wavelet decomposition.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 layer_scale: float = 1e-2, wavelet_scale: float = 0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('kernel_size must be positive odd')
        self.channels = channels
        self.wavelet = HaarWavelet3D(channels)
        self.spatial_conv = nn.Conv3d(channels, channels, kernel_size,
                                      padding=kernel_size // 2, groups=channels)
        self.band_conv = nn.Conv3d(8 * channels, 8 * channels, kernel_size,
                                   padding=kernel_size // 2, groups=8 * channels, bias=False)
        self.band_scale = nn.Parameter(torch.full((1, 8 * channels, 1, 1, 1), wavelet_scale))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1, 1), layer_scale))

    def forward(self, x):
        bands, original_spatial = self.wavelet.analysis(x)
        wavelet_residual = self.wavelet.synthesis(self.band_scale * self.band_conv(bands), original_spatial)
        return x + self.gamma * (self.spatial_conv(x) + wavelet_residual)

    def compute_conv_feature_map_size(self, spatial_size):
        padded_volume = prod([s + s % 2 for s in spatial_size])
        # Analysis, band convolution, synthesis, and full-resolution spatial path.
        return self.channels * (3 * padded_volume + prod(spatial_size))


# Historical checkpoint compatibility; implementations live in legacy_fadsr.
_LEGACY_NAMES = ('_separable_bank', '_make_wavelet_filters', 'WaveletOps', 'WaveletMixer', 'HFShrink', 'LLEnhance', 'drop_path', 'DropPath', 'InceptionDWMixer2d', 'LiteInceptionNeXtBlock2d', 'ASPPModule', 'CBAM', 'BottleneckASPPCBAM', 'FADSR')
__all__ = ['MedNeXtContextBlock3D', 'EMCADMultiScaleBlock3D', 'HaarWavelet3D', 'WTConvResidualBlock3D'] + [n for n in _LEGACY_NAMES if not n.startswith("_")]


def __getattr__(name):
    if name in _LEGACY_NAMES:
        from importlib import import_module
        return getattr(import_module('nnunetv2.utilities.legacy_fadsr.blocks'), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
