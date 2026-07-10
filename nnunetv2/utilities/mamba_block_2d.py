# mamba_block_2d.py
#
# 2D Mamba building blocks for nnUNetv2.
# Uses the official mamba_ssm.Mamba CUDA kernel — same as LightMUNet.py.
#
# Design mirrors LightMUNet (MambaLayer pattern) but adapted for:
#   - 2D inputs  (B, C, H, W)
#   - Bi-directional scan (H-first + W-first) for 2D spatial coverage
#   - Drop-in compatibility with nnUNetv2 PlainConvUNet encoder
#
# Install dependency first:
#   pip install mamba-ssm causal-conv1d

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Official mamba_ssm library (same as LightMUNet)
from mamba_ssm import Mamba


# ============================================================================
#  MambaLayer2D  —  exact mirror of LightMUNet's MambaLayer, but 2D
# ============================================================================

class MambaLayer2D(nn.Module):
    """
    Applies the official Mamba SSM to a 2D feature map.

    Input shape:  (B, C, H, W)
    Output shape: (B, out_channels, H, W)

    Matches LightMUNet's MambaLayer structure exactly:
      flatten → LayerNorm → Mamba + skip_scale → LayerNorm → proj → unflatten
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        # Official Mamba (CUDA-accelerated)
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast fp16 → fp32 (Mamba kernel requirement, same as LightMUNet)
        if x.dtype == torch.float16:
            x = x.float()

        B, C, H, W = x.shape
        assert C == self.input_dim, f"Channel mismatch: got {C}, expected {self.input_dim}"

        # (B, C, H, W) → (B, H*W, C)
        x_flat = x.reshape(B, C, H * W).transpose(-1, -2)

        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)  # (B, H*W, output_dim)

        # (B, H*W, output_dim) → (B, output_dim, H, W)
        return x_mamba.transpose(-1, -2).reshape(B, self.output_dim, H, W)


# ============================================================================
#  Bi-directional scan: H-scan + W-scan  (2D equivalent of TriOriented)
# ============================================================================

class BiDirectionalMamba2D(nn.Module):
    """
    Scans the 2D feature map along two orientations:
      - Row-major  (H then W): standard raster order
      - Col-major  (W then H): transpose order

    Fusing both gives the model access to spatial context in both axes,
    analogous to SegMamba's tri-oriented 3D scan.

    Input/output: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        # Two Mamba instances: one per scan direction
        self.mamba_row = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_col = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)

        self.skip_scale_row = nn.Parameter(torch.ones(1))
        self.skip_scale_col = nn.Parameter(torch.ones(1))

        # Fuse two directions → same channel count
        self.fuse = nn.Linear(channels * 2, channels, bias=False)
        self.norm_out = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()

        B, C, H, W = x.shape

        # ── Row-major scan: flatten (H, W) in row order ──────────────────
        x_row = x.reshape(B, C, H * W).transpose(-1, -2)          # (B, H*W, C)
        x_row_norm = self.norm(x_row)
        y_row = self.mamba_row(x_row_norm) + self.skip_scale_row * x_row  # (B, H*W, C)

        # ── Col-major scan: transpose H,W first, then flatten ─────────────
        x_col = x.permute(0, 1, 3, 2).reshape(B, C, H * W).transpose(-1, -2)  # (B, W*H, C)
        x_col_norm = self.norm(x_col)
        y_col = self.mamba_col(x_col_norm) + self.skip_scale_col * x_col       # (B, W*H, C)

        # Restore col-scan back to (B, H*W, C) matching row-scan layout
        # col result is in (W, H) order → transpose back
        y_col = y_col.transpose(-1, -2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        y_col = y_col.reshape(B, C, H * W).transpose(-1, -2)      # (B, H*W, C)

        # ── Fuse ─────────────────────────────────────────────────────────
        y = torch.cat([y_row, y_col], dim=-1)                      # (B, H*W, 2C)
        y = self.fuse(y)                                            # (B, H*W, C)
        y = self.norm_out(y)

        return y.transpose(-1, -2).reshape(B, C, H, W)             # (B, C, H, W)


# ============================================================================
#  MambaResBlock2D  —  Conv(local) + BiDirectionalMamba2D(global) + residual
# ============================================================================

class MambaResBlock2D(nn.Module):
    """
    The core hybrid block used at bottleneck and deep encoder stages.

    Structure (mirrors LightMUNet's ResMambaBlock logic):
      input
        ├─ DWConv (local context)  → Norm → Act
        └─ BiDirectionalMamba2D   (global context)
      fused → residual add → GroupNorm

    Input/output: (B, channels, H, W)
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        # Local branch: depthwise-separable conv (lightweight, like LightMUNet)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                 groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

        # Global branch: bi-directional Mamba
        self.mamba = BiDirectionalMamba2D(channels, d_state=d_state,
                                          d_conv=d_conv, expand=expand)

        # Output norm
        self.norm2 = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local
        h = self.act(self.norm1(self.pw_conv(self.dw_conv(x))))
        # Global
        m = self.mamba(h)
        # Residual
        return x + self.norm2(m)
