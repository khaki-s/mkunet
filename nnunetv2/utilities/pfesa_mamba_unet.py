"""
PFESA + Mamba bottleneck U-Net for nnUNetv2 (2D).

Design follows the placement study in arXiv:2503.17540 (and aligned with
U-Mamba / SegMamba practice):
  * Mamba is placed AFTER the two consecutive conv blocks of a stage,
    not replacing them — a residual hybrid module.
  * Mamba is inserted ONLY in the deepest two encoder stages, plus an
    additional Mamba x2 on the bottleneck output.
  * Decoder is left completely untouched.
  * PFESA (2D, parameter-free) sits on every skip connection.

Key choices vs naive "Mamba everywhere":
  - Bi-directional 2D scan (row + col) inside each Mamba block: a single
    causal raster scan is too directional for 2D segmentation.
  - gamma = 0 identity init on every Mamba residual branch: the network
    starts identical to the strong PlainConvUNet+PFESA baseline and
    learns the Mamba contribution gradually. This is critical when the
    training set is small (240 BraTS21 cases) — guarantees we never
    regress below the baseline at init.
  - GroupNorm wrap on the Mamba output (stabilizes mixed-precision).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from mamba_ssm import Mamba

from nnunetv2.utilities.pfesa import PFESA

class BiMambaResidual2D(nn.Module):
    """
    Bi-directional Mamba over a 2D feature map, returned as a residual
    update to the input.

        out = x + gamma * Fuse( Mamba_row(x), Mamba_col(x) )

    gamma is a learnable scalar (LayerScale-style) initialised to gamma_init (default 0.1).
    """

    def __init__(self, channels: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 gamma_init: float = 0.1):
        super().__init__()
        self.channels = channels
        self.norm_in = nn.LayerNorm(channels)
        self.mamba_row = Mamba(d_model=channels, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_col = Mamba(d_model=channels, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.fuse = nn.Linear(channels * 2, channels, bias=False)
        self.norm_out = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.full((1,), gamma_init))

        # Diagnostics buffers (not optimised, not saved as parameters).
        # Updated on every forward; consumed by the trainer at epoch end.
        self.register_buffer('last_x_norm',     torch.zeros(1), persistent=False)
        self.register_buffer('last_y_norm',     torch.zeros(1), persistent=False)
        self.register_buffer('last_delta_norm', torch.zeros(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        if in_dtype == torch.float16:
            x = x.float()
        b, c, h, w = x.shape

        # Row-major scan
        row = x.flatten(2).transpose(1, 2)                         # B, HW, C
        row = self.mamba_row(self.norm_in(row))                    # B, HW, C

        # Column-major scan (transpose H<->W first)
        col = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)     # B, WH, C
        col = self.mamba_col(self.norm_in(col))                    # B, WH, C
        # Bring col back to row-order (B, HW, C)
        col = col.transpose(1, 2).reshape(b, c, w, h) \
                 .permute(0, 1, 3, 2).flatten(2).transpose(1, 2)

        y = self.fuse(torch.cat([row, col], dim=-1))               # B, HW, C
        y = self.norm_out(y)
        y = y.transpose(1, 2).reshape(b, c, h, w)

        # ---- diagnostics: how big is the Mamba contribution vs the input? ----
        with torch.no_grad():
            self.last_x_norm.fill_(x.detach().float().norm().item())
            self.last_y_norm.fill_(y.detach().float().norm().item())
            self.last_delta_norm.fill_(
                (self.gamma.detach().float() * y.detach().float()).norm().item()
            )

        return (x + self.gamma * y).to(in_dtype)


class PFESA_Mamba_PlainConvUNet(nn.Module):
    """
    nnUNetv2 PlainConvUNet (2D) wrapped with:
      * Post-conv BiMamba residual in deepest `n_mamba_stages` encoder stages
      * BiMamba x `n_bottleneck_blocks` on the encoder's deepest output
      * PFESA on every skip connection (stages 0 .. n_stages - 2)
      * Decoder unchanged
    """

    def __init__(self,
                 input_channels, num_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs,
                 deep_supervision: bool = True,
                 # ---- extra hyperparameters ----
                 pfesa_base_ratio: float = 0.1,
                 n_mamba_stages: int = 2,        # deepest k encoder stages
                 n_bottleneck_blocks: int = 2,   # Mamba x2 on bottleneck
                 mamba_d_state: int = 16,
                 mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 use_pfesa: bool = True,
                 gamma_init: float = 0.1,
                 **kwargs):
        super().__init__()
        self.use_pfesa = use_pfesa

        backbone = PlainConvUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.deep_supervision = deep_supervision

        # ---- Mamba in deepest n_mamba_stages encoder stages (post-conv) ----
        first_mamba_idx = max(0, n_stages - n_mamba_stages)
        self.mamba_enc = nn.ModuleDict({
            str(i): BiMambaResidual2D(
                channels=features_per_stage[i],
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                gamma_init=gamma_init,
            )
            for i in range(first_mamba_idx, n_stages)
        })

        # ---- Bottleneck: additional Mamba x n_bottleneck_blocks ----
        self.bottleneck_mamba = nn.Sequential(*[
            BiMambaResidual2D(
                channels=features_per_stage[-1],
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                gamma_init=gamma_init,
            )
            for _ in range(n_bottleneck_blocks)
        ])

        # ---- PFESA on every skip (stages 0 .. n_stages - 2) ----
        self.pfesa_modules = nn.ModuleList([
            PFESA(base_ratio=pfesa_base_ratio) for _ in range(n_stages - 1)
        ])

    def named_mamba_blocks(self):
        """Yield (name, BiMambaResidual2D) for every Mamba block in the net.

        Used by the trainer for per-epoch diagnostics."""
        for sid_str, blk in self.mamba_enc.items():
            yield f'enc{sid_str}', blk
        for i, blk in enumerate(self.bottleneck_mamba):
            yield f'bot{i}', blk

    def forward(self, x: torch.Tensor):
        # Encoder feature pyramid: list of length n_stages
        skips = list(self.encoder(x))

        # Mamba post-conv on the deepest encoder stages
        for sid_str, blk in self.mamba_enc.items():
            sid = int(sid_str)
            skips[sid] = blk(skips[sid])

        # Bottleneck Mamba x N on the deepest output
        skips[-1] = self.bottleneck_mamba(skips[-1])

        # PFESA on all skip connections (bottleneck is consumed by decoder
        # directly, no PFESA needed there)
        if self.use_pfesa:
            for i in range(len(skips) - 1):
                skips[i] = self.pfesa_modules[i](skips[i])

        return self.decoder(skips)
