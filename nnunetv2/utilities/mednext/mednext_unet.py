from __future__ import annotations

import os
from typing import List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from nnunetv2.utilities.mednext.blocks import MedNeXtBlock
from nnunetv2.utilities.mednext.pfesa import PFESA, PFESA3D

def _conv_cls(dim: str):
    return nn.Conv2d if dim == '2d' else nn.Conv3d


def _convT_cls(dim: str):
    return nn.ConvTranspose2d if dim == '2d' else nn.ConvTranspose3d


def _norm_cls(dim: str, num_channels: int) -> nn.Module:
    """BatchNorm to match the MKUNet block / GAG design (dim-aware)."""
    return nn.BatchNorm2d(num_channels) if dim == '2d' else nn.BatchNorm3d(num_channels)


class GroupedAttentionGate(nn.Module):

    def __init__(self, F_g: int, F_l: int, F_int: int,
                 kernel_size: int = 3, groups: int = None,
                 dim: str = '2d'):
        super().__init__()
        conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        if kernel_size == 1:
            groups = 1
        if groups is None:
            groups = max(F_int, 1)            # MKUNet default: channels // 2
        # grouped conv needs groups | F_g and groups | F_int
        if groups <= 0 or F_g % groups != 0 or F_int % groups != 0:
            groups = 1
        pad = kernel_size // 2

        self.W_g = nn.Sequential(
            conv(F_g, F_int, kernel_size, stride=1, padding=pad,
                 groups=groups, bias=True),
            _norm_cls(dim, F_int))
        self.W_x = nn.Sequential(
            conv(F_l, F_int, kernel_size, stride=1, padding=pad,
                 groups=groups, bias=True),
            _norm_cls(dim, F_int))
        self.psi = nn.Sequential(
            conv(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            _norm_cls(dim, 1),
            nn.Sigmoid())
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class _DecoderProxy(nn.Module):
    def __init__(self, parent: 'MedNeXtUNet_nnUNet'):
        super().__init__()
        object.__setattr__(self, '_parent', parent)

    @property
    def deep_supervision(self) -> bool:
        return self._parent.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value: bool) -> None:
        self._parent.deep_supervision = bool(value)


class MedNeXtUNet_nnUNet(nn.Module):
    def __init__(self,
                 # ---- mandatory nnUNetv2 arch kwargs ----
                 input_channels: int,
                 num_classes: int,
                 n_stages: int,
                 features_per_stage: Sequence[int],
                 conv_op: Type[nn.Module],
                 kernel_sizes: Sequence,
                 strides: Sequence,
                 n_conv_per_stage: Sequence[int],
                 n_conv_per_stage_decoder: Sequence[int],
                 conv_bias: bool,
                 norm_op,
                 norm_op_kwargs,
                 dropout_op,
                 dropout_op_kwargs,
                 nonlin,
                 nonlin_kwargs,
                 deep_supervision: bool = True,
                 # ---- MedNeXt knobs ----
                 exp_r: Union[int, Sequence[int]] = 2,
                 exp_r_decoder: Optional[Sequence[int]] = None,
                 grn: bool = True,  # kept for API compat; multi-kernel block has no GRN
                 use_checkpoint: bool = True,
                 dw_kernel_size: int = 5,
                 layer_scale_init: float = 1e-6,
                 max_drop_path_rate: float = 0.1,
                 # ---- skip-connection attention knobs ----
                 use_pfesa: bool = True,        # PFESA edge-structure attention on skips
                 use_gag: bool = True,          # GroupedAttentionGate on skips
                 pfesa_base_ratio: float = 0.1,
                 gag_kernel: int = 3,
                 **kwargs):
        super().__init__()

        if conv_op is nn.Conv2d:
            dim = '2d'
        elif conv_op is nn.Conv3d:
            dim = '3d'
        else:
            raise ValueError(f'Unsupported conv_op: {conv_op}')
        self.dim = dim
        self.n_stages = n_stages
        self.deep_supervision = deep_supervision
        # torch.utils.checkpoint is broken on Windows for some configs
        # and pure CPU runs; disable there.
        self.use_checkpoint = bool(use_checkpoint) and (os.name != 'nt')

        # Normalise per-stage expansion ratios to list lengths.
        # Encoder: length n_stages (index = stage s).
        # Decoder: length n_stages-1 (index = di, deepest-first); if None,
        # mirror the encoder (dec stage s reuses exp_r[s]).
        def _norm(seq, length):
            if isinstance(seq, int):
                return [seq] * length
            seq = list(seq)
            if len(seq) < length:
                seq = seq + [seq[-1]] * (length - len(seq))
            elif len(seq) > length:
                seq = seq[:length]
            return seq

        exp_r = _norm(exp_r, n_stages)
        if exp_r_decoder is None:
            # mirror encoder: dec stage s reuses exp_r[s] (di unused)
            dec_exp_fn = lambda s, di: exp_r[s]
        else:
            exp_r_decoder = _norm(exp_r_decoder, n_stages - 1)
            # dec loop index di is deepest-first, matching exp_r_decoder order
            dec_exp_fn = lambda s, di: exp_r_decoder[di]

        enc_blocks = list(n_conv_per_stage)
        dec_blocks = list(n_conv_per_stage_decoder)
        assert len(enc_blocks) == n_stages
        assert len(dec_blocks) == n_stages - 1

        Conv = _conv_cls(dim)
        ConvT = _convT_cls(dim)

        def _scalar(seq_or_int, agg=max):
            if isinstance(seq_or_int, int):
                return seq_or_int
            seq = list(seq_or_int)
            return seq[0] if len(set(seq)) == 1 else int(agg(seq))

        # Linear DropPath schedule across ALL MedNeXt blocks (enc + dec),
        # 0 -> max_drop_path_rate.
        total_blocks = sum(enc_blocks) + sum(dec_blocks)
        dp_rates = [
            max_drop_path_rate * i / max(total_blocks - 1, 1)
            for i in range(total_blocks)
        ]
        dp_iter = iter(dp_rates)

        # Convert the single max-kernel knob into the odd depthwise kernel
        # set used by the multi-kernel block, e.g. dw_kernel_size=5 -> [1,3,5].
        kernel_sizes = list(range(1, int(dw_kernel_size) + 1, 2))

        def _make_block(ch, exp_r_val):
            return MedNeXtBlock(
                ch, ch,
                exp_r=exp_r_val,
                kernel_sizes=kernel_sizes,
                do_res=True,
                dim=dim,
                layer_scale_init=layer_scale_init,
                drop_path_rate=next(dp_iter),
            )

        # ---- Encoder: stem on stage 0, then strided conv between stages ----
        # Stage 0: lift to features_per_stage[0] via a 1-channel-per-output
        # conv (matches PlainConvUNet's first conv before stacking blocks).
        ks0 = _scalar(kernel_sizes[0])
        self.stem = Conv(input_channels, features_per_stage[0],
                         kernel_size=ks0, padding=ks0 // 2, bias=conv_bias)

        # Per-stage MedNeXt block stack.
        self.enc_stages = nn.ModuleList()
        for s in range(n_stages):
            blocks = nn.Sequential(*[
                _make_block(features_per_stage[s], exp_r[s])
                for _ in range(enc_blocks[s])
            ])
            self.enc_stages.append(blocks)

        # Down samplers between stages: cheap strided conv (no MedNeXt
        # block at the resolution change). Stride taken from plans.
        self.down_blocks = nn.ModuleList()
        for s in range(n_stages - 1):
            st = strides[s + 1]
            st_t = tuple(st) if not isinstance(st, int) else (st,) * (2 if dim == '2d' else 3)
            self.down_blocks.append(
                Conv(features_per_stage[s], features_per_stage[s + 1],
                     kernel_size=st_t, stride=st_t, bias=conv_bias)
            )

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        self.reduce_blocks = nn.ModuleList()   # after concat(skip, up)
        self.dec_stages = nn.ModuleList()
        # Skip-connection attention: PFESA (edge-structure, parameter-free)
        # + GroupedAttentionGate (gated by the upsampled decoder signal).
        self.use_pfesa = use_pfesa
        self.use_gag = use_gag
        self.pfesa_modules = nn.ModuleList() if use_pfesa else None
        self.gag_blocks = nn.ModuleList() if use_gag else None
        PFESACls = PFESA if dim == '2d' else PFESA3D
        for di, s in enumerate(range(n_stages - 2, -1, -1)):
            st = strides[s + 1]
            st_t = tuple(st) if not isinstance(st, int) else (st,) * (2 if dim == '2d' else 3)
            # Transposed conv to upsample features_per_stage[s+1] -> [s].
            self.up_blocks.append(
                ConvT(features_per_stage[s + 1], features_per_stage[s],
                      kernel_size=st_t, stride=st_t, bias=conv_bias)
            )
            # 1x1 to fuse the concat(up, skip) back to features[s] channels.
            self.reduce_blocks.append(
                Conv(2 * features_per_stage[s], features_per_stage[s],
                     kernel_size=1, bias=conv_bias)
            )
            self.dec_stages.append(nn.Sequential(*[
                _make_block(features_per_stage[s], dec_exp_fn(s, di))
                for _ in range(dec_blocks[di])
            ]))
            # Per-stage skip attention (channels = features_per_stage[s]).
            if use_pfesa:
                self.pfesa_modules.append(PFESACls(base_ratio=pfesa_base_ratio))
            if use_gag:
                ch = features_per_stage[s]
                f_int = max(ch // 2, 1)
                self.gag_blocks.append(
                    GroupedAttentionGate(ch, ch, f_int,
                                         kernel_size=gag_kernel, dim=dim))

        # Segmentation heads — one per output resolution we emit.
        # Order in self.seg_heads: index s corresponds to stage s
        # (full res at s=0).
        Conv1 = _conv_cls(dim)
        self.seg_heads = nn.ModuleList([
            Conv1(features_per_stage[s], num_classes, kernel_size=1)
            for s in range(n_stages - 1)
        ])

        # Proxy so nnUNetv2's base trainer can toggle DS.
        self.decoder = _DecoderProxy(self)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d,
                              nn.ConvTranspose2d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm, nn.InstanceNorm2d,
                                nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _crop_pad_like(x, ref):
        if x.shape[2:] == ref.shape[2:]:
            return x
        diffs = [r - xs for xs, r in zip(x.shape[2:], ref.shape[2:])]
        pads = []
        for d in reversed(diffs):
            pads += [d // 2, d - d // 2] if d >= 0 else [0, 0]
        if any(p != 0 for p in pads):
            x = F.pad(x, pads)
        slices = [slice(None), slice(None)]
        for xs, r in zip(x.shape[2:], ref.shape[2:]):
            if xs > r:
                start = (xs - r) // 2
                slices.append(slice(start, start + r))
            else:
                slices.append(slice(None))
        return x[tuple(slices)]

    def _run_seq(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            for m in seq:
                x = cp.checkpoint(m, x, use_reentrant=False)
            return x
        return seq(x)

    def forward(self, x):
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for s in range(self.n_stages):
            x = self._run_seq(self.enc_stages[s], x)
            skips.append(x)
            if s < self.n_stages - 1:
                x = self.down_blocks[s](x)

        x = skips[-1]
        outputs: List[torch.Tensor] = []
        for di in range(self.n_stages - 1):
            stage_idx = self.n_stages - 2 - di
            x = self.up_blocks[di](x)                 # decoder gating signal g
            skip = skips[stage_idx]                   # encoder skip x
            # GAG adds Wg(g)+Wx(x), so align spatial shapes first.
            x = self._crop_pad_like(x, skip)
            # PFESA: parameter-free edge-structure attention on the skip.
            # FFT needs float32; run with AMP autocast disabled (PFESA3D does
            # not self-cast like the 2D variant, so this guards mixed-precision
            # training in both 2D and 3D).
            if self.pfesa_modules is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    skip = self.pfesa_modules[di](skip.float()).to(skip.dtype)
            # GAG: gate the (refined) skip with the upsampled decoder signal.
            if self.gag_blocks is not None:
                skip = self.gag_blocks[di](g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
            x = self.reduce_blocks[di](x)
            x = self._run_seq(self.dec_stages[di], x)
            outputs.append(self.seg_heads[stage_idx](x))

        outputs = outputs[::-1]
        if self.deep_supervision:
            return outputs
        return outputs[0]
