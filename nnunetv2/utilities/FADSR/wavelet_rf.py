import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.wtconv.util.wavelet import (
    create_2d_wavelet_filter, wavelet_2d_transform, inverse_2d_wavelet_transform,
)


class WaveletRFBlock(nn.Module):
    """Encoder-only receptive-field enlargement via cascaded DWT + depthwise conv + IDWT.

    RF after `levels` cascades with wavelet filter length Lw and conv kernel k:
        RF = (2**levels - 1) * (Lw - 1) + (k - 1) * 2**levels + 1
    (Lw=2 for haar, Lw=4 for db2). No cross-subband mixing, no gating -- purely
    spatial RF growth, kept orthogonal to FADSR's frequency-domain skip refinement.
    """

    def __init__(self, channels, kernel_size=3, levels=2, wt_type='haar'):
        super().__init__()
        self.channels = channels
        self.levels = levels

        wt_filter, iwt_filter = create_2d_wavelet_filter(wt_type, channels, channels, torch.float)
        self.register_buffer('wt_filter', wt_filter)
        self.register_buffer('iwt_filter', iwt_filter)

        self.convs = nn.ModuleList([
            nn.Conv2d(channels * 4, channels * 4, kernel_size, padding=kernel_size // 2,
                      groups=channels * 4, bias=False)
            for _ in range(levels)
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(channels, channels * 4) for _ in range(levels)])
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))  # 0 -> identity at init

    def forward(self, x):
        B, C, H, W = x.shape
        m = 2 ** self.levels
        pad_h, pad_w = (m - H % m) % m, (m - W % m) % m
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate') if (pad_h or pad_w) else x

        shapes, bands = [], []
        ll = xp
        for i in range(self.levels):
            shapes.append(ll.shape)
            sub4 = wavelet_2d_transform(ll, self.wt_filter)          # (B, C, 4, h, w)
            b, c, _, h, w = sub4.shape
            sub4 = sub4.reshape(b, c * 4, h, w)
            sub4 = self.act(self.norms[i](self.convs[i](sub4)))
            sub4 = sub4.reshape(b, c, 4, h, w)
            ll, lh, hl, hh = sub4.unbind(dim=2)
            bands.append((lh, hl, hh))

        cur = ll
        for i in range(self.levels - 1, -1, -1):
            lh, hl, hh = bands[i]
            cur = torch.stack([cur, lh, hl, hh], dim=2)
            cur = inverse_2d_wavelet_transform(cur, self.iwt_filter)
            cur = cur[:, :, :shapes[i][2], :shapes[i][3]]

        out = cur[:, :, :H, :W]
        return x + self.scale * out
