"""Attribute the v1->v2 change to db2fix vs LL/HF split, using the TRAINED weights.

FADSR(x) = x + scale * branch(x)  ... but only if the wavelet transform is
perfect-reconstruction. With the buggy conv_pad it is really

    FADSR(x) = R(x) + scale * branch'(x)

where R is a FIXED, NON-LEARNED lossy reconstruction. So there are two
independent knobs:
  - db2fix  changes  ||R(x)-x||/||x||   (ungated)
  - LLHF    changes  the branch         (gated by `scale`)

We measure both separately at the real nnU-Net 2d skip resolutions.
No pywt in this env -> filters hardcoded (verified against blocks.py logic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BANKS = {
    'db2': dict(
        dec_lo=[-0.12940952255092145, 0.22414386804185735,
                0.836516303737469, 0.48296291314469025],
        dec_hi=[-0.48296291314469025, 0.836516303737469,
                -0.22414386804185735, -0.12940952255092145],
        rec_lo=[0.48296291314469025, 0.836516303737469,
                0.22414386804185735, -0.12940952255092145],
        rec_hi=[-0.12940952255092145, -0.22414386804185735,
                0.836516303737469, -0.48296291314469025],
    ),
}


def make_filters(wt, channels):
    b = BANKS[wt]
    g = lambda k: torch.tensor(b[k][::-1] if k.startswith('dec') else b[k],
                               dtype=torch.float32)
    dl, dh, rl, rh = g('dec_lo'), g('dec_hi'), g('rec_lo'), g('rec_hi')
    o = lambda a, c: a[:, None] * c[None, :]
    dec = torch.stack([o(dl, dl), o(dl, dh), o(dh, dl), o(dh, dh)], 0)
    rec = torch.stack([o(rl, rl), o(rl, rh), o(rh, rl), o(rh, rh)], 0)
    return dec[:, None].repeat(channels, 1, 1, 1), rec[:, None].repeat(channels, 1, 1, 1)


class WaveletOps(nn.Module):
    def __init__(self, dim, wt_type='db2', exact_recon=False):
        super().__init__()
        dec, rec = make_filters(wt_type, dim)
        self.register_buffer('wt_filter', dec)
        self.register_buffer('iwt_filter', rec)
        self.k = dec.shape[-1]
        self.conv_pad = (self.k - 2) if exact_recon else (self.k // 2 - 1)

    def dwt(self, x):
        C = x.shape[1]
        y = F.conv2d(x, self.wt_filter, stride=2, groups=C, padding=self.conv_pad)
        B, _, h, w = y.shape
        return y.view(B, C, 4, h, w).unbind(2)

    def idwt(self, ll, lh, hl, hh):
        B, C, h, w = ll.shape
        sub = torch.stack([ll, lh, hl, hh], 2).reshape(B, C * 4, h, w)
        return F.conv_transpose2d(sub, self.iwt_filter, stride=2,
                                  groups=C, padding=self.conv_pad)


class LLLargeKernel(nn.Module):
    def __init__(self, dim, kernel=5, band_kernel=7, dilation=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim, bias=False)
        self.dw_dil = nn.Conv2d(dim, dim, band_kernel,
                                padding=band_kernel // 2 * dilation,
                                dilation=dilation, groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.dilation, self.band_kernel = dilation, band_kernel

    def forward(self, ll):
        y = self.dw(ll)
        s = min(y.shape[-2], y.shape[-1])
        eff = min(self.dilation, max(1, (s - 1) // max(1, self.band_kernel - 1)))
        if eff == self.dilation:
            y = self.dw_dil(y)
        else:
            y = F.conv2d(y, self.dw_dil.weight, None, 1,
                         self.band_kernel // 2 * eff, eff, self.dw_dil.groups)
        return self.pw(y), eff


class HFEdgeEnhance(nn.Module):
    def __init__(self, dim, kernel=3):
        super().__init__()
        p = kernel // 2
        self.dw_lh = nn.Conv2d(dim, dim, (1, kernel), padding=(0, p), groups=dim, bias=False)
        self.dw_hl = nn.Conv2d(dim, dim, (kernel, 1), padding=(p, 0), groups=dim, bias=False)
        self.dw_hh = nn.Conv2d(dim, dim, kernel, padding=p, groups=dim, bias=False)
        self.pw = nn.Conv2d(3 * dim, 3 * dim, 1, bias=False)

    def forward(self, lh, hl, hh):
        y = torch.cat([self.dw_lh(lh), self.dw_hl(hl), self.dw_hh(hh)], 1)
        return self.pw(y)


class WaveletMixer(nn.Module):
    def __init__(self, dim, kernel=5, dilation=1, split_ll_hf=False):
        super().__init__()
        self.dim, self.split_ll_hf = dim, split_ll_hf
        if split_ll_hf:
            self.ll_path = LLLargeKernel(dim)
            self.hf_path = HFEdgeEnhance(dim)
        else:
            self.pw = nn.Conv2d(4 * dim, 4 * dim, 1, bias=False)
            self.dw = nn.Conv2d(4 * dim, 4 * dim, kernel,
                                padding=kernel // 2 * dilation,
                                groups=dim, bias=False, dilation=dilation)
        self.norm = nn.GroupNorm(dim, 4 * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, 4 * dim, 1, 1))
        self.last_eff = None

    def forward(self, sub4, gate=True):
        if self.split_ll_hf:
            ll, lh, hl, hh = sub4.chunk(4, 1)
            llo, eff = self.ll_path(ll)
            self.last_eff = eff
            y = torch.cat([llo, self.hf_path(lh, hl, hh)], 1)
        else:
            y = self.dw(self.pw(sub4))
        y = self.act(self.norm(y))
        return sub4 + (self.scale * y if gate else torch.zeros_like(sub4))


class FADSR(nn.Module):
    def __init__(self, enc_dim, wt_levels=3, dw_kernel=7,
                 split_ll_hf=False, exact_recon=False):
        super().__init__()
        self.wt_levels = wt_levels
        self.exact_recon = exact_recon
        self.wavelet = WaveletOps(enc_dim, 'db2', exact_recon)
        if split_ll_hf:
            self.mixer = nn.ModuleList([WaveletMixer(enc_dim, split_ll_hf=True)
                                        for _ in range(wt_levels)])
        else:
            self.mixer = nn.ModuleList(
                [WaveletMixer(enc_dim, dw_kernel, 1) for _ in range(wt_levels - 1)]
                + [WaveletMixer(enc_dim, dw_kernel, 2)])

    def _refine(self, x, gate):
        shapes, hf = [], []
        a = x
        for k in range(self.wt_levels):
            shapes.append(a.shape)
            if self.exact_recon:
                h, w = a.shape[-2:]
                if h % 2 or w % 2:
                    a = F.pad(a, (0, w % 2, 0, h % 2), mode='replicate')
            ll, lh, hl, hh = self.wavelet.dwt(a)
            sub4 = self.mixer[k](torch.cat([ll, lh, hl, hh], 1), gate)
            ll, lh, hl, hh = sub4.chunk(4, 1)
            hf.append((lh, hl, hh))
            a = ll
        cur = a
        for k in range(self.wt_levels - 1, -1, -1):
            cur = self.wavelet.idwt(cur, *hf[k])
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]
        return cur

    def forward(self, x, gate=True):
        H, W = x.shape[-2:]
        m = 2 ** self.wt_levels
        ph, pw = (m - H % m) % m, (m - W % m) % m
        xp = F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x
        return self._refine(xp, gate)[:, :, :H, :W]


R = '/root/autodl-tmp/data/nnUNet_results/Dataset137_BraTS2021'
RUNS = {
    'v1  FADSR_100epochs        (db2 BUG, no split)':
        ('nnUNetTrainer_FADSR_100epochs', False, False),
    'v2  FADSR_LLHF_db2fix      (db2 fixed + split)':
        ('nnUNetTrainer_FADSR_LLHF_db2fix_100epochs', True, True),
}
SIZES = [(192, 160, 32), (96, 80, 64), (48, 40, 128), (24, 20, 256), (12, 10, 512)]
rel = lambda a, b: (a - b).norm().item() / b.norm().item()

torch.manual_seed(0)
X = {C: torch.randn(2, C, H, W) for H, W, C in SIZES}

for label, (trainer, split, fix) in RUNS.items():
    ck = torch.load(f'{R}/{trainer}__nnUNetPlans__2d/fold_0/checkpoint_best.pth',
                    map_location='cpu')
    sd = {k: v for k, v in ck['network_weights'].items() if k.startswith('fadsr_modules.')}
    del ck
    print(f'\n===== {label}')
    print(f'{"skip":>13} {"ungated  R(x)":>14} {"full f(x)":>11} {"gated part":>12} {"ratio":>8}')
    for si, (H, W, C) in enumerate(SIZES):
        mod = FADSR(C, split_ll_hf=split, exact_recon=fix).eval()
        sub = {k[len(f'fadsr_modules.{si}.'):]: v
               for k, v in sd.items() if k.startswith(f'fadsr_modules.{si}.')}
        missing, unexpected = mod.load_state_dict(sub, strict=False)
        assert not [m for m in missing if 'filter' not in m], f'missing {missing}'
        assert not unexpected, f'unexpected {unexpected}'
        x = X[C]
        with torch.no_grad():
            ung = mod(x, gate=False)     # scale forced off -> pure reconstruction
            full = mod(x, gate=True)     # trained scale
        r_ung, r_full = rel(ung, x), rel(full, x)
        r_gate = rel(full, ung)          # what the LEARNED branch adds on top
        eff = mod.mixer[0].last_eff
        print(f'{H:4d}x{W:<3d} C={C:<4d} {100*r_ung:12.3f}% {100*r_full:10.3f}%'
              f' {100*r_gate:11.3f}% {r_gate/max(r_ung,1e-12):7.2f}x'
              + (f'   (LL eff_dil={eff})' if eff is not None else ''))
    del sd
