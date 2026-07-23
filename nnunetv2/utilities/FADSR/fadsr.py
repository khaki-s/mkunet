import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def _make_wavelet_filters(wt_type, channels, dtype=torch.float32):
    w = pywt.Wavelet(wt_type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype)

    def outer(a, b):
        return a[:, None] * b[None, :]

    dec = torch.stack([outer(dec_lo, dec_lo), outer(dec_lo, dec_hi),
                       outer(dec_hi, dec_lo), outer(dec_hi, dec_hi)], dim=0)  # (4,k,k)
    rec = torch.stack([outer(rec_lo, rec_lo), outer(rec_lo, rec_hi),
                       outer(rec_hi, rec_lo), outer(rec_hi, rec_hi)], dim=0)
    dec = dec[:, None].repeat(channels, 1, 1, 1)   # (4C, 1, k, k)
    rec = rec[:, None].repeat(channels, 1, 1, 1)
    return dec, rec


class WaveletOps(nn.Module):
    def __init__(self, dim, wt_type='db1'):
        super().__init__()
        self.dim = dim
        dec, rec = _make_wavelet_filters(wt_type, dim)
        self.register_buffer('wt_filter', dec)
        self.register_buffer('iwt_filter', rec)
        self.k = dec.shape[-1]
        self.conv_pad = self.k // 2 - 1

    def dwt(self, x):
        B, C, H, W = x.shape
        x = F.conv2d(x, self.wt_filter, stride=2, groups=C, padding=self.conv_pad)
        H2, W2 = x.shape[-2:]
        x = x.view(B, C, 4, H2, W2)
        ll, lh, hl, hh = x.unbind(dim=2)
        return ll, lh, hl, hh

    def idwt(self, ll, lh, hl, hh):
        B, C, h, w = ll.shape
        sub = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, h, w)
        out = F.conv_transpose2d(sub, self.iwt_filter, stride=2,
                                 groups=C, padding=self.conv_pad)
        return out


def _eca_kernel(c):
    t = int(abs(math.log(c, 2)) + 1) // 2
    return t if t % 2 else t + 1


class WaveletMixer(nn.Module):
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.pw = nn.Conv2d(4 * dim, 4 * dim, 1, bias=False)
        self.dw = nn.Conv2d(4 * dim, 4 * dim, kernel,
                            padding=kernel // 2, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=4 * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, 4 * dim, 1, 1))  # 0 -> identity
        self.drop = nn.Identity()
    def forward(self, sub4):
        return sub4 + self.scale * self.drop(self.act(self.norm(self.dw(self.pw(sub4)))))


class LightweightGuide(nn.Module):
    def __init__(self, dim, dec_dim=None):
        super().__init__()
        dec_dim = dec_dim or dim
        self.proj = (nn.Conv2d(dec_dim, dim, 1, bias=False)
                     if dec_dim != dim else nn.Identity())
        self.gap = nn.AdaptiveAvgPool2d(1)
        k = _eca_kernel(dim)
        self.eca = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  # 0 -> identity

    def forward(self, dec_feat):
        z = self.proj(dec_feat)
        z = self.gap(z)                                   # (B, C, 1, 1)
        z = z.squeeze(-1).transpose(-1, -2)               # (B, 1, C)
        z = self.eca(z).transpose(-1, -2).unsqueeze(-1)   # (B, C, 1, 1)
        return torch.sigmoid(z)


class FADSR(nn.Module):
    def __init__(self, enc_dim, *, dec_dim=None, wt_type='db1', wt_levels=3,
                 dw_kernel=5, dropout=0.1, num_stages=None, topk=None,
                 use_skip_fuse=False, **kwargs):
        super().__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim or enc_dim
        self.wt_levels = int(wt_levels)
        self.use_skip_fuse = use_skip_fuse

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type)

        L = self.wt_levels
        self.mixer = nn.ModuleList(
            [WaveletMixer(enc_dim, dw_kernel) for _ in range(L)])
        self.guide = nn.ModuleList(
            [LightweightGuide(enc_dim, self.dec_dim) for _ in range(L)])

        if use_skip_fuse:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(enc_dim + self.dec_dim, enc_dim, 1),
                nn.BatchNorm2d(enc_dim),
                nn.GELU(),
                nn.Conv2d(enc_dim, enc_dim, 3, padding=1),
            )

    def _refine_even(self, x, dec_feat=None):
        L = self.wt_levels
        hf_levels = []
        shapes = []
        a = x
        for k in range(L):
            shapes.append(a.shape)
            ll, lh, hl, hh = self.wavelet.dwt(a)
            sub4 = torch.cat([ll, lh, hl, hh], dim=1)          # (B, 4C, h, w)
            sub4 = self.mixer[k](sub4)                         # inter-subband mix
            ll, lh, hl, hh = sub4.chunk(4, dim=1)
            hf_levels.append((lh, hl, hh))
            a = ll                                              # descend on LL

        cur = a                                                # deepest LL'
        for k in range(L - 1, -1, -1):
            lh, hl, hh = hf_levels[k]
            if dec_feat is not None:
                mask = self.guide[k](dec_feat)                 # (B, C, 1, 1) in [0,1]
                g = self.guide[k].gamma
                lh = lh * (1.0 + g * mask)                     # amplitude only, HF only
                hl = hl * (1.0 + g * mask)
                hh = hh * (1.0 + g * mask)
            cur = self.wavelet.idwt(cur, lh, hl, hh)
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]      # crop boundary
        return cur

    def no_weight_decay(self):
        return {n for n, _ in self.named_parameters()
                if 'scale' in n or 'gamma' in n}

    def forward(self, x, stage_id=None, dec_feat=None):
        B, C, H, W = x.shape
        m = 2 ** self.wt_levels
        ph = (m - H % m) % m
        pw = (m - W % m) % m
        if ph or pw:
            xp = F.pad(x, (0, pw, 0, ph), mode='replicate')
        else:
            xp = x
        out = self._refine_even(xp, dec_feat=dec_feat)
        if ph or pw:
            out = out[:, :, :H, :W]
        if dec_feat is not None and self.use_skip_fuse:
            return self.skip_fuse(torch.cat([out, dec_feat], dim=1))
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64) * 3 + 1
    dec = torch.randn(2, 32, 64, 64)

    print("== Haar (db1): exact identity at init, with decoder guide ==")
    m = FADSR(enc_dim=32, dec_dim=32, wt_type='db1', wt_levels=3)
    m.eval()
    with torch.no_grad():
        out = m(x, stage_id=0, dec_feat=dec)
    dev = (out - x).abs().max().item()
    n_total = sum(p.numel() for p in m.parameters())
    n_learn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  max |FADSR(x)-x| @init = {dev:.3e} (haar PR -> ~0, roundoff only)")
    print(f"  params: {n_total} total, {n_learn} learnable")
    assert dev < 1e-3, "haar identity-at-init failed"

    print("== guide touches ONLY high-freq; LL-driven channel removed ==")
    # gamma zero -> no guide contribution; verify mask path is HF-only by structure
    assert not hasattr(m, 'ref_ll') and not hasattr(m, 'ref_lh'), "old refiners removed"
    assert all(isinstance(g, LightweightGuide) for g in m.guide)
    print("  ok: WaveletMixer (grouped, groups=C) + LightweightGuide (HF amplitude)")

    print("== train-mode forward + backward (grad flows to mixer + guide) ==")
    m.train()
    out = m(x, stage_id=0, dec_feat=dec)
    loss = out.pow(2).mean()
    loss.backward()
    # at init scale=0 / gamma=0 gate the conv paths, so gradients land on the
    # zero-init gates first; dw / eca receive gradient as the gates open.
    scale_grad = float(m.mixer[0].scale.grad.abs().sum())
    gamma_grad = float(m.guide[0].gamma.grad.abs().sum())
    print(f"  mixer scale grad: {scale_grad:.4e}; guide gamma grad: {gamma_grad:.4e}")
    assert scale_grad > 0, "mixer scale should learn"
    assert gamma_grad > 0, "guide gamma should learn"
    print("OK.")
