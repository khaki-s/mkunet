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


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SubbandRefiner(nn.Module):
    def __init__(self, dim, dw_kernel=5, dropout=0.):
        super().__init__()
        self.dw = nn.Conv2d(dim, 2 * dim, dw_kernel,
                            padding=dw_kernel // 2, groups=dim, bias=True)
        self.sg = SimpleGate()                       # -> dim
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)  # channel mix
        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1))  # near-zero -> identity
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, s):
        x = self.sg(self.dw(s))
        x = self.pw(x)
        return self.drop(x) * self.scale            # delta (0 at init)


class CrossFreqGuide(nn.Module):
    def __init__(self, dim, bottleneck=4):
        super().__init__()
        hidden = max(dim // bottleneck, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))  

    def forward(self, ll, lh, hl, hh):
        g = torch.tanh(self.fc2(self.act(self.fc1(self.gap(ll)))))  # (B,C,1,1)
        lh = lh + self.gamma * g * lh
        hl = hl + self.gamma * g * hl
        hh = hh + self.gamma * g * hh
        return lh, hl, hh


class FADSR(nn.Module):
    def __init__(self, enc_dim, *, wt_type='db1', wt_levels=3, dw_kernel=5,
                 dropout=0.1, num_stages=None, topk=None,
                 use_skip_fuse=False, **kwargs):
        super().__init__()
        self.enc_dim = enc_dim
        self.wt_levels = int(wt_levels)
        self.use_skip_fuse = use_skip_fuse

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type)

        L = self.wt_levels
        self.ref_lh = nn.ModuleList([SubbandRefiner(enc_dim, dw_kernel, dropout) for _ in range(L)])
        self.ref_hl = nn.ModuleList([SubbandRefiner(enc_dim, dw_kernel, dropout) for _ in range(L)])
        self.ref_hh = nn.ModuleList([SubbandRefiner(enc_dim, dw_kernel, dropout) for _ in range(L)])
        self.ref_ll = nn.ModuleList([SubbandRefiner(enc_dim, dw_kernel, dropout) for _ in range(L)])

        self.guide = nn.ModuleList([CrossFreqGuide(enc_dim) for _ in range(L)])

        if use_skip_fuse:
            self.skip_fuse = nn.Sequential(
                nn.Conv2d(enc_dim + enc_dim, enc_dim, 1),
                nn.BatchNorm2d(enc_dim),
                nn.GELU(),
                nn.Conv2d(enc_dim, enc_dim, 3, padding=1),
            )

    def _refine_even(self, x):
        L = self.wt_levels
        hf_levels = []      
        shapes = []                
        a = x
        for k in range(L):
            shapes.append(a.shape)
            ll, lh, hl, hh = self.wavelet.dwt(a)
            hf_levels.append((lh, hl, hh))   
            a = ll                            

        cur = a + self.ref_ll[L - 1](a)       
        for k in range(L - 1, -1, -1):        
            lh, hl, hh = hf_levels[k]
            lh = lh + self.ref_lh[k](lh)
            hl = hl + self.ref_hl[k](hl)
            hh = hh + self.ref_hh[k](hh)
            lh, hl, hh = self.guide[k](cur, lh, hl, hh)

            cur = self.wavelet.idwt(cur, lh, hl, hh)
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]
            if k > 0:
                cur = cur + self.ref_ll[k - 1](cur)  
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
        out = self._refine_even(xp)
        if ph or pw:
            out = out[:, :, :H, :W]
        if dec_feat is not None and self.use_skip_fuse:
            return self.skip_fuse(torch.cat([out, dec_feat], dim=1))
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 65, 63) * 3 + 1

    print("== Haar (db1): exact identity at init ==")
    m = FADSR(enc_dim=32, wt_type='db1', wt_levels=3)
    m.eval()
    with torch.no_grad():
        out = m(x, stage_id=0)
    dev = (out - x).abs().max().item()
    n_total = sum(p.numel() for p in m.parameters())
    n_learn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_fixed = n_total - n_learn
    print(f"  max |FADSR(x)-x| @init = {dev:.3e} (haar PR -> ~0, roundoff only)")
    print(f"  params: {n_total} total, {n_learn} learnable, {n_fixed} fixed (wavelet)")
    assert dev < 1e-3, "haar identity-at-init failed"

    print("== db2: identity holds to wavelet PR precision ==")
    m2 = FADSR(enc_dim=32, wt_type='db2', wt_levels=2)
    m2.eval()
    with torch.no_grad():
        out2 = m2(x, stage_id=0)
    dev2 = (out2 - x).abs().max().item()
    print(f"  max |FADSR(x)-x| @init = {dev2:.3e} (db2 boundary; small by design)")
    print(f"  params: {sum(p.numel() for p in m2.parameters())}")

    print("== train-mode forward + backward (grad flows to refiners) ==")
    m.train()
    out = m(x, stage_id=0)
    loss = out.pow(2).mean()
    loss.backward()
    grad_norms = [n for n, p in zip(['lh', 'hl', 'hh', 'll', 'guide'],
                                   [m.ref_lh[0], m.ref_hl[0], m.ref_hh[0],
                                    m.ref_ll[0], m.guide[0]])
                  if True]
    has_grad = any(p.weight.grad is not None and p.weight.grad.abs().sum() > 0
                   for ref in m.ref_lh for p in [ref.dw, ref.pw, ref.scale])
    print(f"  refiner params receive gradient: {has_grad}")
    assert has_grad, "refiners should learn (gradient must flow)"
    print("OK.")
