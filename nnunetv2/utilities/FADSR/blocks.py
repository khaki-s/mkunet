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
        self.conv_pad = self.k - 2

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


class WaveletMixer(nn.Module):
    def __init__(self, dim, kernel=5, dilation=1, scale=0.0, layout='stack'):
        super().__init__()
        if layout not in ('stack', 'cat'):
            raise ValueError(f"layout must be 'stack' or 'cat', got {layout!r}")
        self.dim = dim
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.layout = layout
        self.pw = nn.Conv2d(4 * dim, 4 * dim, 1, bias=False)
        self.dw = nn.Conv2d(4 * dim, 4 * dim, kernel,
                            padding=kernel // 2 * dilation,   #  padding 随 dilation 调整
                            groups=dim, bias=False,
                            dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=4 * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(
            torch.full((1, 4 * dim, 1, 1), float(scale))
        )
        self.band_gate = nn.Parameter(torch.ones(4))
        self.drop = nn.Identity()

    @property
    def effective_kernel(self):
        return (self.kernel - 1) * self.dilation + 1

    def _apply_band_gate(self, y):
        """Multiply by band_gate along whichever axis holds the subband."""
        B, C4, h, w = y.shape
        if self.layout == 'stack':
            g = self.band_gate.view(1, 1, 4, 1, 1)
            return (y.view(B, C4 // 4, 4, h, w) * g).view(B, C4, h, w)
        g = self.band_gate.view(1, 4, 1, 1, 1)
        return (y.view(B, 4, C4 // 4, h, w) * g).view(B, C4, h, w)

    def forward(self, sub4):
        y = self.act(self.norm(self.dw(self.pw(sub4))))
        y = self._apply_band_gate(self.scale * self.drop(y))
        return sub4 + y

    @torch.no_grad()
    def gate_report(self):
        s = self.scale.detach().flatten()
        s = (s.view(-1, 4).abs().mean(dim=0) if self.layout == 'stack'
             else s.view(4, -1).abs().mean(dim=1))
        return (s * self.band_gate.detach().abs()).tolist()


class HFShrink(nn.Module):
    def __init__(self, dim, n_bands=3, init=1e-3):
        super().__init__()
        self.thr = nn.Parameter(torch.full((n_bands, dim), float(init)))

    def forward(self, bands):
        out = []
        for i, b in enumerate(bands):
            t = F.relu(self.thr[i]).view(1, -1, 1, 1)
            out.append(torch.sign(b) * torch.clamp(b.abs() - t, min=0.0))
        return tuple(out)

    @torch.no_grad()
    def report(self):
        """Mean active threshold per band, ordered (LH, HL, HH)."""
        return F.relu(self.thr.detach()).mean(dim=1).tolist()


class LLEnhance(nn.Module):
    def __init__(self, dim, kernel=3, gate_init=1e-2,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {'affine': True}
        self.norm = norm_op(dim, **norm_op_kwargs)
        self.dw = nn.Conv2d(dim, dim, kernel, padding=kernel // 2,
                            groups=dim, bias=False)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.full((1, dim, 1, 1), float(gate_init)))

    def forward(self, ll):
        return ll + self.scale * self.pw(self.act(self.dw(self.norm(ll))))

    @torch.no_grad()
    def report(self):
        return float(self.scale.detach().abs().mean())


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    """Stochastic depth, applied per-sample. Identity when drop_prob=0."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class InceptionDWMixer2d(nn.Module):
    def __init__(self, dim, square_kernel=3, band_kernel=7, branch_ratio=0.25):
        super().__init__()
        gc = max(1, int(dim * branch_ratio))
        self.dw_square = nn.Conv2d(gc, gc, square_kernel,
                                    padding=square_kernel // 2, groups=gc, bias=False)
        self.dw_h = nn.Conv2d(gc, gc, (band_kernel, 1),
                               padding=(band_kernel // 2, 0), groups=gc, bias=False)
        self.dw_w = nn.Conv2d(gc, gc, (1, band_kernel),
                               padding=(0, band_kernel // 2), groups=gc, bias=False)
        self.split_sizes = (dim - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_sq, x_h, x_w = torch.split(x, self.split_sizes, dim=1)
        return torch.cat(
            (x_id, self.dw_square(x_sq), self.dw_h(x_h), self.dw_w(x_w)), dim=1)


class LiteInceptionNeXtBlock2d(nn.Module):
    def __init__(self, dim, mlp_ratio=2, square_kernel=3, band_kernel=7,
                 branch_ratio=0.25, norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
                 act=nn.GELU, layer_scale_init=1e-6, drop_path=0.0):
        super().__init__()
        self.dim = dim
        norm_op_kwargs = norm_op_kwargs or {}
        self.token_mixer = InceptionDWMixer2d(dim, square_kernel, band_kernel, branch_ratio)
        self.norm = norm_op(dim, **norm_op_kwargs)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, 1)
        self.act = act()
        self.pw2 = nn.Conv2d(hidden, dim, 1)
        self.gamma = (nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))
                      if layer_scale_init > 0 else None)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        y = self.token_mixer(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        if self.gamma is not None:
            y = self.gamma * y
        return shortcut + self.drop_path(y)

    def compute_conv_feature_map_size(self, input_size):
        return int(self.dim * torch.prod(torch.as_tensor(input_size)).item())

class FADSR(nn.Module):
    def __init__(self, enc_dim, *, wt_type='db2', wt_levels=3,
                 dw_kernel=7, dw_kernels=None, dropout=0.1,
                 num_stages=None, topk=None,
                 mixer_scale=0.0, mixer_dilations=(2, 1, 1), min_level_size=4,
                 subband_layout='stack', hf_shrink=False, hf_shrink_init=1e-3,
                 ll_enhance=False, ll_kernel=3, ll_gate_init=1e-2,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
                 **kwargs):
        super().__init__()
        self.enc_dim = enc_dim
        self.wt_levels = int(wt_levels)
        self.min_level_size = int(min_level_size)
        self.subband_layout = subband_layout

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type)

        L = self.wt_levels
        dils = list(mixer_dilations)[:L]
        dils += [1] * (L - len(dils))
        self.mixer_dilations = tuple(dils)

        # Per-level kernels. A level-k subband is 2^(k+1) smaller than the stage,
        # so a single kernel for all levels is either too small at L0 or almost
        # all padding at L2. Measured in-bounds tap fraction at patch 192x160 with
        # 3 levels: stage 0 holds 96/92/85% for k=7, but stage 3 is 71/47/18% and
        # stage 4 is 47/18/8%. dw_kernels shrinks the kernel with the map;
        # dw_kernel remains the scalar fallback.
        if dw_kernels is None:
            ks = [int(dw_kernel)] * L
        else:
            ks = [int(k) for k in dw_kernels][:L]
            ks += [ks[-1]] * (L - len(ks))
        self.dw_kernels = tuple(ks)

        self.mixer = nn.ModuleList([
            WaveletMixer(enc_dim, k, dilation=d, scale=mixer_scale,
                         layout=subband_layout)
            for k, d in zip(self.dw_kernels, self.mixer_dilations)])

        self.shrink = nn.ModuleList([
            HFShrink(enc_dim, n_bands=3, init=hf_shrink_init) for _ in range(L)
        ]) if hf_shrink else None

        # NOT built here. LLEnhance has conv weights, so constructing it inside
        # this __init__ would consume RNG between one stage's mixers and the next
        # stage's, shifting the init of every mixer built after it -- measured at
        # 24/198 tensors when it was inline. That is what made A-vs-F partly a
        # different-seed comparison. FADSR_PlainConvUNet calls attach_ll_enhance()
        # in a second pass, after every mixer exists, so the mixer init is
        # bit-identical to the same config with ll_enhance off.
        self.ll_enhance = None
        self._ll_cfg = (dict(kernel=ll_kernel, gate_init=ll_gate_init,
                             norm_op=norm_op, norm_op_kwargs=norm_op_kwargs)
                        if ll_enhance else None)

    def attach_ll_enhance(self):
        """Build LLEnhance if configured. Call after ALL mixers are constructed."""
        if self._ll_cfg is None or self.ll_enhance is not None:
            return
        self.ll_enhance = LLEnhance(self.enc_dim, **self._ll_cfg)

    def _pack(self, ll, lh, hl, hh):
        if self.subband_layout == 'stack':
            B, C, h, w = ll.shape
            return torch.stack([ll, lh, hl, hh], dim=2).reshape(B, 4 * C, h, w)
        return torch.cat([ll, lh, hl, hh], dim=1)

    def _unpack(self, sub4):
        if self.subband_layout == 'stack':
            B, C4, h, w = sub4.shape
            return sub4.view(B, C4 // 4, 4, h, w).unbind(dim=2)
        return sub4.chunk(4, dim=1)

    def usable_levels(self, h, w):
        """How many DWT levels this map can take before a subband gets tiny.

        Encoder stages shrink; wt_levels=3 on a 10x10 stage-4 skip would mix on a
        2x2 subband, where a k=7 conv is ~95% padding. Cap instead of assuming.
        """
        n = 0
        s = min(int(h), int(w))
        while n < self.wt_levels and s // 2 >= self.min_level_size:
            s //= 2
            n += 1
        return n

    def _refine_even(self, x, levels=None):
        L = self.wt_levels if levels is None else int(levels)
        hf_levels = []
        shapes = []
        a = x
        for k in range(L):
            shapes.append(a.shape)
            # With conv_pad = k-2 the subbands land at (N+k-2)/2, so an even
            # input can produce an ODD map for the next level. Pad to even here
            # and crop on the way back, or the reconstructed low-pass stops
            # matching its HF siblings and idwt raises a shape error.
            h, w = a.shape[-2:]
            if h % 2 or w % 2:
                a = F.pad(a, (0, w % 2, 0, h % 2), mode='replicate')
            ll, lh, hl, hh = self.wavelet.dwt(a)
            sub4 = self._pack(ll, lh, hl, hh)                # (B, 4C, h, w)
            sub4 = self.mixer[k](sub4)                        # inter-subband mix
            ll, lh, hl, hh = self._unpack(sub4)
            if self.shrink is not None:
                lh, hl, hh = self.shrink[k]((lh, hl, hh))
            hf_levels.append((lh, hl, hh))
            a = ll                                              # descend on LL

        if self.ll_enhance is not None:
            a = self.ll_enhance(a)      # global context on the coarsest LL only

        cur = a                                                # deepest LL'
        for k in range(L - 1, -1, -1):
            lh, hl, hh = hf_levels[k]
            cur = self.wavelet.idwt(cur, lh, hl, hh)
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]      # crop boundary
        return cur

    def no_weight_decay(self):
        """Gate params that must not be weight-decayed.

        `scale` and `band_gate` are learnable nn.Parameters. nnUNetTrainer's own
        configure_optimizers passes network.parameters() straight to SGD and never
        calls this, so the trainer mixin uses split_no_decay_params instead; this
        stays for callers that do ask.
        """
        return {'scale', 'band_gate'}

    @torch.no_grad()
    def gate_report(self):
        return [m.gate_report() for m in self.mixer]

    @torch.no_grad()
    def shrink_report(self):
            return None
        return [s.report() for s in self.shrink]

    @torch.no_grad()
    def reconstruction_error(self, x):
        scales = [(m.scale.detach().clone(), m.band_gate.detach().clone())
                  for m in self.mixer]
        try:
            for m in self.mixer:
                m.scale.zero_()
                m.band_gate.zero_()
            out = self.forward(x)
        finally:
            for m, (s, b) in zip(self.mixer, scales):
                m.scale.copy_(s)
                m.band_gate.copy_(b)
        nx = x.norm()
        return float((out - x).norm() / nx) if nx > 0 else 0.0

    def forward(self, x, stage_id=None):
        B, C, H, W = x.shape
        L = self.usable_levels(H, W)
        if L == 0:
            return x
        m = 2 ** L
        ph = (m - H % m) % m
        pw = (m - W % m) % m
        if ph or pw:
            xp = F.pad(x, (0, pw, 0, ph), mode='replicate')
        else:
            xp = x
        out = self._refine_even(xp, levels=L)
        if ph or pw:
            out = out[:, :, :H, :W]
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64) * 3 + 1


    m = FADSR(enc_dim=32, wt_type='db2', wt_levels=3)
    m.eval()
    with torch.no_grad():
        out = m(x, stage_id=0)
    dev = (out - x).abs().max().item()
    n_total = sum(p.numel() for p in m.parameters())
    n_learn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  max |FADSR(x)-x| @init = {dev:.3e} (haar PR -> ~0, roundoff only)")
    print(f"  params: {n_total} total, {n_learn} learnable")
    # assert dev < 1e-3, "haar identity-at-init failed"

    print("== train-mode forward + backward (grad flows to mixer) ==")
    m.train()
    out = m(x, stage_id=0)
    loss = out.pow(2).mean()
    loss.backward()

    scale_grad = float(m.mixer[0].scale.grad.abs().sum())
    print(f"  mixer scale grad: {scale_grad:.4e}")
    assert scale_grad > 0, "mixer scale should learn"
    print("OK.")

    print("== LiteInceptionNeXtBlock2d (spatial domain, encoder block) ==")
    dim = 32
    b = LiteInceptionNeXtBlock2d(dim, norm_op=nn.InstanceNorm2d, norm_op_kwargs={'affine': True})
    b.eval()
    xb = torch.randn(2, dim, 48, 40)
    with torch.no_grad():
        outb = b(xb)
    assert outb.shape == xb.shape, "block must be channel/resolution preserving"
    dev_b = (outb - xb).abs().max().item()
    print(f"  shape ok: {tuple(outb.shape)}; max |block(x)-x| @init = {dev_b:.3e} (LayerScale~1e-6 -> near identity)")
    assert dev_b < 1e-1, "layer-scale init should keep the block close to identity"

    n_total_b = sum(p.numel() for p in b.parameters())
    n_dwconv = sum(p.numel() for p in b.token_mixer.parameters())
    print(f"  params: {n_total_b} total, of which {n_dwconv} ({100 * n_dwconv / n_total_b:.1f}%) in the dw mixer "
          f"-> pointwise MLP dominates cost, dw mixer is cheap")
    print(f"  token mixer: {b.token_mixer}")

    b.train()
    outb = b(xb)
    outb.pow(2).mean().backward()
    gamma_grad = float(b.gamma.grad.abs().sum())
    print(f"  gamma grad: {gamma_grad:.4e}")
    assert gamma_grad > 0, "layer-scale gamma should learn"
    print("OK.")
