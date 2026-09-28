import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def _separable_bank(lo, hi, ndim):
    bank = []
    for idx in range(2 ** ndim):
        # bit j (from the most significant) selects the filter for axis j
        t = None
        for j in range(ndim):
            f = hi if (idx >> (ndim - 1 - j)) & 1 else lo
            # place f along axis j, singleton elsewhere
            shape = [1] * ndim
            shape[j] = f.numel()
            fj = f.view(shape)
            t = fj if t is None else t * fj
        bank.append(t)
    return torch.stack(bank, dim=0)


def _make_wavelet_filters(wt_type, channels, ndim=2, dtype=torch.float32):
    w = pywt.Wavelet(wt_type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype)

    dec = _separable_bank(dec_lo, dec_hi, ndim)      # (2**ndim, k, ...)
    rec = _separable_bank(rec_lo, rec_hi, ndim)
    # (S*C, 1, k, ...) for grouped conv with groups=C
    dec = dec[:, None].repeat(channels, *([1] * (ndim + 1)))
    rec = rec[:, None].repeat(channels, *([1] * (ndim + 1)))
    return dec, rec


class WaveletOps(nn.Module):
    def __init__(self, dim, wt_type='db1', ndim=2):
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {ndim}')
        self.dim = dim
        self.ndim = int(ndim)
        self.n_sub = 2 ** self.ndim
        dec, rec = _make_wavelet_filters(wt_type, dim, ndim=self.ndim)
        self.register_buffer('wt_filter', dec)
        self.register_buffer('iwt_filter', rec)
        self.k = dec.shape[-1]
        self.conv_pad = self.k - 2
        self._conv = F.conv2d if self.ndim == 2 else F.conv3d
        self._convT = F.conv_transpose2d if self.ndim == 2 else F.conv_transpose3d

    def dwt(self, x):
        """(B,C,*spatial) -> tuple of n_sub tensors, index 0 = approximation."""
        B, C = x.shape[:2]
        y = self._conv(x, self.wt_filter, stride=2, groups=C,
                       padding=self.conv_pad)
        spatial = y.shape[2:]
        y = y.view(B, C, self.n_sub, *spatial)
        return y.unbind(dim=2)

    def idwt(self, bands):
        """Inverse of dwt. `bands` is a sequence of n_sub tensors."""
        assert len(bands) == self.n_sub, \
            f'expected {self.n_sub} subbands, got {len(bands)}'
        B, C = bands[0].shape[:2]
        spatial = bands[0].shape[2:]
        sub = torch.stack(bands, dim=2).reshape(B, C * self.n_sub, *spatial)
        return self._convT(sub, self.iwt_filter, stride=2, groups=C,
                           padding=self.conv_pad)


class WaveletMixer(nn.Module):
    def __init__(self, dim, kernel=5, dilation=1, scale=0.0, layout='stack',
                 ndim=2):
        super().__init__()
        if layout not in ('stack', 'cat'):
            raise ValueError(f"layout must be 'stack' or 'cat', got {layout!r}")
        if ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {ndim}')
        self.dim = dim
        self.kernel = int(kernel)
        self.dilation = int(dilation)
        self.layout = layout
        self.ndim = int(ndim)
        self.n_sub = 2 ** self.ndim
        S = self.n_sub
        conv = nn.Conv2d if self.ndim == 2 else nn.Conv3d
        self.pw = conv(S * dim, S * dim, 1, bias=False)
        self.dw = conv(S * dim, S * dim, kernel,
                       padding=kernel // 2 * dilation,   #  padding 随 dilation 调整
                       groups=dim, bias=False,
                       dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=S * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(
            torch.full((1, S * dim, *([1] * self.ndim)), float(scale))
        )
        self.band_gate = nn.Parameter(torch.ones(S))
        self.drop = nn.Identity()

    @property
    def effective_kernel(self):
        return (self.kernel - 1) * self.dilation + 1

    def _apply_band_gate(self, y):
        """Multiply by band_gate along whichever axis holds the subband."""
        B, CS = y.shape[:2]
        spatial = y.shape[2:]
        S = self.n_sub
        ones = [1] * self.ndim
        if self.layout == 'stack':
            g = self.band_gate.view(1, 1, S, *ones)
            return (y.view(B, CS // S, S, *spatial) * g).view(B, CS, *spatial)
        g = self.band_gate.view(1, S, 1, *ones)
        return (y.view(B, S, CS // S, *spatial) * g).view(B, CS, *spatial)

    def forward(self, sub):
        y = self.act(self.norm(self.dw(self.pw(sub))))
        y = self._apply_band_gate(self.scale * self.drop(y))
        return sub + y

    @torch.no_grad()
    def gate_report(self):
        """Effective per-subband gain, ordered as in _separable_bank."""
        s = self.scale.detach().flatten()
        S = self.n_sub
        s = (s.view(-1, S).abs().mean(dim=0) if self.layout == 'stack'
             else s.view(S, -1).abs().mean(dim=1))
        return (s * self.band_gate.detach().abs()).tolist()


class HFShrink(nn.Module):
    def __init__(self, dim, n_bands=3, init=1e-3, ndim=2):
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {ndim}')
        self.ndim = int(ndim)
        self.thr = nn.Parameter(torch.full((n_bands, dim), float(init)))

    def forward(self, bands):
        assert len(bands) == self.thr.shape[0], \
            f'expected {self.thr.shape[0]} HF bands, got {len(bands)}'
        out = []
        ones = [1] * self.ndim
        for i, b in enumerate(bands):
            t = F.relu(self.thr[i]).view(1, -1, *ones)
            out.append(torch.sign(b) * torch.clamp(b.abs() - t, min=0.0))
        return tuple(out)

    @torch.no_grad()
    def report(self):
        """Mean active threshold per band, ordered (LH, HL, HH)."""
        return F.relu(self.thr.detach()).mean(dim=1).tolist()


class LLEnhance(nn.Module):
    def __init__(self, dim, kernel=3, gate_init=1e-2,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None, ndim=2):
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {ndim}')
        self.ndim = int(ndim)
        norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {'affine': True}
        if ndim == 3 and norm_op is nn.InstanceNorm2d:
            norm_op = nn.InstanceNorm3d      # caller left the 2D default in place
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        self.norm = norm_op(dim, **norm_op_kwargs)
        self.dw = conv(dim, dim, kernel, padding=kernel // 2,
                       groups=dim, bias=False)
        self.pw = conv(dim, dim, 1, bias=False)
        self.act = nn.GELU()
        self.scale = nn.Parameter(
            torch.full((1, dim, *([1] * self.ndim)), float(gate_init)))

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

class ASPPModule(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 3), norm_op=nn.InstanceNorm2d,
                 norm_op_kwargs=None, act=nn.GELU):
        super().__init__()
        norm_op_kwargs = norm_op_kwargs or {}
        self.branch_1x1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.branches = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, groups=dim, bias=False)
            for d in dilations
        ])
        self.pool_proj = nn.Conv2d(dim, dim, 1, bias=False)
        n_branches = 2 + len(dilations)
        self.fuse = nn.Conv2d(n_branches * dim, dim, 1, bias=False)
        self.norm = norm_op(dim, **norm_op_kwargs)
        self.act = act()

    def forward(self, x):
        pooled = self.pool_proj(x.mean(dim=(2, 3), keepdim=True)).expand_as(x)
        feats = [self.branch_1x1(x)] + [b(x) for b in self.branches] + [pooled]
        y = self.fuse(torch.cat(feats, dim=1))
        return self.act(self.norm(y))


class CBAM(nn.Module):
    def __init__(self, dim, reduction=16, spatial_kernel=7):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(avg_out + max_out)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.spatial(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class BottleneckASPPCBAM(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 3), reduction=16, spatial_kernel=7,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None, act=nn.GELU,
                 layer_scale_init=1e-6):
        super().__init__()
        self.dim = dim
        self.aspp = ASPPModule(dim, dilations=dilations, norm_op=norm_op,
                               norm_op_kwargs=norm_op_kwargs, act=act)
        self.cbam = CBAM(dim, reduction=reduction, spatial_kernel=spatial_kernel)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        y = self.cbam(self.aspp(x))
        return x + self.gamma * y

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
                 ndim=2,
                 **kwargs):
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f'ndim must be 2 or 3, got {ndim}')
        self.enc_dim = enc_dim
        self.wt_levels = int(wt_levels)
        self.min_level_size = int(min_level_size)
        self.subband_layout = subband_layout
        self.ndim = int(ndim)
        self.n_sub = 2 ** self.ndim

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type, ndim=self.ndim)

        L = self.wt_levels
        dils = list(mixer_dilations)[:L]
        dils += [1] * (L - len(dils))
        self.mixer_dilations = tuple(dils)
        if dw_kernels is None:
            ks = [int(dw_kernel)] * L
        else:
            ks = [int(k) for k in dw_kernels][:L]
            ks += [ks[-1]] * (L - len(ks))
        self.dw_kernels = tuple(ks)

        self.mixer = nn.ModuleList([
            WaveletMixer(enc_dim, k, dilation=d, scale=mixer_scale,
                         layout=subband_layout, ndim=self.ndim)
            for k, d in zip(self.dw_kernels, self.mixer_dilations)])

        # n_sub - 1 HF bands: 3 in 2D, 7 in 3D
        self.shrink = nn.ModuleList([
            HFShrink(enc_dim, n_bands=self.n_sub - 1, init=hf_shrink_init,
                     ndim=self.ndim)
            for _ in range(L)
        ]) if hf_shrink else None
        self.ll_enhance = None
        self._ll_cfg = (dict(kernel=ll_kernel, gate_init=ll_gate_init,
                             norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                             ndim=self.ndim)
                        if ll_enhance else None)

    def attach_ll_enhance(self):
        """Build LLEnhance if configured. Call after ALL mixers are constructed."""
        if self._ll_cfg is None or self.ll_enhance is not None:
            return
        self.ll_enhance = LLEnhance(self.enc_dim, **self._ll_cfg)

    def _pack(self, bands):
        """Sequence of n_sub tensors -> one (B, n_sub*C, *spatial) tensor."""
        S = self.n_sub
        if self.subband_layout == 'stack':
            B, C = bands[0].shape[:2]
            spatial = bands[0].shape[2:]
            return torch.stack(bands, dim=2).reshape(B, S * C, *spatial)
        return torch.cat(list(bands), dim=1)

    def _unpack(self, sub):
        """Inverse of _pack. Returns a tuple of n_sub tensors."""
        S = self.n_sub
        if self.subband_layout == 'stack':
            B, CS = sub.shape[:2]
            spatial = sub.shape[2:]
            return sub.view(B, CS // S, S, *spatial).unbind(dim=2)
        return sub.chunk(S, dim=1)

    def usable_levels(self, *spatial):
        """Driven by the SMALLEST axis, which in 3D is normally z."""
        n = 0
        s = min(int(v) for v in spatial)
        while n < self.wt_levels and s // 2 >= self.min_level_size:
            s //= 2
            n += 1
        return n

    def _pad_to_even(self, a):
        """Pad trailing odd axes by 1. F.pad wants axes last-first."""
        spatial = a.shape[2:]
        if not any(s % 2 for s in spatial):
            return a
        pad = []
        for s in reversed(spatial):
            pad += [0, s % 2]
        return F.pad(a, pad, mode='replicate')

    def _refine_even(self, x, levels=None):
        L = self.wt_levels if levels is None else int(levels)
        hf_levels = []
        shapes = []
        a = x
        for k in range(L):
            shapes.append(a.shape)
            # With conv_pad = k-2 the subbands land at (N+k-2)/2, so an even input
            # can produce an ODD map for the next level. Pad to even here and crop
            # on the way back, or idwt raises a shape error.
            a = self._pad_to_even(a)
            bands = self.wavelet.dwt(a)             # index 0 = approximation
            sub = self._pack(bands)                 # (B, n_sub*C, *spatial)
            sub = self.mixer[k](sub)                # inter-subband mix
            bands = self._unpack(sub)
            approx, hf = bands[0], list(bands[1:])
            if self.shrink is not None:
                hf = list(self.shrink[k](tuple(hf)))
            hf_levels.append(hf)
            a = approx                              # descend on the approximation

        if self.ll_enhance is not None:
            a = self.ll_enhance(a)      # global context on the coarsest LL only

        cur = a                                     # deepest approximation
        for k in range(L - 1, -1, -1):
            cur = self.wavelet.idwt([cur] + hf_levels[k])
            sl = [slice(None), slice(None)]
            sl += [slice(0, s) for s in shapes[k][2:]]
            cur = cur[tuple(sl)]                    # crop boundary
        return cur

    def no_weight_decay(self):
        return {'scale', 'band_gate'}

    @torch.no_grad()
    def gate_report(self):
        return [m.gate_report() for m in self.mixer]

    @torch.no_grad()
    def shrink_report(self):
        if self.shrink is None:
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
        spatial = x.shape[2:]
        assert len(spatial) == self.ndim, \
            f'FADSR built for ndim={self.ndim} but got input {tuple(x.shape)}'
        L = self.usable_levels(*spatial)
        if L == 0:
            return x
        m = 2 ** L
        pads = [(m - s % m) % m for s in spatial]
        if any(pads):
            flat = []
            for p in reversed(pads):
                flat += [0, p]
            xp = F.pad(x, flat, mode='replicate')
        else:
            xp = x
        out = self._refine_even(xp, levels=L)
        if any(pads):
            sl = [slice(None), slice(None)] + [slice(0, s) for s in spatial]
            out = out[tuple(sl)]
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
