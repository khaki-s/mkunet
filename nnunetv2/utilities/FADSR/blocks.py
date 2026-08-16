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
    """DWT/IDWT as strided grouped conv.

    conv_pad must be k-2 to invert. The old k//2-1 did NOT: for db2 (k=4) one
    level round-trips with 6.25% error, and the 3-level cascade left 6.3%
    (192x160) to 20.5% (12x10) of the skip magnitude behind as a fixed,
    non-learned perturbation. Haar (k=2) inverts either way, which is why this
    stayed hidden until wt_type moved to db2.

    Subbands land at (N+k-2)/2, so an even input can hand an ODD map to the
    next level -- the cascade pads to even per level and crops on the way back.
    """

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
    """Refine one DWT level in place: sub4 -> sub4 + scale * branch(sub4).

    `scale_init` matters more than it looks. At scale=0 the branch output is
    multiplied by 0, so dL/d(branch weights) = dL/dy * scale * (...) = 0 --
    only `scale` itself gets gradient at step 0, and it has to climb out of 0
    driven by a random branch. Over 100 epochs it reaches ~0.009 (measured on
    the trained checkpoints), i.e. the refinement contributes <1% of the skip
    and was never really tested. WTConv, which this is derived from, starts its
    wavelet path at 0.1 instead (see FADSR/wtconv2d.py:30).
    """

    def __init__(self, dim, kernel=5, dilation=1, scale_init=0.0):
        super().__init__()
        self.pw = nn.Conv2d(4 * dim, 4 * dim, 1, bias=False)
        self.dw = nn.Conv2d(4 * dim, 4 * dim, kernel,
                            padding=kernel // 2 * dilation,   #  padding 随 dilation 调整
                            groups=dim, bias=False,
                            dilation=dilation)
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=4 * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.full((1, 4 * dim, 1, 1), float(scale_init)))
        self.drop = nn.Identity()

    def forward(self, sub4):
        return sub4 + self.scale * self.drop(self.act(self.norm(self.dw(self.pw(sub4)))))


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


# class InceptionDWMixer2d(nn.Module):
#     def __init__(self, dim, square_kernel=3, band_kernel=7, branch_ratio=0.25):
#         super().__init__()
#         gc = max(1, int(dim * branch_ratio))
#         self.dw_square = nn.Conv2d(gc, gc, square_kernel,
#                                     padding=square_kernel // 2, groups=gc, bias=False)
#         self.dw_h = nn.Conv2d(gc, gc, (band_kernel, 1),
#                                padding=(band_kernel // 2, 0), groups=gc, bias=False)
#         self.dw_w = nn.Conv2d(gc, gc, (1, band_kernel),
#                                padding=(0, band_kernel // 2), groups=gc, bias=False)
#         self.split_sizes = (dim - 3 * gc, gc, gc, gc)

#     def forward(self, x):
#         x_id, x_sq, x_h, x_w = torch.split(x, self.split_sizes, dim=1)
#         return torch.cat(
#             (x_id, self.dw_square(x_sq), self.dw_h(x_h), self.dw_w(x_w)), dim=1)


# class LiteInceptionNeXtBlock2d(nn.Module):
#     def __init__(self, dim, mlp_ratio=2, square_kernel=3, band_kernel=7,
#                  branch_ratio=0.25, norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
#                  act=nn.GELU, layer_scale_init=1e-6, drop_path=0.0):
#         super().__init__()
#         self.dim = dim
#         norm_op_kwargs = norm_op_kwargs or {}
#         self.token_mixer = InceptionDWMixer2d(dim, square_kernel, band_kernel, branch_ratio)
#         self.norm = norm_op(dim, **norm_op_kwargs)
#         hidden = int(dim * mlp_ratio)
#         self.pw1 = nn.Conv2d(dim, hidden, 1)
#         self.act = act()
#         self.pw2 = nn.Conv2d(hidden, dim, 1)
#         self.gamma = (nn.Parameter(layer_scale_init * torch.ones(1, dim, 1, 1))
#                       if layer_scale_init > 0 else None)
#         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

#     def forward(self, x):
#         shortcut = x
#         y = self.token_mixer(x)
#         y = self.norm(y)
#         y = self.pw2(self.act(self.pw1(y)))
#         if self.gamma is not None:
#             y = self.gamma * y
#         return shortcut + self.drop_path(y)

#     def compute_conv_feature_map_size(self, input_size):
#         return int(self.dim * torch.prod(torch.as_tensor(input_size)).item())


class ASPPModule(nn.Module):
    def __init__(self, dim, dilations=(1, 2, 3), norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
                 act=nn.GELU):
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
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)

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
    """Frequency-domain skip refinement: cascaded DWT -> per-level mix -> IDWT.

    `use_base_branch` restores the parallel full-resolution depthwise conv that
    WTConv has and this dropped (FADSR/wtconv2d.py:23-31, `base_conv` +
    `base_scale` at init_scale=1.0). Two reasons it matters:

    - RepLKNet: a large depthwise kernel wants a parallel small-kernel path or
      it optimizes poorly. The 7x7 subband conv here is exactly such a kernel
      with no bypass.
    - It is ungated by construction. `base_conv` starts at ZERO weight with
      base_scale=1.0, so the module is still identity at init, but
      dL/d(base_conv.weight) = dL/dy * base_scale * x is at FULL strength from
      step 0 -- no LayerScale warm-up. Contrast the wavelet path, where scale=0
      zeroes the gradient to its own weights (see WaveletMixer docstring).

    The cascade depth is capped by resolution, see `effective_levels`.
    """

    def __init__(self, enc_dim, *, wt_type='db2', wt_levels=3,
                 dw_kernel=7, dropout=0.1, num_stages=None, topk=None,
                 use_base_branch=False, base_kernel=5,
                 mixer_scale_init=0.0, min_ll_size=4,
                 **kwargs):
        super().__init__()
        self.enc_dim = enc_dim
        self.wt_levels = int(wt_levels)
        self.min_ll_size = int(min_ll_size)

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type)

        L = self.wt_levels
        self.mixer = nn.ModuleList([
            WaveletMixer(enc_dim, dw_kernel, dilation=1, scale_init=mixer_scale_init)
            for _ in range(L - 1)] +
            [WaveletMixer(enc_dim, dw_kernel, dilation=2, scale_init=mixer_scale_init)])

        if use_base_branch:
            self.base_conv = nn.Conv2d(enc_dim, enc_dim, base_kernel,
                                       padding=base_kernel // 2, groups=enc_dim,
                                       bias=False)
            nn.init.zeros_(self.base_conv.weight)   # identity at init, full grad
            self.base_scale = nn.Parameter(torch.ones(1, enc_dim, 1, 1))
        else:
            self.base_conv = None
            self.base_scale = None

    @staticmethod
    def levels_for(H, W, max_levels=3, min_ll_size=4):
        """Levels that keep the coarsest LL >= min_ll_size px per side.

        wt_levels=3 on a 12x10 skip gave LL grids 8x8 -> 4x4 -> 2x2, and a
        7-tap depthwise on 2x2 is almost entirely padding.
        """
        L = max_levels
        while L > 1 and min(H, W) // (2 ** L) < min_ll_size:
            L -= 1
        return L

    def effective_levels(self, H, W):
        """Never exceed the mixers we built (inference patch may differ)."""
        return min(self.wt_levels,
                   self.levels_for(H, W, self.wt_levels, self.min_ll_size))

    def _refine_even(self, x, L):
        hf_levels = []
        shapes = []
        a = x
        for k in range(L):
            shapes.append(a.shape)
            # Subbands land at (N+k-2)/2, so an even input can hand an ODD map
            # to the next level; pad to even here and crop on the way back,
            # otherwise the reconstructed low-pass stops matching its HF
            # siblings and idwt raises a shape error.
            h, w = a.shape[-2:]
            if h % 2 or w % 2:
                a = F.pad(a, (0, w % 2, 0, h % 2), mode='replicate')
            ll, lh, hl, hh = self.wavelet.dwt(a)
            sub4 = torch.cat([ll, lh, hl, hh], dim=1)          # (B, 4C, h, w)
            sub4 = self.mixer[k](sub4)                         # inter-subband mix
            ll, lh, hl, hh = sub4.chunk(4, dim=1)
            hf_levels.append((lh, hl, hh))
            a = ll                                              # descend on LL

        cur = a                                                # deepest LL'
        for k in range(L - 1, -1, -1):
            lh, hl, hh = hf_levels[k]
            cur = self.wavelet.idwt(cur, lh, hl, hh)
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]      # undo pad + boundary
        return cur

    def no_weight_decay(self):
        return {n for n, _ in self.named_parameters()
                if 'scale' in n or n.endswith('gamma')}

    def forward(self, x, stage_id=None):
        B, C, H, W = x.shape
        L = self.effective_levels(H, W)
        m = 2 ** L
        ph = (m - H % m) % m
        pw = (m - W % m) % m
        if ph or pw:
            xp = F.pad(x, (0, pw, 0, ph), mode='replicate')
        else:
            xp = x
        out = self._refine_even(xp, L)
        if ph or pw:
            out = out[:, :, :H, :W]
        if self.base_conv is not None:
            out = out + self.base_scale * self.base_conv(x)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, 64) * 3 + 1

    print("== FADSR (frequency domain, skip refinement): Haar (db1) identity at init ==")
    m = FADSR(enc_dim=32, wt_type='db1', wt_levels=3)
    m.eval()
    with torch.no_grad():
        out = m(x, stage_id=0)
    dev = (out - x).abs().max().item()
    n_total = sum(p.numel() for p in m.parameters())
    n_learn = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  max |FADSR(x)-x| @init = {dev:.3e} (haar PR -> ~0, roundoff only)")
    print(f"  params: {n_total} total, {n_learn} learnable")
    assert dev < 1e-3, "haar identity-at-init failed"

    print("== train-mode forward + backward (grad flows to mixer) ==")
    m.train()
    out = m(x, stage_id=0)
    loss = out.pow(2).mean()
    loss.backward()

    scale_grad = float(m.mixer[0].scale.grad.abs().sum())
    print(f"  mixer scale grad: {scale_grad:.4e}")
    assert scale_grad > 0, "mixer scale should learn"
    print("OK.")

    print("== db2 now inverts: identity at init, all skip resolutions ==")
    for H, W in [(192, 160), (96, 80), (48, 40), (24, 20), (12, 10), (25, 19)]:
        md = FADSR(enc_dim=32).eval()
        xs = torch.randn(2, 32, H, W) * 3 + 1
        with torch.no_grad():
            o = md(xs, stage_id=0)
        dev = (o - xs).abs().max().item()
        assert o.shape == xs.shape, f"{(H, W)} -> {tuple(o.shape)}"
        assert dev < 1e-3, f"db2 must be identity at init, got {dev}"
        print(f"  {H:3d}x{W:3d}: L={md.effective_levels(H, W)}  max|f(x)-x|={dev:.2e}")

    print("== base branch: identity at init, full-strength grad at step 0 ==")
    mb = FADSR(enc_dim=32, use_base_branch=True, mixer_scale_init=0.0).train()
    mb(torch.randn(2, 32, 48, 40), stage_id=0).pow(2).mean().backward()
    g_base = float(mb.base_conv.weight.grad.abs().sum())
    g_dw = float(mb.mixer[0].dw.weight.grad.abs().sum())
    print(f"  base_conv.weight grad = {g_base:.3e}  (ungated)")
    print(f"  mixer.dw.weight  grad = {g_dw:.3e}  (gated by scale=0 -> exactly 0)")
    assert g_base > 0, "base_conv must get gradient at step 0"
    assert g_dw == 0.0, "documents the scale=0 gradient block"
    print("OK.")
