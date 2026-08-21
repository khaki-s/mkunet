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


class WaveletMixer(nn.Module):
    def __init__(self, dim, kernel=5,dilation = 1):
        super().__init__()
        self.pw = nn.Conv2d(4 * dim, 4 * dim, 1, bias=False)
        self.dw = nn.Conv2d(4 * dim, 4 * dim, kernel,
                            padding=kernel // 2 * dilation,   #  padding 随 dilation 调整
                            groups=dim, bias=False,
                            dilation=dilation) 
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=4 * dim)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.zeros(1, 4 * dim, 1, 1))  # 0 -> identity
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
    def __init__(self, enc_dim, *, wt_type='db2', wt_levels=3,
                 dw_kernel=7, dropout=0.1, num_stages=None, topk=None, **kwargs):
        super().__init__()
        self.enc_dim = enc_dim
        self.wt_levels = int(wt_levels)

        self.wavelet = WaveletOps(enc_dim, wt_type=wt_type)

        L = self.wt_levels
        self.mixer = nn.ModuleList([
            WaveletMixer(enc_dim, dw_kernel, dilation=1) for _ in range(L - 1)] +
            [WaveletMixer(enc_dim, dw_kernel, dilation=2)])

    def _refine_even(self, x):
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
            cur = self.wavelet.idwt(cur, lh, hl, hh)
            cur = cur[:, :, :shapes[k][2], :shapes[k][3]]      # crop boundary
        return cur

    def no_weight_decay(self):
        return {n for n, _ in self.named_parameters() if 'scale' in n}

    def forward(self, x, stage_id=None):
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
