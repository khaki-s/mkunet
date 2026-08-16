"""Quantify what the db2 conv_pad bug actually does to the skip, at init.

No pywt in this env -> hardcode the db1/db2 filter banks exactly as
pywt.Wavelet(...).dec_lo/dec_hi/rec_lo/rec_hi so we faithfully reproduce
blocks._make_wavelet_filters.
"""
import torch
import torch.nn.functional as F

S2 = 2 ** 0.5
BANKS = {
    'db1': dict(
        dec_lo=[0.7071067811865476, 0.7071067811865476],
        dec_hi=[-0.7071067811865476, 0.7071067811865476],
        rec_lo=[0.7071067811865476, 0.7071067811865476],
        rec_hi=[0.7071067811865476, -0.7071067811865476],
    ),
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
    t = lambda k: torch.tensor(b[k][::-1] if k.startswith('dec') else b[k],
                               dtype=torch.float32)
    dec_lo, dec_hi = t('dec_lo'), t('dec_hi')
    rec_lo, rec_hi = t('rec_lo'), t('rec_hi')
    outer = lambda a, c: a[:, None] * c[None, :]
    dec = torch.stack([outer(dec_lo, dec_lo), outer(dec_lo, dec_hi),
                       outer(dec_hi, dec_lo), outer(dec_hi, dec_hi)], 0)
    rec = torch.stack([outer(rec_lo, rec_lo), outer(rec_lo, rec_hi),
                       outer(rec_hi, rec_lo), outer(rec_hi, rec_hi)], 0)
    return dec[:, None].repeat(channels, 1, 1, 1), rec[:, None].repeat(channels, 1, 1, 1)


class Ops:
    def __init__(self, wt, dim, pad):
        self.dec, self.rec = make_filters(wt, dim)
        self.pad = pad

    def dwt(self, x):
        C = x.shape[1]
        y = F.conv2d(x, self.dec, stride=2, groups=C, padding=self.pad)
        B, _, h, w = y.shape
        return y.view(B, C, 4, h, w).unbind(2)

    def idwt(self, ll, lh, hl, hh):
        B, C, h, w = ll.shape
        sub = torch.stack([ll, lh, hl, hh], 2).reshape(B, C * 4, h, w)
        return F.conv_transpose2d(sub, self.rec, stride=2, groups=C, padding=self.pad)


def cascade(x, ops, levels=3, pad_even=False):
    """FADSR._refine_even with scale=0 (mixer is exactly identity)."""
    shapes, hf = [], []
    a = x
    for _ in range(levels):
        shapes.append(a.shape)
        if pad_even:
            h, w = a.shape[-2:]
            if h % 2 or w % 2:
                a = F.pad(a, (0, w % 2, 0, h % 2), mode='replicate')
        ll, lh, hl, hh = ops.dwt(a)
        hf.append((lh, hl, hh))
        a = ll
    cur = a
    for k in range(levels - 1, -1, -1):
        cur = ops.idwt(cur, *hf[k])
        cur = cur[:, :, :shapes[k][2], :shapes[k][3]]
    return cur


def fadsr_forward(x, ops, levels=3, pad_even=False):
    """FADSR.forward: pad to multiple of 2**levels, cascade, crop back."""
    H, W = x.shape[-2:]
    m = 2 ** levels
    ph, pw = (m - H % m) % m, (m - W % m) % m
    xp = F.pad(x, (0, pw, 0, ph), mode='replicate') if (ph or pw) else x
    out = cascade(xp, ops, levels, pad_even)
    return out[:, :, :H, :W]


def rel(a, b):
    return (a - b).norm().item() / b.norm().item()


torch.manual_seed(0)
print("=== single-level DWT->IDWT roundtrip, which conv_pad inverts ===")
x = torch.randn(1, 8, 64, 64)
for wt in ('db1', 'db2'):
    k = len(BANKS[wt]['dec_lo'])
    print(f"  {wt} (k={k}):  legacy k//2-1={k//2-1}   fixed k-2={k-2}")
    for pad in range(0, k + 1):
        o = Ops(wt, 8, pad)
        try:
            rt = o.idwt(*o.dwt(x))
        except RuntimeError:
            continue
        if rt.shape[-2:] != x.shape[-2:]:
            print(f"     pad={pad}: shape {tuple(rt.shape[-2:])} != input, skip")
            continue
        print(f"     pad={pad}: rel err = {100*rel(rt, x):8.4f}%"
              f"   {'<-- inverts' if rel(rt, x) < 1e-5 else ''}")

print()
print("=== 3-level cascade at REAL nnU-Net 2d skip resolutions (scale=0) ===")
print("    i.e. what FADSR does to the skip at init, before any learning")
sizes = [(192, 160, 32), (96, 80, 64), (48, 40, 128), (24, 20, 256), (12, 10, 512)]
CASES = (
    ('db2', 1, False, 'CURRENT (buggy pad, no even-pad)'),
    ('db2', 2, False, 'pad fixed ONLY (no even-pad) -> expect shape crash'),
    ('db2', 2, True,  'pad fixed + even-pad  [full fix]'),
    ('db1', 0, False, 'haar baseline'),
)
for wt, pad, pe, tag in CASES:
    print(f"  -- {wt} pad={pad} even_pad={pe}  [{tag}]")
    for H, W, C in sizes:
        xs = torch.randn(2, C, H, W)
        o = Ops(wt, C, pad)
        try:
            out = fadsr_forward(xs, o, 3, pe)
        except RuntimeError as e:
            print(f"     {H:3d}x{W:3d} C={C:3d}: RuntimeError: {str(e)[:70]}")
            continue
        ok = out.shape == xs.shape
        r = rel(out, xs) if ok else float('nan')
        print(f"     {H:3d}x{W:3d} C={C:3d}: shape={'ok' if ok else tuple(out.shape)}"
              f"  ||f(x)-x||/||x|| = {100*r:7.3f}%")
