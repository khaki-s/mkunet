"""Mathematically explicit 3D blocks; see documentation/brats2021_four_directions.md.

Only SegMambaV2Residual3D reuses a paper's implementation verbatim (vendored
MambaLayer). Other blocks are new, deliberately small hypotheses, not paper
reproductions. Channels: T1/T1ce/T2/FLAIR; regions: WT/TC/ET.
"""
import torch
from torch import nn
from torch.nn import functional as F


def raw_to_internal(labels):
    """Raw 0/1/2/4 -> nnU-Net 0/2/1/3, without in-place collisions."""
    if not torch.all((labels == 0) | (labels == 1) | (labels == 2) | (labels == 4)):
        raise ValueError('Expected raw BraTS labels 0,1,2,4')
    out = torch.zeros_like(labels)
    for source, dest in ((1, 2), (2, 1), (4, 3)):
        out[labels == source] = dest
    return out


def regions_to_exclusive(p):
    """Differentiable upward hierarchy closure, then a valid 4-way simplex.

    ET'=ET, TC'=max(TC,ET), WT'=max(WT,TC'). This is a chosen closure,
    not the Euclidean/isotonic projection. Order: background, ED, NCR, ET.
    """
    et = p[:, 2:3]
    tc = torch.maximum(p[:, 1:2], et)
    wt = torch.maximum(p[:, 0:1], tc)
    return torch.cat((1 - wt, wt - tc, tc - et, et), dim=1)


class ExclusiveHierarchyLoss(nn.Module):
    """Base region loss + 0.1 exclusive Dice/empty FP + 0.05 hierarchy.

    Empty classes use mean predicted probability instead of a near-singular
    empty Dice. Nonempty class Dice is per sample, never pooled over patients.
    """
    def __init__(self, base_loss, exclusive_weight=0.1, hierarchy_weight=0.05):
        super().__init__()
        self.base_loss = base_loss
        self.exclusive_weight = exclusive_weight
        self.hierarchy_weight = hierarchy_weight

    def forward(self, logits, target):
        if logits.shape != target.shape or logits.shape[1] != 3:
            raise ValueError('Expected matching B,3,D,H,W WT/TC/ET tensors; no ignore channel')
        p = logits.float().sigmoid()
        y = target.float()
        q = regions_to_exclusive(p)[:, 1:]
        t = torch.cat((y[:, 0:1] - y[:, 1:2], y[:, 1:2] - y[:, 2:3], y[:, 2:3]), 1)
        axes = tuple(range(2, q.ndim))
        size = t.sum(axes)
        dice_loss = 1 - (2 * (q * t).sum(axes) + 1e-5) / (q.sum(axes) + size + 1e-5)
        exclusive = torch.where(size > 0, dice_loss, q.mean(axes)).mean()
        hierarchy = (F.relu(p[:, 1] - p[:, 0]) + F.relu(p[:, 2] - p[:, 1])).mean()
        return self.base_loss(logits, target) + self.exclusive_weight * exclusive + self.hierarchy_weight * hierarchy


class BoundedModalityGate3D(nn.Module):
    """x'_m=x_m*(1+0.5*tanh(a_m(x))), exactly identity at initialization."""
    def __init__(self):
        super().__init__()
        self.router = nn.Sequential(nn.Conv3d(4, 8, 3, padding=1), nn.LeakyReLU(0.01), nn.Conv3d(8, 4, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, x):
        # Route at half resolution to limit memory on 128x160x112 patches.
        a = self.router(F.avg_pool3d(x, 2, ceil_mode=True))
        a = F.interpolate(a, size=x.shape[2:], mode='trilinear', align_corners=False)
        return x * (1 + 0.5 * a.tanh())


class ClassConditionalMoE3D(nn.Module):
    """Dense soft MoE on a half-resolution skip; no GT is used for routing.

    Four single-modality experts plus a joint T1/T1ce/T2/FLAIR expert.
    Three learned routing queries (ED/NCR/ET), all with access to all experts.
    A soft initial bias is an ablation, not a fixed MRI rule. Recombined class
    features refine a shared skip; the decoder still emits WT/TC/ET logits.
    This is feature routing inspired by SimMLM, NOT its DMoMEOutputLevel.
    """
    def __init__(self, channels, width=8, prior=True, uniform=False):
        super().__init__()
        self.uniform = uniform
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv3d(n, width, 3, padding=1), nn.GroupNorm(1, width),
                          nn.LeakyReLU(0.01), nn.Conv3d(width, width, 1))
            for n in (1, 1, 1, 1, 4)
        ])
        self.router = None if uniform else nn.Sequential(
            nn.Conv3d(4 + channels, 16, 1), nn.LeakyReLU(0.01), nn.Conv3d(16, 15, 1))
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            bias = torch.zeros(3, 5)
            if prior:
                # ED: FLAIR; NCR: joint information; ET: T1ce. exp(log 2)=2.
                bias[0, 3] = bias[1, 4] = bias[2, 1] = 0.69314718056
            with torch.no_grad():
                self.router[-1].bias.copy_(bias.flatten())
        self.project = nn.Conv3d(3 * width, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))
        self.register_buffer('route_sum', torch.zeros(3, 5), persistent=False)
        self.register_buffer('route_count', torch.zeros(()), persistent=False)

    def forward(self, feature, image):
        x = F.interpolate(image, size=feature.shape[2:], mode='trilinear', align_corners=False)
        experts = torch.stack([expert(x[:, m:m+1] if m < 4 else x)
                               for m, expert in enumerate(self.experts)], dim=1)
        if self.uniform:
            weights = x.new_full((x.shape[0], 3, 5, *x.shape[2:]), 0.2)
        else:
            logits = self.router(torch.cat((x, feature), dim=1))
            weights = logits.reshape(x.shape[0], 3, 5, *x.shape[2:]).float().softmax(2).to(experts.dtype)
        # Avoid a B x class x expert x feature x volume broadcast temporary.
        fused = torch.cat([sum(weights[:, c, e:e+1] * experts[:, e] for e in range(5))
                           for c in range(3)], dim=1)
        if self.training:
            with torch.no_grad():
                self.route_sum.add_(weights.float().mean((0, 3, 4, 5)))
                self.route_count.add_(1)
        return feature + self.scale.tanh() * self.project(fused)


class FrequencySkip3D(nn.Module):
    """F'=F+tanh(alpha)*sigmoid(g(F))*(F-L(F)); fixed local box low-pass.

    Replicate padding preserves a constant signal at edges. This is a feature
    frequency proxy, not MRI k-space, PSF estimation, or global image sharpening.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv3d(channels, channels, 1, groups=channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))

    def forward(self, x):
        low = F.avg_pool3d(F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate'), 3, stride=1)
        return x + self.scale.tanh() * self.gate(x).sigmoid() * (x - low)


class SegMambaV2Residual3D(nn.Module):
    """Call the unmodified official MambaLayer, with an external small residual.

    U(F)=F+sum_j P_j^-1 M(LN(P_j F)); output=F+tanh(alpha)*(U(F)-F)/3.
    FP32 protects selective-state computation under nnU-Net FP16 autocast.
    No fake convolution/CPU fallback is substituted for Mamba.
    """
    def __init__(self, channels):
        super().__init__()
        try:
            from .vendor.segmambav2 import MambaLayer
            self.official = MambaLayer(dim=channels, d_state=16, d_conv=4, expand=2)
        except (ImportError, OSError) as error:
            raise RuntimeError('Mamba requires compatible mamba_ssm, causal_conv1d, MONAI and einops; run preflight --mamba.') from error
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), 1e-3))

    def forward(self, x):
        if x.device.type != 'cuda':
            raise RuntimeError('Official selective-scan Mamba requires CUDA; no CPU surrogate is used.')
        with torch.autocast(device_type='cuda', enabled=False):
            xf = x.float()
            out = xf + self.scale.float().tanh() * (self.official(xf) - xf) / 3
        return out.to(x.dtype)
