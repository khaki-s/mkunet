"""Region-semantic logit fusion; an adaptation, not a SimMLM reproduction."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vendor.simmlm_unet import UNet


MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')
REGIONS = ('WT', 'TC', 'ET')
EXPERT_WIDTHS = (8, 16, 32, 64, 80, 80)


class RegionSemanticRouter3D(nn.Module):
    """One B x region x modality distribution per patch, no GT input.

    Five Conv/IN/ReLU stages follow SimMLM's router design. The linear input
    size is derived from the current plans, avoiding the original 128^3 lock
    and nondeterministic adaptive-pooling backward on CUDA.
    """
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = tuple(patch_size)
        layers = []
        for _ in range(5):
            layers.extend((nn.Conv3d(4, 4, 3, stride=2, padding=1),
                           nn.InstanceNorm3d(4, eps=1e-5, affine=True), nn.ReLU(inplace=True)))
        self.features = nn.Sequential(*layers)
        spatial = [math.ceil(n / 32) for n in patch_size]
        if math.prod(spatial) <= 1:
            raise ValueError('Router InstanceNorm needs more than one final spatial element')
        self.head = nn.Linear(4 * math.prod(spatial), 12)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        if tuple(x.shape[2:]) != self.patch_size:
            raise ValueError(f'Expected planned router patch {self.patch_size}, got {tuple(x.shape[2:])}')
        return self.head(self.features(x).flatten(1)).reshape(x.shape[0], 3, 4).float().softmax(-1)


class SemanticModalityExperts3D(nn.Module):
    """Four directly supervised official UNets plus an explicit joint blend.

    z_moe,c=sum_m w_cm z_mc; z_final,c=(1-beta_c)z_joint,c+beta_c z_moe,c.
    Auxiliary losses supervise z_m and z_moe directly, outside beta scaling.
    """
    def __init__(self, kernel_sizes, strides, patch_size, uniform=False):
        super().__init__()
        if len(strides) != 6 or len(kernel_sizes) != 6:
            raise ValueError('The pinned lightweight SimMLM experts require six stages')
        self.uniform = bool(uniform)
        self.checkpoint_experts = True
        self.experts = nn.ModuleList([
            UNet(1, 3, 6, list(EXPERT_WIDTHS), kernel_sizes, strides, apply_deep_supervision=False)
            for _ in MODALITIES])
        # Upstream creates low-resolution heads even with DS off; they are unused.
        for expert in self.experts:
            for head in expert.seg_layers[:-1]:
                head.requires_grad_(False)
        # Construct identically in both variants, including the frozen control.
        self.router = RegionSemanticRouter3D(patch_size)
        self.beta_logit = nn.Parameter(torch.full((3,), math.log(.2 / .8)))
        self.register_buffer('fusion_active', torch.tensor(False))
        self.set_joint_stage(False)

    def set_joint_stage(self, active):
        self.fusion_active.fill_(bool(active))
        self.router.requires_grad_(bool(active) and not self.uniform)
        self.beta_logit.requires_grad_(bool(active))

    def weights(self, x, active=None):
        active = bool(self.fusion_active.item()) if active is None else active
        if self.uniform or not active:
            return torch.full((x.shape[0], 3, 4), .25, device=x.device, dtype=torch.float32)
        return self.router(x)

    @staticmethod
    def fuse(experts, weights):
        # Accumulate in FP32 without a B x expert x region x volume stack.
        return sum(z.float() * weights[:, :, m, None, None, None] for m, z in enumerate(experts))

    def blend(self, joint, moe, active=None):
        active = bool(self.fusion_active.item()) if active is None else active
        if not active:
            return joint
        beta = self.beta_logit.float().sigmoid()[None, :, None, None, None]
        return (1 - beta) * joint.float() + beta * moe.float()

    def forward(self, x, joint):
        experts = []
        for m, expert in enumerate(self.experts):
            image = x[:, m:m+1]
            if self.training and torch.is_grad_enabled() and self.checkpoint_experts:
                z = checkpoint(expert, image, use_reentrant=False, preserve_rng_state=False)
            else:
                z = expert(image)
            experts.append(z)
        active = bool(self.fusion_active.item())
        weights = self.weights(x, active)
        moe = self.fuse(experts, weights)
        return {'final': self.blend(joint, moe, active), 'joint': joint, 'experts': experts,
                'moe': moe, 'weights': weights, 'active': active}
