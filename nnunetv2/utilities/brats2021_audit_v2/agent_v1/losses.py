"""Present-expert auxiliary loss and independent LoMix-style 3D supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..agent_resize import resize_trilinear


def present_expert_loss(experts, target, mask, region_loss):
    """First average available branches per sample, then average samples."""
    per_sample, by_modality = [], [[] for _ in range(4)]
    for b in range(mask.shape[0]):
        values = []
        for m, z in enumerate(experts):
            if mask[b, m].item():
                value = region_loss(z[b:b + 1].float(), target[b:b + 1])
                values.append(value)
                by_modality[m].append(value.detach())
        per_sample.append(torch.stack(values).mean())
    means = [torch.stack(v).mean().item() if v else None for v in by_modality]
    return torch.stack(per_sample).mean(), means


def positive_region_loss(logits, target):
    if logits.shape != target.shape:
        raise ValueError(f'Region logit/target mismatch: {logits.shape} vs {target.shape}')
    p, y = logits.float().sigmoid(), target.float()
    axes = tuple(range(2, p.ndim))
    dice = (2 * (p * y).sum(axes) + 1e-5) / (p.sum(axes) + y.sum(axes) + 1e-5)
    return F.binary_cross_entropy_with_logits(logits.float(), y) + (1 - dice).mean()


class LoMixSupervision3D(nn.Module):
    """Training-only three-scale adaptation, not a literal LoMix reproduction."""
    combinations = ((0, 1), (0, 2), (0, 1, 2))

    def __init__(self, epsilon=.1):
        super().__init__()
        if not 0 < epsilon <= 1:
            raise ValueError('epsilon must be in (0,1]')
        self.epsilon = float(epsilon)
        self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(3, len(c))) for c in self.combinations])
        self.loss_logits = nn.Parameter(torch.zeros(len(self.combinations)))

    def loss_weights(self):
        return (1 - self.epsilon) * self.loss_logits.float().softmax(0) + self.epsilon / len(self.combinations)

    def forward(self, outputs, target):
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise ValueError('LoMix requires at least three joint decoder scales (deep supervision on)')
        if any(o.ndim != 5 or o.shape[1] != 3 for o in outputs[:3]):
            raise ValueError('LoMix expects three sigmoid-region logits at each 3D scale')
        if tuple(outputs[0].shape[2:]) != tuple(target.shape[2:]):
            raise ValueError('Highest-resolution target must match the main decoder output')
        if any(any(a < b for a, b in zip(outputs[i].shape[2:], outputs[i + 1].shape[2:])) for i in (0, 1)):
            raise ValueError('Decoder outputs must be ordered from highest to lowest resolution')
        losses = []
        for combo, theta in zip(self.combinations, self.alpha):
            def mixed_loss(alpha, *selected):
                weights = alpha.float().softmax(-1)
                z = sum(resize_trilinear(logit.float(), target.shape[2:]) * weights[:, i][None, :, None, None, None]
                        for i, logit in enumerate(selected))
                return positive_region_loss(z, target)
            args = (theta, *(outputs[s] for s in combo))
            value = (checkpoint(mixed_loss, *args, use_reentrant=False, preserve_rng_state=False)
                     if self.training and torch.is_grad_enabled() else mixed_loss(*args))
            losses.append(value)
        values = torch.stack(losses)
        return (self.loss_weights() * values).sum(), values.detach()
