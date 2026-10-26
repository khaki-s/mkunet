"""Generalize existing directly supervised losses to five observed experts.

Dice+BCE is verified against SimMLM ICCV 2025 loss/dice_bce_loss.py.
We retain the established nnU-Net main/aux losses and coefficients for the
controlled comparison. No RL reward, MoFe ranking or LLM loss is claimed.
"""
import torch


def observed_expert_loss(experts, target, mask, region_loss):
    if mask.ndim != 2 or mask.shape[1] != len(experts) or not mask.any(1).all():
        raise ValueError('A nonempty observed-expert set is required per patient')
    per_sample, by_expert = [], [[] for _ in experts]
    for b in range(mask.shape[0]):
        values = []
        for m, logits in enumerate(experts):
            if mask[b, m]:
                value = region_loss(logits[b:b+1].float(), target[b:b+1])
                values.append(value); by_expert[m].append(value.detach())
        per_sample.append(torch.stack(values).mean())
    means = [torch.stack(v).mean().item() if v else None for v in by_expert]
    return torch.stack(per_sample).mean(), means
