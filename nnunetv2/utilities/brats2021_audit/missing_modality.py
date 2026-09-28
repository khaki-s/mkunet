"""Explicit case-level availability and stateless modality sampling.

Channel order is always T1/T1ce/T2/FLAIR; a zero-valued patch is NOT absence.
"""
import hashlib
import itertools

import numpy as np
import torch

MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')
MASK_CODES = (1, 2, 4, 8, 3, 5, 9, 6, 10, 12, 7, 11, 13, 14, 15)


def mask_from_code(code, *, device=None):
    if isinstance(code, bool) or int(code) != code or not 1 <= int(code) <= 15:
        raise ValueError('mask_code must be an integer in 1..15')
    return torch.tensor([(int(code) >> i) & 1 for i in range(4)], dtype=torch.bool, device=device)


def validate_mask(mask, batch_size, *, device=None):
    if mask is None:
        raise ValueError('Explicit modality_mask is required; use all ones for complete input')
    mask = torch.as_tensor(mask, device=device)
    if tuple(mask.shape) != (batch_size, 4):
        raise ValueError(f'Expected availability [{batch_size},4], got {tuple(mask.shape)}')
    if not ((mask == 0) | (mask == 1)).all().item():
        raise ValueError('Availability must contain only 0/1 values')
    mask = mask.bool()
    if not mask.any(1).all().item():
        raise ValueError('At least one modality must be present in every sample')
    return mask


def mask_codes(mask):
    mask = validate_mask(mask, len(mask))
    return (mask.long() * torch.tensor([1, 2, 4, 8], device=mask.device)).sum(1)


def apply_modality_mask(x, mask):
    if x.ndim != 5 or x.shape[1] != 4:
        raise ValueError('Expected a B x 4 x D x H x W image')
    mask = validate_mask(mask, x.shape[0], device=x.device)
    if not torch.isfinite(x).all().item():
        raise ValueError('Input contains NaN/Inf')
    return x.masked_fill(~mask[:, :, None, None, None], 0), mask


def sample_masks(seed, logical_step, batch_size, *, stream=0, policy='cardinality',
                 available=None, device=None):
    """Independent RNG per logical sample; never consumes augmentation RNG.

    Supports physically incomplete input by sampling only nonempty subsets of
    its availability. Cardinality-uniform gives full input probability 1/4.
    """
    if logical_step < 0 or batch_size < 1:
        raise ValueError('Nonnegative step and positive batch size required')
    if policy not in ('cardinality', 'uniform15'):
        raise ValueError(f'Unknown mask sampling policy {policy}')
    availability = (torch.ones(batch_size, 4, dtype=torch.bool) if available is None
                    else validate_mask(available, batch_size).cpu())
    result = torch.zeros_like(availability)
    for b in range(batch_size):
        key = f'brats_mask_v1:{seed}:{logical_step}:{b}:{stream}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        indices = torch.where(availability[b])[0].tolist()
        if policy == 'cardinality':
            k = int(rng.integers(1, len(indices) + 1))
            choices = list(itertools.combinations(indices, k))
        else:
            choices = [s for k in range(1, len(indices) + 1) for s in itertools.combinations(indices, k)]
        result[b, list(choices[int(rng.integers(len(choices)))])] = True
    return result.to(device=device)
