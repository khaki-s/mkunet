"""Keep spatial feature attention separate from final region-logit routing."""
import json
from pathlib import Path

import numpy as np
import torch


def feature_summary(network, parts):
    attention = parts['feature_attention']
    gradients = [p.grad.detach().float().square().sum()
                 for p in network.feature_injection.parameters() if p.grad is not None]
    return dict(
        active=bool(network.feature_injection_active),
        gradient_l2=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.,
        decoder_stage=network.decoder.injection_stage,
        scope='spatial expert attention, latent heads; not causal importance',
        attention_mean=(attention.float().mean((3, 4, 5)).cpu().tolist() if attention is not None else None),
        attention_spatial_std=(attention.float().flatten(3).std(-1, unbiased=False).cpu().tolist()
                               if attention is not None else None),
        relative_update_rms=(parts['feature_update_relative_rms'].cpu().tolist()
                             if parts['feature_update_relative_rms'] is not None else None))


def export_feature_attention(parts, filename, *, patch_metadata):
    """Export one forward's feature maps, without pretending they are case maps.

    patch_metadata must describe the caller's window origin, orientation and
    mirroring. This utility does not reverse TTA or resample native-space MRI.
    Output `.npz` deliberately differs from existing output-router CSVs.
    """
    path = Path(filename)
    if path.suffix != '.npz':
        raise ValueError('Spatial attention export requires a .npz destination')
    if not isinstance(patch_metadata, dict) or not patch_metadata:
        raise ValueError('Supply patch/window coordinate and TTA metadata')
    attention = parts['feature_attention']
    if attention is None:
        raise ValueError('Feature injection was inactive, disabled, or return_aux was False')
    metadata = dict(patch_metadata=patch_metadata,
                    axes=['batch', 'latent_attention_head', 'expert', 'D', 'H', 'W'],
                    expert_names=list(parts['expert_names']),
                    note='preprocessed feature grid; neither native-space nor causal contribution')
    encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, feature_attention=attention.detach().float().cpu().numpy(),
        physical_mask=parts['modality_mask'].detach().cpu().numpy(),
        expert_mask=parts['expert_mask'].detach().cpu().numpy(),
        output_routing=parts['weights'].detach().float().cpu().numpy(),
        metadata_json=np.array(encoded))
