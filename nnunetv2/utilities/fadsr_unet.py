from math import prod

import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from nnunetv2.utilities.FADSR.blocks import (
    MedNeXtContextBlock3D, EMCADMultiScaleBlock3D, WTConvResidualBlock3D,
)


class ModuleRefinedPlainConvUNet(nn.Module):
    """Wrap an already initialized PlainConvUNet, preserving shared key names.

    Encoder indices start at zero (highest resolution). Decoder placement is
    computed from this index, rather than assuming a fixed number of stages.
    """
    _BLOCKS = {
        'mednext_encoder': MedNeXtContextBlock3D,
        'emcad_decoder': EMCADMultiScaleBlock3D,
        'wtconv_skip': WTConvResidualBlock3D,
    }

    def __init__(self, backbone: PlainConvUNet, direction: str, refine_stages=(1, 2)):
        super().__init__()
        if not isinstance(backbone, PlainConvUNet) or backbone.encoder.conv_op is not nn.Conv3d:
            raise ValueError('These experiments require a Conv3d PlainConvUNet configuration')
        if direction not in self._BLOCKS:
            raise ValueError(f'Unknown direction: {direction}')
        stages = tuple(refine_stages)
        n_stages = len(backbone.encoder.stages)
        if not stages or len(set(stages)) != len(stages) or any(
                not isinstance(s, int) or s < 0 or s >= n_stages - 1 for s in stages):
            raise ValueError('refine_stages must contain unique non-bottleneck encoder indices')
        self.encoder, self.decoder = backbone.encoder, backbone.decoder
        self.direction, self.refine_stages = direction, stages
        self.refiners = nn.ModuleDict({
            str(s): self._BLOCKS[direction](self.encoder.output_channels[s]) for s in stages
        })
        # Initialize only the new blocks: shared backbone is never reinitialized.
        self.refiners.apply(PlainConvUNet.initialize)
        self.experiment_config = {
            'direction': direction, 'refine_stages': list(stages),
            'layer_scale_init': 1e-2, 'training_framework': 'unchanged nnU-Net',
            'module_settings': {
                'mednext_encoder': {'kernel_size': 5, 'expansion': 2},
                'emcad_decoder': {'kernel_sizes': [1, 3, 5], 'expansion': 2},
                'wtconv_skip': {'kernel_size': 3, 'wavelet': 'haar', 'levels': 1,
                                'band_scale_init': 0.1},
            }[direction],
        }

    def forward(self, x):
        if self.direction == 'mednext_encoder':
            skips = []
            for index, stage in enumerate(self.encoder.stages):
                x = stage(x)
                if str(index) in self.refiners:
                    x = self.refiners[str(index)](x)
                skips.append(x)
        else:
            skips = self.encoder(x)
        if self.direction == 'wtconv_skip':
            skips = [self.refiners[str(i)](s) if str(i) in self.refiners else s
                     for i, s in enumerate(skips)]
        if self.direction != 'emcad_decoder':
            return self.decoder(skips)
        return self._decode_with_refinement(skips)

    def _decode_with_refinement(self, skips):
        low = skips[-1]
        outputs = []
        for index, stage in enumerate(self.decoder.stages):
            up = self.decoder.transpconvs[index](low)
            skip = skips[-(index + 2)]
            x = stage(torch.cat([up, skip], dim=1))
            encoder_index = len(skips) - index - 2
            if str(encoder_index) in self.refiners:
                x = self.refiners[str(encoder_index)](x)
            if self.decoder.deep_supervision or index == len(self.decoder.stages) - 1:
                outputs.append(self.decoder.seg_layers[index](x))
            low = x
        outputs.reverse()
        return outputs if self.decoder.deep_supervision else outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        if len(input_size) != 3:
            raise ValueError('Expected three spatial dimensions, without batch/channels')
        total = (self.encoder.compute_conv_feature_map_size(input_size)
                 + self.decoder.compute_conv_feature_map_size(input_size))
        shape = list(input_size)
        for index, stride in enumerate(self.encoder.strides):
            shape = [(s + st - 1) // st for s, st in zip(shape, stride)]
            if str(index) in self.refiners:
                total += self.refiners[str(index)].compute_conv_feature_map_size(shape)
        return total


# Historical checkpoint compatibility; implementations live in legacy_fadsr.
_LEGACY_NAMES = ('FADSR_PlainConvUNet',)
__all__ = ['ModuleRefinedPlainConvUNet'] + [n for n in _LEGACY_NAMES if not n.startswith("_")]


def __getattr__(name):
    if name in _LEGACY_NAMES:
        from importlib import import_module
        return getattr(import_module('nnunetv2.utilities.legacy_fadsr.network'), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
