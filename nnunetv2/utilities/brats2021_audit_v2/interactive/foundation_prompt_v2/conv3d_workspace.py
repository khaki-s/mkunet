"""Bound intermediate Conv3D workspace and retain all output voxels."""
from functools import partial
import math

import torch
from torch.nn import functional as F


def conv3d_depth_tiles(image, weight, bias, tile_depth):
    """Compute stride-one, 3×3×3 convolution with one input halo per tile."""
    depth = image.shape[-3]
    result = None
    for start in range(0, depth, tile_depth):
        stop = min(start + tile_depth, depth)
        slab = image[:, :, max(0, start - 1):min(depth, stop + 1)]
        if start == 0 or stop == depth:
            slab = F.pad(slab, (0, 0, 0, 0, int(start == 0), int(stop == depth)))
        part = F.conv3d(slab.contiguous(), weight, bias, padding=(0, 1, 1))
        if result is None:
            result = part.new_empty((*part.shape[:2], depth, *part.shape[-2:]))
        result[:, :, start:stop] = part
    return result


def _forward(layer, tile_depth, image):
    # Bound large temporary lowering buffers in the decoder. Small feature
    # maps continue through the usual PyTorch dispatch.
    lowering_bytes_fp16 = math.prod(image.shape) * 27 * 2
    if image.ndim == 5 and image.is_cuda and lowering_bytes_fp16 > 4 * 2**30:
        return conv3d_depth_tiles(image, layer.weight, layer.bias, tile_depth)
    return layer._conv_forward(image, layer.weight, layer.bias)


def install_depth_tiles(network, tile_depth):
    if type(tile_depth) is not int or not 1 <= tile_depth < 192:
        raise ValueError('Conv3D tile depth must be an integer in [1,191]')
    for layer in network.modules():
        if (type(layer) is torch.nn.Conv3d and layer.groups == 1 and layer.padding_mode == 'zeros'
                and layer.kernel_size == (3, 3, 3) and layer.padding == (1, 1, 1)
                and layer.stride == layer.dilation == (1, 1, 1)):
            layer.forward = partial(_forward, layer, tile_depth)
