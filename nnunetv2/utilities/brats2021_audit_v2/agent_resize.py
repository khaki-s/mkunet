"""Trilinear resize with a deterministic CUDA adjoint for PyTorch 2.0.

Forward is exactly torch interpolate(align_corners=False). Backward applies
its separable linear interpolation matrices in reverse order, avoiding the
atomic CUDA upsample backward. The existing CUBLAS workspace setting applies.
"""
import torch
from torch.nn import functional as F


class _DeterministicTrilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size):
        ctx.input_size = tuple(x.shape[2:])
        ctx.input_dtype = x.dtype
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)

    @staticmethod
    def backward(ctx, grad):
        # Accumulate in FP32 under AMP, FP64 for numerical gradient checks.
        dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.to(dtype)
            for axis in (4, 3, 2):
                n_in, n_out = ctx.input_size[axis - 2], g.shape[axis]
                if n_in == n_out:
                    continue
                pos = ((torch.arange(n_out, device=g.device, dtype=dtype) + .5)
                       * (n_in / n_out) - .5).clamp(min=0, max=n_in - 1)
                lo = pos.floor().long()
                hi = (lo + 1).clamp(max=n_in - 1)
                frac = pos - lo
                columns = torch.arange(n_in, device=g.device)[None, :]
                matrix = ((columns == lo[:, None]).to(dtype) * (1 - frac[:, None])
                          + (columns == hi[:, None]).to(dtype) * frac[:, None])
                g = torch.matmul(g.movedim(axis, -1), matrix).movedim(-1, axis)
        return g.to(ctx.input_dtype), None


def resize_trilinear(x, size):
    size = tuple(int(s) for s in size)
    if tuple(x.shape[2:]) == size:
        return x
    if x.is_cuda and torch.are_deterministic_algorithms_enabled() and x.requires_grad:
        return _DeterministicTrilinear.apply(x, size)
    return F.interpolate(x, size=size, mode='trilinear', align_corners=False)
