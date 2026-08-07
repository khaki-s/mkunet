import torch
import torch.nn as nn
from nnunetv2.utilities.fadsr_unet import FADSR_PlainConvUNet

net = FADSR_PlainConvUNet(
    input_channels=4,
    num_classes=4,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 512, 512],
    conv_op=nn.Conv2d,
    kernel_sizes=[[3, 3]] * 6,
    strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
    n_conv_per_stage=[2] * 6,
    n_conv_per_stage_decoder=[2] * 5,
    conv_bias=True,
    norm_op=nn.InstanceNorm2d,
    norm_op_kwargs={'affine': True, 'eps': 1e-5},
    dropout_op=None,
    dropout_op_kwargs=None,
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    deep_supervision=True,
)
print('built net')

n_total = sum(p.numel() for p in net.parameters())
base_stage_params = sum(
    p.numel() for d in range(5) for p in net.encoder.stages[d][0].parameters()
)
spatial_block_params = sum(
    p.numel() for d in range(5) for p in net.encoder.stages[d][1].parameters()
)
print('total params:', n_total)
print('spatial block params (5 stages):', spatial_block_params,
      f'({100 * spatial_block_params / n_total:.2f}% of total)')

x = torch.randn(1, 4, 96, 80)
out = net(x)
print('output type:', type(out), 'n outputs:', len(out) if isinstance(out, list) else 1)
for o in (out if isinstance(out, list) else [out]):
    print('  shape:', o.shape)

loss = sum(o.float().pow(2).mean() for o in out) if isinstance(out, list) else out.pow(2).mean()
loss.backward()
n_no_grad = sum(1 for p in net.parameters() if p.requires_grad and p.grad is None)
print('params with requires_grad but no grad after backward:', n_no_grad)
print('OK')
