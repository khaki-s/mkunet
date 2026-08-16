import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.FADSR.blocks import FADSR, BottleneckASPPCBAM
 

class FADSR_PlainConvUNet(nn.Module):
    def __init__(self, input_channels, num_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision=True,
                 fadsr_topk=2,
                 use_bottleneck_attn=True,
                 bottleneck_dilations=(1, 2), bottleneck_reduction=16,
                 bottleneck_spatial_kernel=7, bottleneck_layer_scale_init=1e-6,
                 fadsr_use_base_branch=False, fadsr_base_kernel=5,
                 fadsr_mixer_scale_init=0.0,
                 patch_size=None,
                 **kwargs):
        super().__init__()

        unet = PlainConvUNet(
            input_channels=input_channels,
            num_classes=num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )

        self.encoder = unet.encoder
        self.decoder = unet.decoder

        num_enc = n_stages - 1

        if use_bottleneck_attn and conv_op is nn.Conv2d:
            bottleneck_block = BottleneckASPPCBAM(
                features_per_stage[-1],
                dilations=bottleneck_dilations,
                reduction=bottleneck_reduction,
                spatial_kernel=bottleneck_spatial_kernel,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                layer_scale_init=bottleneck_layer_scale_init,
            )
            stage = self.encoder.stages[-1]
            stage.add_module(str(len(stage)), bottleneck_block)

        # Build only the wavelet levels each skip can actually use. The cascade
        # is capped by resolution (FADSR.effective_levels), so with a fixed
        # wt_levels=3 the two coarsest skips carried mixers that were never
        # called: 10.5M of 19.1M mixer params, ~25% of the whole network, still
        # saved into checkpoints and still walked by the optimizer.
        skip_sizes = self._skip_sizes(patch_size, strides, num_enc)
        self.fadsr_modules = nn.ModuleList([
            FADSR(enc_dim=features_per_stage[d],
                  wt_levels=(FADSR.levels_for(*skip_sizes[d])
                             if skip_sizes is not None else 3),
                  use_base_branch=fadsr_use_base_branch,
                  base_kernel=fadsr_base_kernel,
                  mixer_scale_init=fadsr_mixer_scale_init)
            for d in range(num_enc)
        ])

    @staticmethod
    def _skip_sizes(patch_size, strides, num_enc):
        """Spatial size of each skip, or None if patch_size was not supplied."""
        if patch_size is None:
            return None
        cur = list(patch_size)
        out = []
        for d in range(num_enc):
            s = strides[d]
            s = [s] * len(cur) if isinstance(s, int) else list(s)
            cur = [c // f for c, f in zip(cur, s)]
            out.append(tuple(cur))
        return out

        self.deep_supervision = deep_supervision

    def forward(self, x):
        skips = list(self.encoder(x))
        for d in range(len(skips) - 1):
            skips[d] = self.fadsr_modules[d](skips[d], stage_id=d)
        return self.decoder(skips)