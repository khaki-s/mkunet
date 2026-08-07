import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.FADSR.blocks import FADSR, LiteInceptionNeXtBlock2d, BottleneckASPPCBAM


class FADSR_PlainConvUNet(nn.Module):
    def __init__(self, input_channels, num_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision=True,
                 fadsr_topk=2,
                 use_spatial_block=False,
                 spatial_mlp_ratio=2, spatial_branch_ratio=0.25,
                 spatial_square_kernel=3, spatial_band_kernel=7,
                 spatial_drop_path=0.1, spatial_layer_scale_init=1e-6,
                 spatial_stage_start=0,
                 use_bottleneck_attn=True,
                 bottleneck_dilations=(1, 2, 3), bottleneck_reduction=16,
                 bottleneck_spatial_kernel=7, bottleneck_layer_scale_init=1e-6,
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

        if use_spatial_block and conv_op is nn.Conv2d:
            for d in range(spatial_stage_start, num_enc):
                block = LiteInceptionNeXtBlock2d(
                    features_per_stage[d],
                    mlp_ratio=spatial_mlp_ratio,
                    branch_ratio=spatial_branch_ratio,
                    square_kernel=spatial_square_kernel,
                    band_kernel=spatial_band_kernel,
                    drop_path=spatial_drop_path,
                    layer_scale_init=spatial_layer_scale_init,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                )
                stage = self.encoder.stages[d]
                stage.add_module(str(len(stage)), block)

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

        self.fadsr_modules = nn.ModuleList([
            FADSR(enc_dim=features_per_stage[d])
            for d in range(num_enc)
        ])

        self.deep_supervision = deep_supervision

    def forward(self, x):
        skips = list(self.encoder(x))
        for d in range(len(skips) - 1):
            skips[d] = self.fadsr_modules[d](skips[d], stage_id=d)
        return self.decoder(skips)