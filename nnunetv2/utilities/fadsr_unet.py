import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.utilities.FADSR.fadsr import FADSR


class FADSR_PlainConvUNet(nn.Module):
    def __init__(self, input_channels, num_classes,
                 n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                 n_conv_per_stage, n_conv_per_stage_decoder,
                 conv_bias, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision=True,
                 fadsr_topk=2, **kwargs):
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
        self.fadsr_modules = nn.ModuleList([
            FADSR(
                enc_dim=features_per_stage[d],
                use_skip_fuse=False,        
            )
            for d in range(num_enc)
        ])

        self.deep_supervision = deep_supervision

    def forward(self, x):
        skips = list(self.encoder(x))
        for d in range(len(skips) - 1):
            skips[d] = self.fadsr_modules[d](skips[d], stage_id=d)
        return self.decoder(skips)
