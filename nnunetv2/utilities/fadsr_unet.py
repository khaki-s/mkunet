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
                dec_dim=self.decoder.transpconvs[n_stages - d - 2].out_channels,
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

        # dec = self.decoder
        # seg_outputs = []
 
        # x = skips[-1]
 
        # num_dec_stages = len(dec.stages)
 
        # for s in range(num_dec_stages):
        #     x = dec.transpconvs[s](x)
 
        #     skip_idx = len(skips) - s - 2  
        #     refined_skip = self.fadsr_modules[skip_idx](
        #         skips[skip_idx],
        #         stage_id=skip_idx,
        #         dec_feat=x                   
        #     )

        #     x = torch.cat((x, refined_skip), 1)
 
        #     x = dec.stages[s](x)
 
        #     if self.deep_supervision:
        #         seg_outputs.append(dec.seg_layers[s](x))
 
        # if self.deep_supervision:
        #     return seg_outputs[::-1]
        # else:
        #     return dec.seg_layers[-1](x)