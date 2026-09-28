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
                 fadsr_mixer_scale=0.0,
                 fadsr_wt_type='db2', fadsr_wt_levels=3, fadsr_dw_kernel=7,
                 fadsr_dw_kernels=None,
                 fadsr_mixer_dilations=(2, 1, 1), fadsr_min_level_size=4,
                 fadsr_subband_layout='stack',
                 fadsr_hf_shrink=False, fadsr_hf_shrink_init=1e-3,
                 fadsr_ll_enhance=False, fadsr_ll_kernel=3,
                 fadsr_ll_gate_init=1e-2,
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

        # Drive FADSR's dimensionality off conv_op, so 3d_fullres works with no
        # extra flag. LiteInceptionNeXtBlock2d and BottleneckASPPCBAM are 2D-only
        # and stay gated on Conv2d below, exactly as before.
        if conv_op is nn.Conv3d:
            fadsr_ndim = 3
        elif conv_op is nn.Conv2d:
            fadsr_ndim = 2
        else:
            raise ValueError(f'FADSR supports Conv2d/Conv3d, got {conv_op}')

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
            FADSR(enc_dim=features_per_stage[d],
                  wt_type=fadsr_wt_type,
                  wt_levels=fadsr_wt_levels,
                  dw_kernel=fadsr_dw_kernel,
                  dw_kernels=fadsr_dw_kernels,
                  mixer_scale=fadsr_mixer_scale,
                  mixer_dilations=fadsr_mixer_dilations,
                  min_level_size=fadsr_min_level_size,
                  subband_layout=fadsr_subband_layout,
                  hf_shrink=fadsr_hf_shrink,
                  hf_shrink_init=fadsr_hf_shrink_init,
                  ll_enhance=fadsr_ll_enhance,
                  ll_kernel=fadsr_ll_kernel,
                  ll_gate_init=fadsr_ll_gate_init,
                  norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                  ndim=fadsr_ndim)
            for d in range(num_enc)
        ])

        # Second pass, on purpose: LLEnhance owns conv weights, so building it
        # inside the loop above would shift the RNG for every later stage's mixer
        # and stop this from being a clean single-knob change against the same
        # config with ll_enhance off.
        for m in self.fadsr_modules:
            m.attach_ll_enhance()

        self.fadsr_config = dict(
            ndim=fadsr_ndim,
            wt_type=fadsr_wt_type, wt_levels=fadsr_wt_levels,
            dw_kernel=fadsr_dw_kernel,
            dw_kernels=(tuple(fadsr_dw_kernels) if fadsr_dw_kernels else None),
            mixer_scale=fadsr_mixer_scale,
            mixer_dilations=tuple(fadsr_mixer_dilations),
            min_level_size=fadsr_min_level_size,
            subband_layout=fadsr_subband_layout,
            hf_shrink=fadsr_hf_shrink,
            ll_enhance=fadsr_ll_enhance,
            use_spatial_block=use_spatial_block,
            use_bottleneck_attn=use_bottleneck_attn,
        )

        self.deep_supervision = deep_supervision

    @torch.no_grad()
    def fadsr_gate_report(self):
        """{stage -> [[per-subband gain] per wavelet level]} for logging.

        4 bands in 2D (LL, LH, HL, HH), 8 in 3D (LLL..HHH); index 0 is always the
        approximation band.
        """
        return {d: m.gate_report() for d, m in enumerate(self.fadsr_modules)}

    @torch.no_grad()
    def fadsr_ll_report(self):
        """{stage -> mean |LLEnhance gate|}, empty if disabled."""
        return {d: m.ll_enhance.report()
                for d, m in enumerate(self.fadsr_modules)
                if m.ll_enhance is not None}

    @torch.no_grad()
    def fadsr_shrink_report(self):
        """{stage -> [[LH, HL, HH] thresholds per level]}, empty if disabled."""
        out = {}
        for d, m in enumerate(self.fadsr_modules):
            r = m.shrink_report()
            if r is not None:
                out[d] = r
        return out

    def forward(self, x):
        skips = list(self.encoder(x))
        for d in range(len(skips) - 1):
            skips[d] = self.fadsr_modules[d](skips[d], stage_id=d)
        return self.decoder(skips)