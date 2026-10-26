"""Replace only expert five with an independent plans-sized PlainConvUNet.

The four availability channels of the old generalist are retained. Therefore
the encoder/decoder topology and stage widths match the joint backbone, but
the first convolution has eight inputs instead of four. This is an expert
architecture/capacity ablation, not a change of information-fusion order.
"""
import torch
from torch import nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from ...baseline.network import build_audit_network
from ..generalist.network import BraTSGeneralistAgentUNet


class PlansSizedFeatureUNet(nn.Module):
    """Call the official encoder/decoder, exposing one decoder state by a hook.

    No upstream decoder implementation is copied or modified. The temporary
    hook is local to one forward call and removed even on error, including
    checkpoint recomputation. The five-expert branch always uses its final
    region head; unused coarse heads are retained for stable state dictionaries
    and frozen, as in the original lightweight experts.
    """
    def __init__(self, configuration_manager, feature_level):
        super().__init__()
        official = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            input_channels=8, output_channels=3, allow_init=True,
            deep_supervision=False)
        self.encoder, self.decoder = official.encoder, official.decoder
        self.feature_level = int(feature_level)
        self.feature_channels = int(self.encoder.output_channels[self.feature_level])
        self.feature_stage = len(self.encoder.output_channels) - 2 - self.feature_level
        if not 0 <= self.feature_stage < len(self.decoder.stages):
            raise ValueError('Selected Generalist feature level must be a decoder level')
        for head in self.decoder.seg_layers[:-1]:
            head.requires_grad_(False)

    def forward(self, x, return_features=False):
        if x.ndim != 5 or x.shape[1] != 8:
            raise ValueError('Generalist expects four masked MRI plus four availability channels')
        skips = self.encoder(x)
        if not return_features:
            return self.decoder(skips)
        captured = []
        handle = self.decoder.stages[self.feature_stage].register_forward_hook(
            lambda module, inputs, output: captured.append(output))
        try:
            logits = self.decoder(skips)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one official decoder feature')
        return logits, captured[0]


class BraTSFullSizeGeneralistLegacyUNet(BraTSGeneralistAgentUNet):
    """Legacy peer communication and logit fusion, larger fifth expert only."""
    def __init__(self, backbone, configuration_manager, **options):
        if options.get('include_self', False) or options.get('output_gate', False):
            raise ValueError('This ablation preserves LegacyV2 peer-only, ungated communication')
        options.update(include_self=False, output_gate=False)
        # Construct the entire old network first: all retained weights have the
        # same initialization and RNG consumption as the matched LegacyV2 run.
        super().__init__(backbone, configuration_manager, **options)
        with torch.random.fork_rng(devices=[]):
            expert = PlansSizedFeatureUNet(configuration_manager, self.semantic.feature_level)
            self.semantic.experts[4] = expert
            # State grids match because both branches use the same plan strides.
            # Only this projection changes shape; all peer attention layers and
            # all four specialist projections keep their old initial tensors.
            self.communication.project[4] = nn.Conv3d(
                expert.feature_channels, self.communication.channels, 1)
        self.architecture_version.fill_(5)
        self.set_epoch(0)


def build_fullsize_generalist_legacy_network(configuration_manager, num_input_channels=4,
                                             num_output_channels=3,
                                             enable_deep_supervision=True, **options):
    backbone = build_audit_network(configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision, 'baseline')
    with torch.random.fork_rng(devices=[]):
        return BraTSFullSizeGeneralistLegacyUNet(backbone, configuration_manager, **options)
