"""Trainable residual corrector reusing dynamic_network_architectures.PlainConvUNet.

This is a BraTS-specific adapter, not an nnInteractive pretrained architecture.
Interaction/previous-prediction conditioning follows established interactive
segmentation practice. The upstream package supplies every UNet block.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet

INPUT_CHANNELS = 21
INPUT_LAYOUT = {
    "mri": (0, 4), "availability": (4, 8), "previous_probability": (8, 11),
    "target": (11, 14), "positive": (14, 17), "negative": (17, 20), "box": (20, 21),
}


@dataclass(frozen=True)
class CorrectorConfig:
    features: tuple = (16, 32, 64)
    convolutions_per_stage: int = 2
    max_delta: float = 8.0

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        if len(self.features) < 2 or any(type(v) is not int or v < 2 for v in self.features):
            raise ValueError("features requires at least two positive stages")
        if self.convolutions_per_stage < 1 or self.max_delta <= 0:
            raise ValueError("Invalid corrector configuration")


class TorchPromptCorrector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CorrectorConfig()
        stages = len(self.config.features)
        self.unet = PlainConvUNet(
            input_channels=INPUT_CHANNELS, n_stages=stages,
            features_per_stage=self.config.features, conv_op=nn.Conv3d,
            kernel_sizes=[3] * stages, strides=[1] + [2] * (stages - 1),
            n_conv_per_stage=self.config.convolutions_per_stage,
            num_classes=3,
            n_conv_per_stage_decoder=[self.config.convolutions_per_stage] * (stages - 1),
            conv_bias=True, norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
            deep_supervision=False,
        )
        # Every prediction starts as identity; only one zero-initialized output
        # projection, no zero-times-zero residual gate that blocks all learning.
        for layer in self.unet.decoder.seg_layers:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.is_trained_checkpoint = False

    def forward(self, model_input):
        if model_input.ndim != 5 or model_input.shape[1] != INPUT_CHANNELS:
            raise ValueError("Corrector expects [B,21,D,H,W]")
        if not torch.isfinite(model_input).all():
            raise ValueError("Corrector input contains nonfinite values")
        original_shape = model_input.shape[2:]
        divisor = 2 ** (len(self.config.features) - 1)
        # Keep at least two spatial samples per bottleneck axis for InstanceNorm.
        padded_shape = [max(2 * divisor, ((n + divisor - 1) // divisor) * divisor)
                        for n in original_shape]
        pads = []
        for n, padded in zip(reversed(original_shape), reversed(padded_shape)):
            pads.extend([0, padded - n])
        padded = F.pad(model_input, pads)
        raw = self.unet(padded)[(...,) + tuple(slice(0, n) for n in original_shape)]
        active = model_input[:, 14:].abs().flatten(1).amax(1) > 0
        delta = self.config.max_delta * torch.tanh(raw / self.config.max_delta)
        return delta * active[:, None, None, None, None].to(delta.dtype)


def save_corrector(model, path, training_metadata=None):
    """Save a self-describing adapter; never substitute a base segmentation checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "brats_prompt_corrector_v1",
        "config": asdict(model.config),
        "input_layout": INPUT_LAYOUT,
        "state_dict": model.state_dict(),
        "training_metadata": dict(training_metadata or {}),
        "trained": bool(model.is_trained_checkpoint),
    }, path)


def load_corrector(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "brats_prompt_corrector_v1" or payload.get("input_layout") != INPUT_LAYOUT:
        raise ValueError("Incompatible corrector checkpoint or input layout")
    model = TorchPromptCorrector(CorrectorConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.is_trained_checkpoint = bool(payload.get("trained", False))
    model.eval()
    return model
