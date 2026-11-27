"""Residual 3D U-Net that corrects the nnU-Net baseline via a zero-initialised delta.

The network predicts a four-class residual ``delta`` that is added to the nested
baseline prior logits (``data3d.prior_logits``). The final 1x1x1 convolution is
zero-initialised, so at the start of training the output equals the baseline and
the region scores equal arm P. Region probabilities r_WT/r_TC/r_ET come from the
four-class softmax, so ET<=TC<=WT holds by construction.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ...common import check_file, file_identity

SCHEMA = "brats_missing15_residual3d_v2"
INPUT_CHANNELS = 15
CLASSES = 4


def _norm(channels):
    return nn.InstanceNorm3d(channels, affine=True)


class ResidualBlock(nn.Module):
    """Two 3x3x3 Conv-InstanceNorm-LeakyReLU layers with an identity/projection skip."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.skip = (nn.Conv3d(in_channels, out_channels, 1, bias=False)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.skip(x))


class ResidualUNet3D(nn.Module):
    """5-level residual encoder/decoder; 2 blocks per encoder level, 1 per decoder level."""

    def __init__(self, in_channels=INPUT_CHANNELS, channels=(32, 64, 128, 256, 320)):
        super().__init__()
        self.channels = tuple(channels)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(nn.Sequential(ResidualBlock(previous, width),
                                                ResidualBlock(width, width)))
            previous = width
        for i in range(len(channels) - 1):
            self.downs.append(nn.Conv3d(channels[i], channels[i], 2, stride=2, bias=False))
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose3d(channels[i], channels[i - 1], 2, stride=2, bias=False))
            self.decoders.append(ResidualBlock(2 * channels[i - 1], channels[i - 1]))
        self.head = nn.Conv3d(channels[0], CLASSES, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, prior):
        skips = []
        x = features
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            if level < len(self.downs):
                skips.append(x)
                x = self.downs[level](x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = decoder(torch.cat([x, skip], dim=1))
        delta = self.head(x)
        return prior + delta

    def configuration(self):
        return {"schema": SCHEMA, "in_channels": self.encoders[0][0].conv1.in_channels,
                "channels": list(self.channels), "classes": CLASSES}


def regions_from_softmax_t(posterior):
    """Torch nested region probabilities r_WT/r_TC/r_ET from a four-class posterior (B,4,...)."""
    r_wt = 1.0 - posterior[:, 0]
    r_tc = posterior[:, 2] + posterior[:, 3]
    r_et = posterior[:, 3]
    return torch.stack([r_wt, r_tc, r_et], dim=1)


def build_model(configuration=None, device="cpu"):
    config = configuration or {"channels": (32, 64, 128, 256, 320), "in_channels": INPUT_CHANNELS}
    if configuration is not None and configuration.get("schema") != SCHEMA:
        raise ValueError("Unsupported residual3d configuration")
    model = ResidualUNet3D(config.get("in_channels", INPUT_CHANNELS),
                           tuple(config.get("channels", (32, 64, 128, 256, 320))))
    return model.to(device)


def implementation():
    folder = Path(__file__).parent
    names = ("model3d.py", "data3d.py", "train3d.py", "infer3d.py")
    return {name: file_identity(folder / name)["sha256"] for name in names}


def atomic_save(value, path):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def restore(identity, device):
    check_file(identity)
    state = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if state["schema"] != SCHEMA:
        raise ValueError("Checkpoint schema mismatch for residual3d")
    model = build_model(state["configuration"], device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, state
