"""Training interface for the original SAM3 decoder with its zero-up Q/V LoRA."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..common import check_file, file_identity
from ..sam_domain_model import build_domain_decoder, sam_supervised_loss, source_identity
from .prompts import slice_prompt

SCHEMA = "brats_missing15_sam3_lora_v1"


def implementation():
    folder = Path(__file__).parent
    names = ("model.py", "prompts.py", "dataset.py", "../sam_domain_model.py", "../sam_domain_data.py")
    return {name: file_identity(folder / name)["sha256"] for name in names}


def validate_encoder(contract):
    check_file(contract["sam_checkpoint"])
    if source_identity(contract["sam_source"]) != contract["sources"]:
        raise ValueError("The SAM3 source changed since feature encoding")


def new_model(encoder, device, rank=4):
    return build_domain_decoder(
        source=encoder["sam_source"], checkpoint=encoder["sam_checkpoint"]["path"],
        mode="qv_lora", lora_rank=rank, device=device,
    )


def restore(identity, encoder, device):
    check_file(identity)
    state = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if (state["schema"] != SCHEMA or state["encoder"] != encoder
            or state["implementation"] != implementation()):
        raise ValueError("Checkpoint and encoder or code contracts disagree")
    network = build_domain_decoder(configuration=state["configuration"], device=device)
    network.load_state_dict(state["state_dict"], strict=True)
    network.eval().requires_grad_(False)
    return network, state


def prepare_prompts(labels, probability, head, mask_size):
    values = slice_prompt(np.asarray(labels) > head, np.asarray(probability))
    prompts = {k: torch.from_numpy(values[k][None]) for k in ("points", "point_labels", "box")}
    dense = torch.from_numpy(values["dense"])[None, None]
    prompts["mask_input"] = F.interpolate(dense, size=mask_size, mode="bilinear", align_corners=False)
    return prompts


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def loss(logits, target, base, source, edit_weight, keep_weight):
    """Whole-mask supervision with a separately weighted baseline-error term."""
    target = target.to(logits.device, dtype=torch.float32)
    base = base.to(logits.device, dtype=torch.bool)
    seg, _ = sam_supervised_loss(logits, target, source=source)
    residual_error = target.bool() != base
    bce = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    correct = ~residual_error
    edit = (bce * residual_error).sum() / residual_error.sum().clamp_min(1)
    protect = (bce * correct).sum() / correct.sum().clamp_min(1)
    return seg + edit_weight * edit + keep_weight * protect
