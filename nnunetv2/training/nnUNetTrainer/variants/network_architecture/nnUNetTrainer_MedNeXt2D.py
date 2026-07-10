from __future__ import annotations

import pydoc
from typing import Tuple

import torch
import torch.nn as nn
from torch._dynamo import OptimizedModule

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.mednext.mednext_unet import MedNeXtUNet_nnUNet
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager, PlansManager,
)


class nnUNetTrainer_MedNeXt2D(nnUNetTrainer):
    EXP_R              = [2, 2, 2, 3, 3]   # per-stage encoder expansion factor (t)
    EXP_R_DECODER      = [3, 3, 2, 2]      # decoder (deepest-first), symmetric
    GRN: bool          = False             # legacy; multi-kernel block has no GRN
    USE_CHECKPOINT: bool = True
    # Largest depthwise kernel; the block expands it to the odd multi-kernel
    # set, e.g. 5 -> {1, 3, 5}.
    DW_KERNEL_SIZE: int = 5
    LAYER_SCALE_INIT: float = 1e-6         # ConvNeXt LayerScale gamma
    MAX_DROP_PATH_RATE: float = 0.05        # linear stochastic-depth schedule
    # ---- skip-connection attention knobs ----
    # PFESA (parameter-free edge-structure attention) + GroupedAttentionGate
    # (MKUNet GAG) applied to each decoder skip, in that order.
    USE_PFESA: bool = True
    USE_GAG: bool = True
    PFESA_BASE_RATIO: float = 0.1
    GAG_KERNEL: int = 3                     # grouped attention-gate kernel

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # MedNeXt does not train well with SGD@1e-2. Use the paper recipe.
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
        for ri in configuration_manager.network_arch_init_kwargs_req_import:
            if arch_kwargs[ri] is not None:
                arch_kwargs[ri] = pydoc.locate(arch_kwargs[ri])

        return MedNeXtUNet_nnUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            exp_r=cls.EXP_R,
            exp_r_decoder=cls.EXP_R_DECODER,
            grn=cls.GRN,
            use_checkpoint=cls.USE_CHECKPOINT,
            dw_kernel_size=cls.DW_KERNEL_SIZE,
            layer_scale_init=cls.LAYER_SCALE_INIT,
            max_drop_path_rate=cls.MAX_DROP_PATH_RATE,
            use_pfesa=cls.USE_PFESA,
            use_gag=cls.USE_GAG,
            pfesa_base_ratio=cls.PFESA_BASE_RATIO,
            gag_kernel=cls.GAG_KERNEL,
            **arch_kwargs,
        )

    # Defensive override: the base trainer assumes the network exposes
    # a real `.decoder` with a `.deep_supervision` attribute. Our model
    # has a tiny proxy that does exactly that, so the base call works,
    # but we override here too in case torch.compile rewraps things.
    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        if hasattr(mod, 'decoder') and hasattr(mod.decoder, 'deep_supervision'):
            mod.decoder.deep_supervision = enabled
        else:
            mod.deep_supervision = enabled

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, object]:
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4,
            betas=(0.9, 0.999),
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainer_MedNeXt2D_100epochs(nnUNetTrainer_MedNeXt2D):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_MedNeXt3D(nnUNetTrainer_MedNeXt2D):
    """MedNeXt-block U-Net with conservative 3D block settings."""

    EXP_R              = [2, 2, 2, 2, 2]
    EXP_R_DECODER      = [2, 2, 2, 2]
    GRN: bool          = False
    USE_CHECKPOINT: bool = True
    DW_KERNEL_SIZE: int = 3
    LAYER_SCALE_INIT: float = 1e-6
    MAX_DROP_PATH_RATE: float = 0.03

class nnUNetTrainer_MedNeXt3D_100epochs(nnUNetTrainer_MedNeXt3D):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
