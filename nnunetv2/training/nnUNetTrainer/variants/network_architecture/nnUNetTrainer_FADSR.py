import pydoc

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.fadsr_unet import FADSR_PlainConvUNet


class nnUNetTrainer_FADSR(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
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
        return FADSR_PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs
        )


class nnUNetTrainer_FADSR_99epochs(nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 99


class nnUNetTrainer_FADSR_100epochs(nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100