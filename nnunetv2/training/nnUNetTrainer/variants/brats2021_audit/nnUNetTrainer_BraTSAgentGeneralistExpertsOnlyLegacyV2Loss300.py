"""Registered J-removal ablation: four small specialists plus one small G."""
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.experts_only.trainer import (
    BraTSFiveSmallExpertsLegacyTrainer300,
)


class nnUNetTrainer_BraTSAgentGeneralistExpertsOnlyLegacyV2Loss300(BraTSFiveSmallExpertsLegacyTrainer300):
    """Five small UNets, LegacyV2 communication/router, no large joint backbone."""
    pass
