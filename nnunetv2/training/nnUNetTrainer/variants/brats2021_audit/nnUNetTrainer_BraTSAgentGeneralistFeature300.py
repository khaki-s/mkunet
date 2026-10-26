"""Matched feature-injection ablations; keep existing trained classes intact."""
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.feature_injection.trainer import (
    BraTSGeneralistFeatureTrainer300,
)


class nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300(BraTSGeneralistFeatureTrainer300):
    """LegacyLocal5 plus expert-to-joint spatial cross-attention at epoch 50."""
    local_only = True


class nnUNetTrainer_BraTSAgentGeneralistFeatureComm300(BraTSGeneralistFeatureTrainer300):
    """LegacyComm5 plus the same expert-to-joint spatial cross-attention."""
    pass
