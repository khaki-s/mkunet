"""Independent 300-epoch V2 ablations; old trainer names keep their semantics."""
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.trainer import BraTSAgentV2Trainer300


class nnUNetTrainer_BraTSAgentCommV2Loss300(BraTSAgentV2Trainer300):
    """P1: identity residual initialization + direct refined auxiliary supervision."""


class nnUNetTrainer_BraTSAgentLocalV2Loss300(nnUNetTrainer_BraTSAgentCommV2Loss300):
    """Parameter/loss-matched local-only control of P1."""
    local_only = True


class nnUNetTrainer_BraTSAgentInitV2Loss300(nnUNetTrainer_BraTSAgentCommV2Loss300):
    """Initialization-only ablation: original losses, fixed scale/zero head."""
    refined_aux_max = 0.


class nnUNetTrainer_BraTSAgentAuxV2Loss300(nnUNetTrainer_BraTSAgentCommV2Loss300):
    """Auxiliary-only ablation: original trainable zero rho and random last head."""
    residual_mode = 'legacy'


class nnUNetTrainer_BraTSAgentNullV2Loss300(nnUNetTrainer_BraTSAgentCommV2Loss300):
    """Optional P2a: P1 with a zero-valued null peer (including k=2 rejection)."""
    null_peer = True


class nnUNetTrainer_BraTSAgentNullLocalV2Loss300(nnUNetTrainer_BraTSAgentNullV2Loss300):
    """Parameter/loss-matched local-only control of P2a."""
    local_only = True
