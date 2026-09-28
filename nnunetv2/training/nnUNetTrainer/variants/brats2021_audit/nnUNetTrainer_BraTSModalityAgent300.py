"""Discoverable 300-epoch missing-modality controls and agent experiments."""
from nnunetv2.utilities.brats2021_audit.agent_trainer import BraTSModalityAgentTrainer300


class nnUNetTrainer_BraTSModDrop300(BraTSModalityAgentTrainer300):
    use_experts = False
    exclusive_loss = False


class nnUNetTrainer_BraTSModDropLoss300(nnUNetTrainer_BraTSModDrop300):
    exclusive_loss = True


class nnUNetTrainer_BraTSSemanticMissing300(BraTSModalityAgentTrainer300):
    pass


class nnUNetTrainer_BraTSSemanticMissingUniform300(nnUNetTrainer_BraTSSemanticMissing300):
    uniform_fusion = True


class nnUNetTrainer_BraTSSemanticMissingBeta300(nnUNetTrainer_BraTSSemanticMissing300):
    conditional_beta = True


class nnUNetTrainer_BraTSAgentCommLoss300(nnUNetTrainer_BraTSSemanticMissing300):
    communication = True


class nnUNetTrainer_BraTSAgentLocalLoss300(nnUNetTrainer_BraTSAgentCommLoss300):
    local_only = True


class nnUNetTrainer_BraTSAgentLoMixLoss300(nnUNetTrainer_BraTSAgentCommLoss300):
    lomix = True


class nnUNetTrainer_BraTSSemanticMissingLoMix300(nnUNetTrainer_BraTSSemanticMissing300):
    """Independent LoMix ablation without cross-modality communication."""
    lomix = True
