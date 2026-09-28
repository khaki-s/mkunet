"""CLI registration; implementation resides in utilities/brats2021_audit."""
from nnunetv2.utilities.brats2021_audit.trainer import BraTSAuditTrainer300


class nnUNetTrainer_BraTSAuditBaseline300(BraTSAuditTrainer300):
    pass


class nnUNetTrainer_BraTSHard300(BraTSAuditTrainer300):
    class_sampling = True


class nnUNetTrainer_BraTSHardLoss300(nnUNetTrainer_BraTSHard300):
    exclusive_loss = True


class nnUNetTrainer_BraTSExclusiveLoss300(BraTSAuditTrainer300):
    exclusive_loss = True

class nnUNetTrainer_BraTSModGateLoss300(nnUNetTrainer_BraTSExclusiveLoss300):
    mode = 'gate'
    
class nnUNetTrainer_BraTSModGate300(BraTSAuditTrainer300):
    mode = 'gate'


class nnUNetTrainer_BraTSMoE300(BraTSAuditTrainer300):
    mode = 'moe'


class nnUNetTrainer_BraTSMoENoPrior300(BraTSAuditTrainer300):
    mode = 'moe_no_prior'


class nnUNetTrainer_BraTSMoEUniform300(BraTSAuditTrainer300):
    mode = 'moe_uniform'


class nnUNetTrainer_BraTSMoENoPriorLoss300(nnUNetTrainer_BraTSExclusiveLoss300):
    """Learned feature MoE without MRI bias, preserving ExclusiveLoss."""
    mode = 'moe_no_prior'


class nnUNetTrainer_BraTSMoEUniformLoss300(nnUNetTrainer_BraTSExclusiveLoss300):
    """Fixed uniform expert fusion with the identical auxiliary loss."""
    mode = 'moe_uniform'


class nnUNetTrainer_BraTSFreqSkip300(BraTSAuditTrainer300):
    mode = 'frequency'


class nnUNetTrainer_BraTSMamba300(BraTSAuditTrainer300):
    mode = 'mamba'


class nnUNetTrainer_BraTSMoEMamba300(BraTSAuditTrainer300):
    """Combination for AFTER separate MoE and Mamba ablations."""
    mode = 'moe_mamba'


from nnunetv2.utilities.brats2021_audit.semantic_trainer import BraTSSemanticMoETrainer300


class nnUNetTrainer_BraTSSemanticMoELoss300(BraTSSemanticMoETrainer300):
    """ExclusiveLoss + supervised modality experts + learned region routing."""
    pass


class nnUNetTrainer_BraTSSemanticMoEUniformLoss300(BraTSSemanticMoETrainer300):
    """Identical experts, losses and warmup; fixed 1/4 expert routing."""
    mode = 'semantic_moe_uniform'
    uniform_fusion = True
