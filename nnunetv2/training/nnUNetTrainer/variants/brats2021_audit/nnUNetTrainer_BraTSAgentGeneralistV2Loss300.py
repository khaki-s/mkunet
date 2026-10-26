"""Discoverable new trainers. Existing V1/V2 trainer classes are unchanged."""
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.generalist.trainer import BraTSGeneralistTrainer300


class nnUNetTrainer_BraTSAgentGeneralistV2Loss300(BraTSGeneralistTrainer300):
    """Five experts, self-inclusive attention, query-dependent output gate."""
    pass


class nnUNetTrainer_BraTSAgentGeneralistLocalV2Loss300(BraTSGeneralistTrainer300):
    """Matched five-expert control: each receiver can attend only to itself."""
    local_only = True


class nnUNetTrainer_BraTSAgentGeneralistNoComm300(BraTSGeneralistTrainer300):
    """Fifth-expert ablation: direct expert fusion, no residual correction."""
    correction_enabled = False


class nnUNetTrainer_BraTSAgentGeneralistUngatedV2Loss300(BraTSGeneralistTrainer300):
    """Keep self-inclusive attention, disable only the sigmoid output gate."""
    output_gate = False


class nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300(BraTSGeneralistTrainer300):
    """Five experts with the old peer-only, ungated communication equation."""
    output_gate = False
    include_self = False


class nnUNetTrainer_BraTSAgentGeneralistLegacyLocalV2Loss300(BraTSGeneralistTrainer300):
    """Exact old LocalV2 equation with five experts: zero all message inputs."""
    output_gate = False
    include_self = False
    local_only = True
