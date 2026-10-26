"""Discoverable, independent Generalist-capacity ablation; legacy classes stay intact."""
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.generalist_fullsize.trainer import (
    BraTSFullSizeGeneralistLegacyTrainer300,
)


class nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300(BraTSFullSizeGeneralistLegacyTrainer300):
    """LegacyV2 with an independent plans-sized fifth expert and unchanged fusion."""
    pass
