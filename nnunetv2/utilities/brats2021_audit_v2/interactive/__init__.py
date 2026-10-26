"""Independent prompt correction for existing automatic BraTS predictions.

Existing segmentation trainers/checkpoints are unchanged. New learned residual
models require offline training; optional nnInteractive adapters load their own
official checkpoints. LLMs are optional action parsers, not spatial ground truth.
"""
from .schemas import REGIONS, ATOMIC_REGIONS, PointPrompt, SpatialROI, StructuredAction
from .session import CorrectionSession

__all__ = [
    "REGIONS", "ATOMIC_REGIONS", "PointPrompt", "SpatialROI",
    "StructuredAction", "CorrectionSession",
]
