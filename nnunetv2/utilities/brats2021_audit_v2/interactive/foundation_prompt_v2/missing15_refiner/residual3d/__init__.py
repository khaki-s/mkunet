"""v2: 3D residual corrector (R3D) + optional SAM3 evidence arm for BraTS missing-modality refinement.

Input: available MRI, availability, baseline WT/TC/ET probability, baseline one-hot.
Output: four-class logits = log(nested baseline prior) + zero-initialised residual.
See 15a_三维残差精修v2_代码结构说明_20261126.md for the module map.

R3D_DISABLE_CUDNN=1 turns cuDNN off for every v2 entry point (workaround when the
installed cuDNN does not match the torch build; slower but numerically equivalent).
"""
import os as _os

if _os.environ.get("R3D_DISABLE_CUDNN") == "1":
    import torch as _torch
    _torch.backends.cudnn.enabled = False

__version__ = "2.0.0"
