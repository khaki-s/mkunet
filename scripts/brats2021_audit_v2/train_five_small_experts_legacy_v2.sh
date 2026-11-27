#!/usr/bin/env bash
# BRATS_GPU=1 ./scripts/brats2021_audit_v2/train_five_small_experts_legacy_v2.sh
# Extra arguments (for example --c or --val --val_best) go to nnU-Net.
set -Eeuo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${BRATS_TRAIN_PY:-/database2/chujingyi_env/lightmunet/bin/python}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/brats_five_experts_mpl}"
export CUDA_VISIBLE_DEVICES="${BRATS_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export LD_LIBRARY_PATH="/database2/chujingyi_env/lightmunet/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export nnUNet_raw="${nnUNet_raw:-/data02/chujingyi/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/data02/chujingyi/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/data02/chujingyi/data/nnUNet_results}"
export BRATS_AUDIT_SEED="${BRATS_AUDIT_SEED:-0}"
cd -- "$REPO_DIR"
exec "$PYTHON_BIN" -u -m nnunetv2.run.run_training 137 3d_fullres 0 \
  -tr nnUNetTrainer_BraTSAgentGeneralistExpertsOnlyLegacyV2Loss300 -p nnUNetPlans "$@"
