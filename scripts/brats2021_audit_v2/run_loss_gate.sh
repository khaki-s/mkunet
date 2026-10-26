#!/usr/bin/env bash
set -euo pipefail

# Independent launcher; preserves the user's existing run_experiment.sh.
# Optional successful-run shutdown: BRATS_SHUTDOWN_ON_SUCCESS=1 bash this_file
export PATH="/root/miniconda3/envs/lightmunet/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export nnUNet_raw="${nnUNet_raw:-/root/autodl-tmp/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/root/autodl-tmp/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/root/autodl-tmp/data/nnUNet_results}"
export nnUNet_compile=False
export BRATS_AUDIT_SEED="${BRATS_AUDIT_SEED:-0}"
cd /root/nnUNet

# Advisory allocation margin, not a guarantee of exact training disk usage.
python - <<'PY'
import os
import shutil
from pathlib import Path
p = Path(os.environ['nnUNet_results'])
while not p.exists():
    p = p.parent
free_gib = shutil.disk_usage(p).free / 1024**3
if free_gib < 3:
    raise SystemExit(f'Only {free_gib:.2f} GiB free on results filesystem; need a 3 GiB starting margin.')
print(f'Results filesystem free: {free_gib:.2f} GiB. Training starts only after this check.', flush=True)
PY

trainer=nnUNetTrainer_BraTSModGateLoss200
if nnUNetv2_train 137 3d_fullres 0 --npz -tr "$trainer" "$@"; then
  if [[ "${BRATS_SHUTDOWN_ON_SUCCESS:-0}" == 1 ]]; then
    /usr/bin/shutdown -h now
  fi
else
  train_status=$?
  printf 'Training/validation failed (exit %s); machine left running for inspection.\n' "$train_status" >&2
  exit "$train_status"
fi
