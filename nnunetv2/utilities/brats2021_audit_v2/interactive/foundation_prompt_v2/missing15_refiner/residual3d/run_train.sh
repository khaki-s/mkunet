#!/usr/bin/env bash
# Usage: run_train.sh GPU_CSV [extra train3d args, e.g. --steps 30000 --batch 2]
#   DDP over the listed GPUs; resumes from training3d/checkpoint_last.pth.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gpu_csv=${1:?GPU indices, e.g. 2,4}
shift
IFS=, read -ra gpus <<< "$gpu_csv"
exec 8>"$R3D_OUTPUT/train3d.lock"
flock -n 8 || { echo "Another process owns R3D training"; exit 1; }
stage "train3d on GPUs $gpu_csv"
CUDA_VISIBLE_DEVICES=$gpu_csv "$python_bin" -B -u -m torch.distributed.run --standalone \
    --nproc_per_node="${#gpus[@]}" --module "$module.train3d" \
    --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" "$@" >> "$R3D_OUTPUT/logs/train3d.log" 2>&1
stage "train3d done: $(cat "$R3D_OUTPUT/training3d/completed.json" | head -c 200)"
