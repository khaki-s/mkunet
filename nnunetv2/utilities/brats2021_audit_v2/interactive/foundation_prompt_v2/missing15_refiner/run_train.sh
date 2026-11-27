#!/usr/bin/env bash
# Usage: run_train.sh GPU_CSV OUTPUT [train options]
set -euo pipefail

gpu_csv=${1:?GPU indices required}
output=${2:?Run directory required}
shift 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$gpu_csv"
python_bin=${BRATS_PYTHON:-/database2/chujingyi_env/nninteractive/bin/python}
deps_dir="$(cd "$(dirname "$python_bin")/.." && pwd)/sam3_refiner_deps"
export PYTHONPATH="$repo:$deps_dir"
module=nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.missing15_refiner.train
IFS=, read -ra gpus <<< "$gpu_csv"
mkdir -p "$output/logs"
exec 8>"$output/train.lock"
flock -n 8 || { echo "Another worker group owns training"; exit 1; }
exec "$python_bin" -B -u -m torch.distributed.run --standalone --nproc_per_node="${#gpus[@]}" \
    --module "$module" --output "$output" "$@" >> "$output/logs/train.log" 2>&1
