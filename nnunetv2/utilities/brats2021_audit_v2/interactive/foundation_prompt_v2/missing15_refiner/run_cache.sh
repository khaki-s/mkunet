#!/usr/bin/env bash
# Usage: run_cache.sh GPU_CSV OUTPUT SPLIT [cache options]
set -euo pipefail

gpu_csv=${1:?GPU indices required}
output=${2:?Run directory required}
split=${3:?Split required}
shift 3
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../.." && pwd)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export PYTHONNOUSERSITE=1
python_bin=${BRATS_PYTHON:-/database2/chujingyi_env/nninteractive/bin/python}
deps_dir="$(cd "$(dirname "$python_bin")/.." && pwd)/sam3_refiner_deps"
export PYTHONPATH="$repo:$deps_dir"
module=nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.missing15_refiner.cache_features

IFS=, read -ra gpus <<< "$gpu_csv"
mkdir -p "$output/logs"
exec 8>"$output/cache_${split}.lock"
flock 8
"$python_bin" -B -m "$module" --output "$output" --initialize "$@"
pids=()
for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$python_bin" -B -u -m "$module" \
        --output "$output" --split "$split" --shards "${#gpus[@]}" --shard "$i" "$@" \
        >> "$output/logs/cache_${split}_${gpus[$i]}.log" 2>&1 &
    pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
