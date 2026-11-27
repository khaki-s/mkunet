#!/usr/bin/env bash
# Shared environment for every residual3d v2 script. Source it; do not execute it.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../../../.." && pwd)
export BRATS_PROJECT=${BRATS_PROJECT:-/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113}
export R3D_OUTPUT=${R3D_OUTPUT:-$BRATS_PROJECT/sam_protocol/refiner/residual3d_v2}
python_bin=${BRATS_PYTHON:-/database2/chujingyi_env/nninteractive/bin/python}
deps_dir="$(cd "$(dirname "$python_bin")/.." && pwd)/sam3_refiner_deps"
export PYTHONPATH="$repo:$deps_dir" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
# 1 = disable cuDNN (the nninteractive env currently fails with CUDNN_STATUS_NOT_INITIALIZED).
export R3D_DISABLE_CUDNN=${R3D_DISABLE_CUDNN:-1}
module=nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.missing15_refiner.residual3d
mkdir -p "$R3D_OUTPUT/logs"

stage() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

# run_sharded LABEL N_SHARDS GPU_CSV|- MODULE ARGS...   ('-' = CPU-only shards)
run_sharded() {
    local label=$1 shards=$2 gpu_csv=$3 mod=$4
    shift 4
    local gpus=() pids=() i dev status=0
    if [[ $gpu_csv != - ]]; then IFS=, read -ra gpus <<< "$gpu_csv"; fi
    for ((i = 0; i < shards; i++)); do
        dev=""
        if ((${#gpus[@]})); then dev=${gpus[$((i % ${#gpus[@]}))]}; fi
        CUDA_VISIBLE_DEVICES=$dev "$python_bin" -B -u -m "$module.$mod" "$@" \
            --shards "$shards" --shard "$i" >> "$R3D_OUTPUT/logs/${label}_${i}.log" 2>&1 &
        pids+=("$!")
    done
    trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    trap - INT TERM
    return "$status"
}
