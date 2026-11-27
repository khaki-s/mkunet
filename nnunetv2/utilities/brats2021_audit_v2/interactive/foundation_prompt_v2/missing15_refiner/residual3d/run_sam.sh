#!/usr/bin/env bash
# Usage: run_sam.sh train GPU [extra sam_branch args]
#        run_sam.sh score SPLIT GPU_CSV [extra sam_branch args, e.g. --codes 3 8]
#   SAM features come from the v2 cache, then the v1 cache (sam3_qv_priority_v1),
#   and only the remainder is encoded. Optional arm: skip this script to run P/R only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
stage_name=${1:?train|score}
shift
if [[ $stage_name == train ]]; then
    gpu=${1:?one GPU index}
    shift
    stage "sam_branch train on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$python_bin" -B -u -m "$module.sam_branch" train \
        --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" --device cuda:0 "$@" \
        >> "$R3D_OUTPUT/logs/sam_train.log" 2>&1
else
    split=${1:?validation|test}
    gpu_csv=${2:?GPU indices}
    shift 2
    IFS=, read -ra gpus <<< "$gpu_csv"
    stage "sam_branch score $split on GPUs $gpu_csv"
    run_sharded "sam_score_${split}" "${#gpus[@]}" "$gpu_csv" sam_branch score \
        --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" --split "$split" --device cuda:0 "$@"
fi
stage "sam_branch $stage_name done"
