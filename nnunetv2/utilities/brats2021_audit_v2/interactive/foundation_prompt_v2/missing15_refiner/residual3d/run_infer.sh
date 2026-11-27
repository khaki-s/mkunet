#!/usr/bin/env bash
# Usage: run_infer.sh SPLIT GPU_CSV [extra infer3d args, e.g. --codes 3 8]
#   SPLIT = validation | test. One shard per GPU; rerun skips finished patients.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
split=${1:?split: validation|test}
gpu_csv=${2:?GPU indices}
shift 2
IFS=, read -ra gpus <<< "$gpu_csv"
stage "infer3d $split on GPUs $gpu_csv"
run_sharded "infer3d_${split}" "${#gpus[@]}" "$gpu_csv" infer3d \
    --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" --split "$split" --device cuda:0 "$@"
stage "infer3d $split done: $(ls "$R3D_OUTPUT/r3d_scores/$split"/*.json 2>/dev/null | wc -l) patients"
