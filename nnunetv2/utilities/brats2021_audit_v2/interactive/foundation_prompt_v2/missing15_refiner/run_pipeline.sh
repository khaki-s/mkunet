#!/usr/bin/env bash
# Usage: run_pipeline.sh GPU_CSV OUTPUT [EPOCHS [TRAIN_SLICES [EVAL_SLICES [priority|all]]]]
set -euo pipefail

gpu_csv=${1:?GPU indices required}
output=${2:?Run directory required}
epochs=${3:-2}
train_slices=${4:-3}
eval_slices=${5:-5}
scope=${6:-priority}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../../../../../.." && pwd)
project=${BRATS_PROJECT:-/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113}
python_bin=${BRATS_PYTHON:-/database2/chujingyi_env/nninteractive/bin/python}
deps_dir="$(cd "$(dirname "$python_bin")/.." && pwd)/sam3_refiner_deps"
export PYTHONPATH="$repo:$deps_dir" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
module=nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.missing15_refiner
IFS=, read -ra gpus <<< "$gpu_csv"

mkdir -p "$output/logs"
exec 9>"$output/pipeline.lock"
flock -n 9 || { echo "A pipeline already owns $output"; exit 1; }
exec >> "$output/logs/pipeline.log" 2>&1

stage() {
    printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*"
}

parallel_gpu() {
    local label=$1
    shift
    local pids=() i status=0
    for i in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$python_bin" -B -u "$@" \
            --num-shards "${#gpus[@]}" --shard-index "$i" \
            >> "$output/logs/${label}_${gpus[$i]}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

infer() {
    local split=$1
    local pids=() i status=0
    for i in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$python_bin" -B -u -m "$module.infer" \
            --output "$output" --project "$project" --split "$split" --max-slices "$eval_slices" --scope "$scope" \
            --shards "${#gpus[@]}" --shard "$i" >> "$output/logs/infer_${split}_${gpus[$i]}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    return "$status"
}

trap 'status=$?; printf "%s\n" "$status" > "$output/pipeline.exit"' EXIT
if [[ -n ${BRATS_WAIT_FOR_CACHE_PID:-} ]]; then
    stage "waiting for existing train320 cache process $BRATS_WAIT_FOR_CACHE_PID"
    while kill -0 "$BRATS_WAIT_FOR_CACHE_PID" 2>/dev/null; do
        sleep 15
    done
fi
stage "train320 feature cache"
bash "$here/run_cache.sh" "$gpu_csv" "$output" train --project "$project" --max-slices "$train_slices" --scope all
stage "train320 DDP LoRA, $epochs epochs"
bash "$here/run_train.sh" "$gpu_csv" "$output" --project "$project" --max-slices "$train_slices" --epochs "$epochs"
stage "val80 feature cache"
bash "$here/run_cache.sh" "$gpu_csv" "$output" validation --project "$project" --max-slices "$eval_slices" --scope "$scope"
stage "val80 decoder inference"
infer validation
stage "val80 nested calibration and freeze"
"$python_bin" -B -u -m "$module.select_val80" --output "$output" --project "$project" \
    --max-slices "$eval_slices" --scope "$scope"
codes=$("$python_bin" -B -m "$module.apply_test" codes --output "$output" --project "$project")
if [[ -n $codes ]]; then
    read -ra selected <<< "$codes"
    stage "test100 baseline export, enabled masks $codes"
    parallel_gpu export_test "$project/addition_gate/export_missing15.py" --split test --masks "${selected[@]}"
    stage "test100 feature cache"
    bash "$here/run_cache.sh" "$gpu_csv" "$output" test --project "$project" --max-slices "$eval_slices" \
        --scope "$scope" --codes "${selected[@]}"
    stage "test100 decoder inference"
    infer test
fi
stage "test100 frozen application"
"$python_bin" -B -u -m "$module.apply_test" predict --output "$output" --project "$project"
stage "test100 paired scoring"
"$python_bin" -B -u -m "$module.apply_test" score --output "$output" --project "$project"
stage "complete"
