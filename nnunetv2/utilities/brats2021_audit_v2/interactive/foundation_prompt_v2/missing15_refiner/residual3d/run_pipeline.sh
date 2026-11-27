#!/usr/bin/env bash
# Usage: run_pipeline.sh GPU_CSV [STEPS [USE_SAM]]
#   GPU_CSV  e.g. 2,4      STEPS  optimizer steps per rank (default 30000)
#   USE_SAM  0 = arms P/R only (default), 1 = also train/score the SAM arm (RS, RS0)
# Every stage is resumable; finished patients/checkpoints are verified and skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gpu_csv=${1:?GPU indices, e.g. 2,4}
steps=${2:-30000}
use_sam=${3:-0}
cpu_shards=${CPU_SHARDS:-16}
IFS=, read -ra gpus <<< "$gpu_csv"
exec 9>"$R3D_OUTPUT/pipeline.lock"
flock -n 9 || { echo "A pipeline already owns $R3D_OUTPUT"; exit 1; }
exec >> "$R3D_OUTPUT/logs/pipeline.log" 2>&1
trap 'printf "%s\n" "$?" > "$R3D_OUTPUT/pipeline.exit"' EXIT

stage "1/8 window cache train320 + val80"
bash "$here/run_prepare.sh" train "$cpu_shards"
bash "$here/run_prepare.sh" validation "$cpu_shards"

stage "2/8 R3D training, $steps steps/rank"
bash "$here/run_train.sh" "$gpu_csv" --steps "$steps"

stage "3/8 R3D inference val80"
bash "$here/run_infer.sh" validation "$gpu_csv"

select_args=()
if [[ $use_sam == 1 ]]; then
    stage "4/8 SAM arm train (K=6) + score val80 (K=10)"
    bash "$here/run_sam.sh" train "${gpus[0]}" --k 6
    bash "$here/run_sam.sh" score validation "$gpu_csv" --k 10
else
    stage "4/8 SAM arm skipped (USE_SAM=0)"
    select_args+=(--no-sam)
fi

stage "5/8 val80 nested selection and freeze"
"$python_bin" -B -u -m "$module.select3d" --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" \
    "${select_args[@]}"
codes=$("$python_bin" -B -m "$module.writeback" codes --project "$BRATS_PROJECT" --output "$R3D_OUTPUT")

if [[ -n $codes ]]; then
    read -ra selected <<< "$codes"
    stage "6/8 test100 export + window cache + inference for enabled masks: $codes"
    export_py=${EXPORT_PYTHON:-/data02/chujingyi/.venvs/brats_foundation_prompt/bin/python}
    pids=()
    for i in "${!gpus[@]}"; do
        (cd /tmp && CUDA_VISIBLE_DEVICES=${gpus[$i]} "$export_py" -I -u \
            "$BRATS_PROJECT/addition_gate/export_missing15.py" --split test --masks "${selected[@]}" \
            --shard-index "$i" --num-shards "${#gpus[@]}" \
            >> "$R3D_OUTPUT/logs/export_test_${gpus[$i]}.log" 2>&1) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    bash "$here/run_prepare.sh" test "$cpu_shards" --codes "${selected[@]}"
    bash "$here/run_infer.sh" test "$gpu_csv" --codes "${selected[@]}"
    if [[ $use_sam == 1 ]]; then
        bash "$here/run_sam.sh" score test "$gpu_csv" --k 10 --codes "${selected[@]}"
    fi
else
    stage "6/8 no unit enabled; test100 equals the official baseline"
fi

stage "7/8 test100 write-back on the official fold_0 baseline"
"$python_bin" -B -u -m "$module.writeback" predict --project "$BRATS_PROJECT" --output "$R3D_OUTPUT"
stage "8/8 test100 paired scoring against fold_0/test100_15_modalities/per_case_metrics.csv"
"$python_bin" -B -u -m "$module.writeback" score --project "$BRATS_PROJECT" --output "$R3D_OUTPUT"
stage "complete"
