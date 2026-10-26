#!/usr/bin/env bash
# Explicitly invoke train/resume/test; this script is not run by the implementation task.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash run.sh {comm|local|init|aux|null|null-local|moddrop-loss|local-v1} GPU_ID {train|resume|test|print}' >&2
  exit 2
fi
variant=$1
gpu=$2
action=${3:-print}
[[ $gpu =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one integer.' >&2; exit 2; }
v2_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
code_repo=$v2_repo
case $variant in
  comm) trainer=nnUNetTrainer_BraTSAgentCommV2Loss300 ;;
  local) trainer=nnUNetTrainer_BraTSAgentLocalV2Loss300 ;;
  init) trainer=nnUNetTrainer_BraTSAgentInitV2Loss300 ;;
  aux) trainer=nnUNetTrainer_BraTSAgentAuxV2Loss300 ;;
  null) trainer=nnUNetTrainer_BraTSAgentNullV2Loss300 ;;
  null-local) trainer=nnUNetTrainer_BraTSAgentNullLocalV2Loss300 ;;
  moddrop-loss) trainer=nnUNetTrainer_BraTSModDropLoss300; code_repo=/data02/chujingyi/nnUNet ;;
  local-v1) trainer=nnUNetTrainer_BraTSAgentLocalLoss300; code_repo=/data02/chujingyi/nnUNet ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
export BRATS_AUDIT_SEED=${BRATS_AUDIT_SEED:-0}
[[ $BRATS_AUDIT_SEED =~ ^[0-9]+$ ]] || { echo 'Seed must be a nonnegative integer.' >&2; exit 2; }
# Separate seeds cannot accidentally overwrite the existing seed-0 experiment.
if [[ $BRATS_AUDIT_SEED != 0 && -z ${nnUNet_results:-} ]]; then
  export nnUNet_results=/data02/chujingyi/data/nnUNet_results_agent_v2_seed${BRATS_AUDIT_SEED}
fi
export nnUNet_raw=${nnUNet_raw:-/data02/chujingyi/data/nnUNet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-/data02/chujingyi/data/nnUNet_preprocessed}
export nnUNet_results=${nnUNet_results:-/data02/chujingyi/data/nnUNet_results}
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-12}
export BRATS_DIAG_EVERY=${BRATS_DIAG_EVERY:-50}
export BRATS_ABLATE_EVERY=${BRATS_ABLATE_EVERY:-25}
export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONPATH=$code_repo${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=/database2/chujingyi_env/lightmunet/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python_bin=${BRATS_PYTHON:-/database2/chujingyi_env/lightmunet/bin/python}
if [[ -n ${BRATS_AGENT_INITIAL_WEIGHTS:-} ]]; then
  echo 'Unset BRATS_AGENT_INITIAL_WEIGHTS: these are from-scratch controlled ablations.' >&2
  exit 2
fi
fold=$nnUNet_results/Dataset137_BraTS2021/${trainer}__nnUNetPlans__3d_fullres/fold_0
command=("$python_bin" -m nnunetv2.run.run_training 137 3d_fullres 0 -tr "$trainer")
case $action in
  train)
    if [[ -e $fold ]]; then
      echo "Result folder already exists: $fold. Inspect it and use resume for an interrupted run." >&2
      exit 2
    fi ;;
  resume)
    if [[ ! -f $fold/checkpoint_latest.pth && ! -f $fold/checkpoint_final.pth && ! -f $fold/checkpoint_best.pth ]]; then
      echo "No checkpoint found in $fold" >&2; exit 2
    fi
    command+=(--c) ;;
  test)
    [[ -f $fold/checkpoint_best.pth ]] || { echo "Missing $fold/checkpoint_best.pth" >&2; exit 2; }
    command=("$python_bin" "$code_repo/scripts/brats2021_audit/evaluate_15_modalities.py"
      --dataset 137 --trainer "$trainer" --fold 0 --split test
      --checkpoint checkpoint_best.pth --masks all15
      --manifest /data02/chujingyi/brats_dataset_audit/test100_four_modalities_manifest.json
      --gt "$nnUNet_raw/Dataset137_BraTS2021/labelsTs"
      --output "$fold/test100_15_modalities") ;;
  print) ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
printf 'Code: %s\nTrainer: %s\nSeed: %s\nResult: %s\n' "$code_repo" "$trainer" "$BRATS_AUDIT_SEED" "$fold"
printf '%q ' "${command[@]}"
printf '\n'
[[ $action == print ]] && exit 0
cd -- "$code_repo"
exec "${command[@]}"
