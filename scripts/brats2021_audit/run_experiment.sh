#!/usr/bin/env bash
set -euo pipefail

# One requested experiment per invocation. No training is started by code setup.
# Example: bash scripts/brats2021_audit/run_experiment.sh moe
experiment=${1:?Usage: run_experiment.sh baseline|hard|hardloss|loss|lossgate|gate|moe|moe_uniform|moe_no_prior|moe_no_prior_loss|moe_uniform_loss|frequency|mamba|moe_mamba|semantic_moe_loss|semantic_moe_uniform_loss [nnUNet training flags]}
shift
case "$experiment" in
  baseline) trainer=nnUNetTrainer_BraTSAuditBaseline300 ;;
  hard) trainer=nnUNetTrainer_BraTSHard300 ;;
  hardloss) trainer=nnUNetTrainer_BraTSHardLoss300 ;;
  loss) trainer=nnUNetTrainer_BraTSExclusiveLoss300 ;;
  lossgate) trainer=nnUNetTrainer_BraTSModGateLoss300 ;;
  gate) trainer=nnUNetTrainer_BraTSModGate300 ;;
  moe) trainer=nnUNetTrainer_BraTSMoE300 ;;
  moe_uniform) trainer=nnUNetTrainer_BraTSMoEUniform300 ;;
  moe_no_prior) trainer=nnUNetTrainer_BraTSMoENoPrior300 ;;
  moe_no_prior_loss) trainer=nnUNetTrainer_BraTSMoENoPriorLoss300 ;;
  moe_uniform_loss) trainer=nnUNetTrainer_BraTSMoEUniformLoss300 ;;
  frequency) trainer=nnUNetTrainer_BraTSFreqSkip300 ;;
  mamba) trainer=nnUNetTrainer_BraTSMamba300 ;;
  moe_mamba) trainer=nnUNetTrainer_BraTSMoEMamba300 ;;
  semantic_moe_loss) trainer=nnUNetTrainer_BraTSSemanticMoELoss300 ;;
  semantic_moe_uniform_loss) trainer=nnUNetTrainer_BraTSSemanticMoEUniformLoss300 ;;
  *) printf 'Unknown experiment: %s\n' "$experiment" >&2; exit 2 ;;
esac

export PATH="/root/miniconda3/envs/lightmunet/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export nnUNet_raw="${nnUNet_raw:-/root/autodl-tmp/data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/root/autodl-tmp/data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/root/autodl-tmp/data/nnUNet_results}"
export nnUNet_compile=False
export BRATS_AUDIT_SEED="${BRATS_AUDIT_SEED:-0}"
export PYTHONHASHSEED="$BRATS_AUDIT_SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /root/nnUNet

if [[ "$experiment" == hard || "$experiment" == hardloss ]]; then
  python -m nnunetv2.utilities.brats2021_audit.sampling --prepare \
    --root "$nnUNet_preprocessed/Dataset137_BraTS2021" --fold 0
fi

set +e
nnUNetv2_train 137 3d_fullres 0 --npz -tr "$trainer" "$@"
train_status=$?

/usr/bin/shutdown -h now

exit "$train_status"