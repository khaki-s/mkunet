#!/usr/bin/env bash
# Usage: run_prepare.sh SPLIT [N_CPU_SHARDS] [extra prepare3d args, e.g. --codes 3 8]
#   SPLIT = train | validation | test.  CPU only; rerun skips finished patients.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
split=${1:?split: train|validation|test}
shards=${2:-16}
shift $(( $# >= 2 ? 2 : 1 ))
stage "prepare3d $split, $shards CPU shards"
run_sharded "prepare_${split}" "$shards" - prepare3d \
    --project "$BRATS_PROJECT" --output "$R3D_OUTPUT" --split "$split" "$@"
stage "prepare3d $split done: $(ls "$R3D_OUTPUT/windows/$split"/*.json 2>/dev/null | wc -l) patients"
