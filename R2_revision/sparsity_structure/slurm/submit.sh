#!/bin/bash
# Submit the whole A2b sweep: permuted detections, then scoring (real + permuted), then concat.
#
#   bash slurm/submit.sh [max_concurrent]        # default 10
#
# Array sizes come from a2_config.py, so changing N_PERM needs no edit here. Scoring waits on
# `afterok` -- it reads the detection output, so there is nothing to score if detection failed.
# Concat waits on `afterany` and lists whatever is missing.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAXC="${1:-10}"

# Counted from a2_config alone -- importing run_permutation_detect here would pull in mcDETECT,
# and this script is meant to run on a login node with no env activated.
N_DETECT=$(python3 -c "import a2_config as C; print(len(C.SAMPLES) * len(C.PERM_SEEDS))")
N_SCORE=$(python3 -c "import a2_config as C; print(len(C.all_arms()))")
echo "detection tasks: ${N_DETECT}   scoring arms: ${N_SCORE}"

DET=$(sbatch --parsable --array=0-$((N_DETECT - 1))%"${MAXC}" slurm/run_permutation.sh)
echo "detection array: ${DET}"

SCORE=$(sbatch --parsable --dependency=afterok:"${DET}" \
        --array=0-$((N_SCORE - 1))%"${MAXC}" slurm/score_embedding.sh)
echo "scoring array:   ${SCORE}"

CONCAT=$(sbatch --parsable --dependency=afterany:"${SCORE}" slurm/concat.sh)
echo "concat job:      ${CONCAT}"
