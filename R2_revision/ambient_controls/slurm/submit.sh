#!/bin/bash
# Submit the A3 detection array.
#
# Array size is derived from a3_config so it can never drift from the task table. This runs on a
# login node WITHOUT activating the env -- a3_config imports nothing but os/pathlib, deliberately.
#
# Usage:  bash slurm/submit.sh [max_concurrent]
#
# Order note: the cheap sets (set0, set3) come first in C.SETS_TO_DETECT, so with a small --array
# slice you can validate the whole path end-to-end before committing set1's 200G x 2 jobs. To do
# that explicitly:
#     sbatch --array=0-3 slurm/run_detection.sh     # set0 + set3, both samples
#     sbatch --array=4-5 slurm/run_detection.sh     # set1, both samples

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

MAXC="${1:-4}"
N=$(python3 -c "import a3_config as C; print(len(C.SETS_TO_DETECT) * len(C.SAMPLES))")
echo "submitting ${N} detection tasks (max ${MAXC} concurrent)"

JOB=$(sbatch --parsable --array=0-$((N - 1))%"${MAXC}" slurm/run_detection.sh)
echo "detection array: ${JOB}"
echo
echo "watch:   squeue -u \$USER"
echo "results: output/detect/<set>_<sample>/{spheres,sphere_dict}.parquet"
