#!/bin/bash
# Convenience launcher: size both detection arrays from the manifest and submit
# baysor + ssam in parallel, then concat with an afterok dependency on both.
#
# Prereq: build_manifest.sh has already produced manifests/n_jobs.txt.
#
#   bash slurm/submit_arrays.sh            # concurrency %50 (default)
#   bash slurm/submit_arrays.sh 30         # concurrency %30
#
# Run from a login node (this only submits jobs; it does no compute).

set -euo pipefail

cd ~/hulab/projects/mcDETECT/other_analysis/baysor_ssam_merscope
mkdir -p logs

CONCURRENCY="${1:-50}"

N_JOBS_FILE="$(python3 -c 'import config; print(config.N_JOBS_PATH)')"
if [[ ! -f "$N_JOBS_FILE" ]]; then
    echo "n_jobs.txt not found ($N_JOBS_FILE). Run build_manifest.sh first." >&2
    exit 1
fi
N=$(cat "$N_JOBS_FILE")
LAST=$((N - 1))
echo "Manifest has $N jobs -> array 0-${LAST}%${CONCURRENCY}"

BAYSOR_ID=$(sbatch --parsable --array=0-${LAST}%${CONCURRENCY} slurm/run_baysor.sh)
echo "Submitted Baysor array: $BAYSOR_ID"

SSAM_ID=$(sbatch --parsable --array=0-${LAST}%${CONCURRENCY} slurm/run_ssam.sh)
echo "Submitted SSAM array:   $SSAM_ID"

CONCAT_ID=$(sbatch --parsable \
    --dependency=afterok:${BAYSOR_ID}:${SSAM_ID} \
    slurm/concat.sh)
echo "Submitted concat (afterok both): $CONCAT_ID"
