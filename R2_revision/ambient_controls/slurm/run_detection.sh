#!/bin/bash
#SBATCH --job-name=a3_detect
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=240:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-5
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

# One array task = one (set x sample) detection. 3 sets x 2 samples = 6 tasks.
# Indices 0-3 = set0 + set3 (cheap); 4-5 = set1 (the only expensive pair).
# A default --array=0-5 is declared so a bare `sbatch slurm/run_detection.sh` works;
# override with `sbatch --array=0-3 ...` to run only the cheap sets first.
#
# Resources copied from code/3_detection.sh: the fine pass over a 103M-transcript section is what
# set1 does, so it needs the same envelope. set0 / set3 are far cheaper (set3 seeds on
# ~2.9M NC transcripts against set1's 33.3M) but share the wrapper -- SLURM only ever grants less
# than the ceiling, and one wrapper is one thing to keep correct.
#
# Reruns: finished tasks are skipped (both spheres.parquet and sphere_dict.parquet must exist), so
# resubmit only the failed ids, e.g.  sbatch --array=2,5 slurm/run_detection.sh

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/ambient_controls
mkdir -p logs      # SLURM opens the log files before the job body runs; submit.sh
                   # is the only other place this happens, so a direct
                   # `sbatch slurm/run_detection.sh` would fail with no log at all.

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID}"
which python
python --version

python3 run_detection_sets.py

echo "Job finished at $(date)"
