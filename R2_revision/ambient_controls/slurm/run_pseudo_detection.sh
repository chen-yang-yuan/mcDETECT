#!/bin/bash
#SBATCH --job-name=a3e_detect
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=240:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

# A3e stage 2: one array task = one sample's re-detection over the relabelled transcript table.
# 2 tasks (WT, AD).
#
# Resources copied from slurm/run_detection.sh, which copied them from code/3_detection.sh. This
# is the same 20-marker pass over the same 103M-transcript section that Set 1 ran, plus an
# nc_filter, so the envelope is the same. Set 1's recorded wall times were 226 min (WT) and 96 min
# (AD); expect the same order.
#
# PREREQUISITE, and this script fails loudly rather than quietly if it is missing:
# output/a3e/a3e_relabel_<sample>.parquet and a3e_relabel_scope.csv, built locally by
# A3e_pseudo_granules.ipynb sections 1-4 and copied here. Nothing is constructed on the node --
# the node only applies a patch that was already decided, which is what makes the run auditable.
#
# Reruns: finished tasks are skipped (both spheres.parquet and sphere_dict.parquet must exist), so
# resubmit only the failed ids, e.g.  sbatch --array=1 slurm/run_pseudo_detection.sh

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/ambient_controls
mkdir -p logs      # SLURM opens the log files before the job body runs

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}  Array task: ${SLURM_ARRAY_TASK_ID}"
which python
python --version

python3 run_pseudo_detection.py

echo "Job finished at $(date)"
