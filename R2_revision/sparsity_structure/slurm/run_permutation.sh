#!/bin/bash
#SBATCH --job-name=a2b_permutation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=240:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

# A2b stage 1 -- one permuted detection per array task (2 samples x N_PERM seeds).
# Resources match code/3_detection.sh: this runs the same chain on the same section size.
# Finished tasks are skipped, so a failed id can be resubmitted on its own:
#   sbatch --array=<id> slurm/run_permutation.sh

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/sparsity_structure

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
which python
python --version

python3 run_permutation_detect.py

echo "Job finished at $(date)"
