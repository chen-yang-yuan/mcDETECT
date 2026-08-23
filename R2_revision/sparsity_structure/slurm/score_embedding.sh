#!/bin/bash
#SBATCH --job-name=a2b_score
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

# A2b stage 2 -- score one arm (real sample or permuted detection).
# The cost is the k = 2..30 sweep with 5 stability seeds and n_init = 20; the silhouette is
# subsampled (C.SILHOUETTE_SAMPLE_SIZE) because it is O(n^2) in distances.

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/sparsity_structure

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"

python3 score_embedding.py

echo "Job finished at $(date)"
