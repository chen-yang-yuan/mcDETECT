#!/bin/bash
#SBATCH --job-name=a2b_score
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

# A2b stage 2 -- build one COMBINED WT+AD arm, score it, and embed it.
# One task per arm: 1 real + N_PERM permuted (6 by default), NOT per sample -- the embedding the
# paper reports is the combined object from 4_post_detection.ipynb cell 19.
#
# Two costs, in order: the k = 2..30 sweep (5 stability seeds, n_init = 20), which joblib spreads
# across the 16 cpus; then the full-population t-SNE, which sc.tl.tsne hands to sklearn's
# Barnes-Hut with the same n_jobs. t-SNE is Barnes-Hut, so it scales ~n log n and the arms will
# NOT take equal time -- the stage finishes when the largest arm does.

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/sparsity_structure

# joblib memmaps the ~300 MB marker matrix so the 16 workers share one copy instead of pickling
# 16; keep that file on node-local scratch, not on a network filesystem.
export JOBLIB_TEMP_FOLDER="${TMPDIR:-/tmp}"

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"

python3 score_embedding.py

echo "Job finished at $(date)"
