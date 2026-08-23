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
# It now stops at the profile -- the per-sample normalise/PCA/t-SNE was removed, since the
# embedding that matters is the combined WT+AD one built in score_embedding.py. That removed the
# slowest step (sc.tl.tsne falls back to single-threaded sklearn here), so expect a shorter run
# than 3_detection.py. Detection itself is serial within a task: sklearn's DBSCAN takes no
# n_jobs, dbscan() loops the 20 markers, and merge_sphere() is a Python row loop -- the log
# prints the sphere count entering the merge so a blow-up is visible early.
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
