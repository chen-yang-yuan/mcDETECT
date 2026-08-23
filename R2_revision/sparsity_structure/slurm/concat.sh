#!/bin/bash
#SBATCH --job-name=a2b_concat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=nodes

# Stitches the per-arm tables and names any arm that did not finish. Runs on `afterany`, so a
# partial sweep still produces readable tables with the gaps listed rather than silently dropped.

set -euo pipefail

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

cd ~/hulab/projects/mcDETECT/R2_revision/sparsity_structure
python3 score_embedding.py --concat
