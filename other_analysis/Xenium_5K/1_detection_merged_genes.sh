#!/bin/bash
#SBATCH --job-name=Xenium_5K_detection_merged_genes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=240:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

set -euo pipefail

# Load conda + activate env (non-interactive safe)
module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env

# Always run from repo root so relative paths behave
cd ~/hulab/projects/mcDETECT/other_analysis/Xenium_5K/

# (Optional) print debugging info
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "PWD:  $(pwd)"
which python
python --version

# Run your script
python3 1_detection_merged_genes.py

echo "Job finished at $(date)"