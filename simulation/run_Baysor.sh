#!/bin/bash
#SBATCH --job-name=run_Baysor
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
conda activate baysor_env

# Always run from repo root so relative paths behave
cd ~/hulab/projects/mcDETECT/simulation

# (Optional) print debugging info
echo "Host: $(hostname)"
echo "Job:  $SLURM_JOB_ID"
echo "PWD:  $(pwd)"
which python
python --version

# Prevent interference from ~/.local packages
export PYTHONNOUSERSITE=1

# If you installed Baysor as a Julia package in this env, ~/.julia/bin/baysor
# is usually the CLI location. Uncomment/set the next line if needed:
# export BAYSOR_BINARY="$HOME/.julia/bin/baysor"

# Run the Baysor benchmark
python3 run_Baysor.py

echo "Job finished at $(date)"

