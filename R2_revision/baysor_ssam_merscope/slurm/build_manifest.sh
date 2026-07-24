#!/bin/bash
# Build the tile job manifest (reads real MERSCOPE data; run once before the arrays).
# Fast + light: can be run directly on a login node, or submitted with sbatch.
#
#   bash slurm/build_manifest.sh        # run here
#   sbatch slurm/build_manifest.sh      # or submit
#
#SBATCH --job-name=bsm_manifest
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

set -euo pipefail

# Scripts directory on HGCC (adjust if your checkout lives elsewhere).
cd ~/hulab/projects/mcDETECT/R2_revision/baysor_ssam_merscope
mkdir -p logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env
export PYTHONNOUSERSITE=1

echo "Host: $(hostname)  PWD: $(pwd)"
python3 build_manifest.py
echo "Manifest built at $(date). N jobs:"
cat "$(python3 -c 'import config; print(config.N_JOBS_PATH)')"
