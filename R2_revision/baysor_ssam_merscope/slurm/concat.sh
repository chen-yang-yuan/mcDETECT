#!/bin/bash
# Concatenate per-tile detections into one sphere table per config.
# Run after BOTH detection arrays finish.
#
#   sbatch slurm/concat.sh
#
#SBATCH --job-name=bsm_concat
#SBATCH --output=logs/concat_%j.out
#SBATCH --error=logs/concat_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

set -euo pipefail

cd ~/hulab/projects/mcDETECT/R2_revision/baysor_ssam_merscope
mkdir -p logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate mcDETECT-env
export PYTHONNOUSERSITE=1

echo "Host: $(hostname)  PWD: $(pwd)"
python3 concat_spheres.py --method both
echo "Concatenation finished at $(date)"
