#!/bin/bash
# Baysor detection array: one array task = one manifest tile (job_id == array index).
#
# Submit AFTER build_manifest.sh, sizing the array from n_jobs.txt:
#   N=$(cat output/manifests/n_jobs.txt)
#   sbatch --array=0-$((N-1))%50 slurm/run_baysor.sh
# (the --array on the command line overrides the placeholder directive below.)
#
#SBATCH --job-name=bsm_baysor
#SBATCH --output=logs/baysor_%A_%a.out
#SBATCH --error=logs/baysor_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

set -euo pipefail

cd ~/hulab/projects/mcDETECT/R2_revision/baysor_ssam_merscope
mkdir -p logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate baysor_env
export PYTHONNOUSERSITE=1
export BAYSOR_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# If Baysor is not on PATH inside baysor_env, point to it explicitly:
# export BAYSOR_BINARY="$HOME/.julia/bin/baysor"

echo "Host: $(hostname)  Array task: ${SLURM_ARRAY_TASK_ID}  PWD: $(pwd)"
python3 run_baysor_tile.py --task-id "${SLURM_ARRAY_TASK_ID}"
echo "Baysor task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
