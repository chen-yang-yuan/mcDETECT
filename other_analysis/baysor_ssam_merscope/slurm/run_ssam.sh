#!/bin/bash
# SSAM detection array: one array task = one manifest tile (job_id == array index).
#
# Submit AFTER build_manifest.sh, sizing the array from n_jobs.txt:
#   N=$(cat output/MERSCOPE_WT_AD_comparison/baysor_ssam/manifests/n_jobs.txt)
#   sbatch --array=0-$((N-1))%50 slurm/run_ssam.sh
# (the --array on the command line overrides the placeholder directive below.)
#
#SBATCH --job-name=bsm_ssam
#SBATCH --output=logs/ssam_%A_%a.out
#SBATCH --error=logs/ssam_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=nodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuan36@emory.edu

set -euo pipefail

cd ~/hulab/projects/mcDETECT/other_analysis/baysor_ssam_merscope
mkdir -p logs

module purge
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate ssam_hpc
export PYTHONNOUSERSITE=1

echo "Host: $(hostname)  Array task: ${SLURM_ARRAY_TASK_ID}  PWD: $(pwd)"
python3 run_ssam_tile.py --task-id "${SLURM_ARRAY_TASK_ID}"
echo "SSAM task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
