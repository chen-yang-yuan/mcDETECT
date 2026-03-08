#!/bin/bash
# Run Baysor in parallel on HGCC (multi-marker data only, 200 seeds = 20 blocks of 10).
#
# Usage:
#   ./run_Baysor.sh              # submit 20 block jobs (blocks 1–20, 10 seeds each)
#   sbatch run_Baysor.sh 5       # run block 5 (seeds 41–50) in one job
#   sbatch run_Baysor.sh eval    # run evaluation only (after all blocks have finished)
#
# Each block job runs: python run_Baysor.py --block BLOCK
# After all blocks finish, run the eval job to compute metrics and write result CSVs:
#   sbatch run_Baysor.sh eval

set -euo pipefail

# Always run from the simulation directory in your project (match run_SSAM.sh)
cd ~/hulab/projects/mcDETECT/simulation

# ---------- Single-job modes ----------

# One arg: BLOCK (1–20) or "eval"
if [[ $# -eq 1 ]]; then
    ARG="$1"
    if [[ "$ARG" == "eval" ]]; then
        module purge
        module load miniconda3
        eval "$(conda shell.bash hook)"
        conda activate baysor_env
        export PYTHONNOUSERSITE=1
        echo "Host: $(hostname)  Eval-only run  PWD: $(pwd)"
        python3 run_Baysor.py
        echo "Evaluation finished at $(date)"
        exit 0
    fi
    if ! [[ "$ARG" =~ ^[0-9]+$ ]] || (( ARG < 1 || ARG > 20 )); then
        echo "Usage: sbatch run_Baysor.sh [1-20|eval]  (block index or eval)" >&2
        exit 1
    fi
    BLOCK="$ARG"
    module purge
    module load miniconda3
    eval "$(conda shell.bash hook)"
    conda activate baysor_env
    export PYTHONNOUSERSITE=1
    # export BAYSOR_BINARY="$HOME/.julia/bin/baysor"  # uncomment if needed
    echo "Host: $(hostname)  Block: $BLOCK  PWD: $(pwd)"
    python3 run_Baysor.py --block "$BLOCK"
    echo "Block $BLOCK finished at $(date)"
    exit 0
fi

# ---------- No args: submit 20 block jobs ----------

for BLOCK in $(seq 1 20); do
    sbatch --job-name="Baysor_b${BLOCK}" \
           --output="logs/Baysor_block${BLOCK}_%j.out" \
           --error="logs/Baysor_block${BLOCK}_%j.err" \
           --time=240:00:00 \
           --mem=50G \
           --cpus-per-task=16 \
           --partition=nodes \
           --mail-type=END,FAIL \
           --mail-user=cyuan36@emory.edu \
           "$0" "$BLOCK"
done

echo "Submitted 20 block jobs (multi_marker 3D all, 10 seeds each). After they finish, run:"
echo "  sbatch run_Baysor.sh eval"