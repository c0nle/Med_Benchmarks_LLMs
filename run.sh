#!/usr/bin/env bash
#SBATCH --account=rwth1954
#SBATCH --job-name=med_bench
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --partition=c23g
# The project may only use c23g, which requires >=1 GPU. Inference itself runs
# remotely on the LLM server, so the GPU stays idle.
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Usage (arguments are passed to main.py):
#   sbatch run.sh                                   # benchmarks/limit from config.yaml
#   sbatch run.sh --run-dir results/run_X --limit all --benchmark label_extraction_arm
# Resume an interrupted run by submitting again with the same --run-dir.

set -euo pipefail

echo "=== Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date) ==="
echo "Args: $*"

PROJECT_DIR="/rwthfs/rz/cluster/home/rwth1954/Med_Benchmarks_LLMs"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source .venv/bin/activate
python3 -V
srun python3 main.py "$@"

echo "=== Job finished at $(date) ==="
