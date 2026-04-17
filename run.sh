#!/usr/bin/env bash
#SBATCH --account=rwth1954
#SBATCH --job-name=med_bench
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err
#SBATCH --partition=c23g
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="

PROJECT_DIR="/rwthfs/rz/cluster/home/rwth1954/Med_Benchmarks_LLMs"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source .venv/bin/activate
python3 -V
srun python3 main.py

echo "=== Job finished at $(date) ==="
