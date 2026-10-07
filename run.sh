#!/usr/bin/env bash
# SLURM settings for the RWTH CLAIX cluster (project rwth1954); adapt account,
# partition and GPU request for other sites, or run main.py directly without SLURM.
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

# Usage (submit from the project directory; arguments are passed to main.py):
#   sbatch run.sh                                   # benchmarks/limit from config.yaml
#   sbatch run.sh --run-dir results/run_X --limit all --benchmark label_extraction_arm
#   sbatch run.sh --config configs_local/qwen.yaml --model Qwen/Qwen3-32B
# Resume an interrupted run by submitting again with the same --run-dir.
# Two jobs may share a --run-dir only for different benchmarks (per-benchmark lock).

set -euo pipefail

echo "=== Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date) ==="
echo "Args: $*"

# sbatch runs a copy of this script from the spool directory, so $0 does not point
# to the project; SLURM_SUBMIT_DIR is the directory sbatch was called from.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${PROJECT_DIR}"
if [[ ! -f main.py ]]; then
    echo "ERROR: main.py not found in ${PROJECT_DIR} – submit run.sh from the project directory." >&2
    exit 2
fi
mkdir -p results

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source .venv/bin/activate
python3 -V

# main.py exits 1 if a benchmark failed or is incomplete, 2 on lock or startup configuration errors (health check, fingerprint); a configuration error during a run counts as a stopped benchmark (1)
rc=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun python3 main.py "$@" || rc=$?
else
    python3 main.py "$@" || rc=$?
fi

echo "=== Job finished at $(date) (exit code ${rc}) ==="
exit "${rc}"
