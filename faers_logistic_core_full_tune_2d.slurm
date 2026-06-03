#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --job-name=FAERS_LOGIT_CORE
#SBATCH -o /share/home/ycg_luanjingjie/baoan/logs/faers-logistic-core-%j.out
#SBATCH -e /share/home/ycg_luanjingjie/baoan/logs/faers-logistic-core-%j.err

set -euo pipefail

PROJECT_DIR="/share/home/ycg_luanjingjie/program_FAERS"
CONDA_SH="/share/home/ycg_luanjingjie/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="faers-ml"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export FAERS_SEARCH_N_JOBS=1
export FAERS_SEARCH_BACKEND=""
export FAERS_ESTIMATOR_N_JOBS=4
export FAERS_ML_OUTPUT_ROOT="${PROJECT_DIR}/runs/mainline_2026-05-20/OUTPUT_ML"

mkdir -p /share/home/ycg_luanjingjie/baoan/logs
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date '+%F %T')"
echo "Running Logistic Regression core"

python ml_project/01_logistic_regression.py \
  --feature-version v2 \
  --period-token 2004_2025 \
  --cohort all \
  --target-col is_fall \
  --train-end-year 2023 \
  --valid-year 2024 \
  --test-year 2025 \
  --feature-set core \
  --train-sample-n 0 \
  --cv-folds 3 \
  --search-mode full \
  --bootstrap-iterations 100 \
  --random-state 42

echo "Finished Logistic Regression core: $(date '+%F %T')"
