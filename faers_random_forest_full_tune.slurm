#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --job-name=FAERS_RF
#SBATCH -o /share/home/ycg_luanjingjie/baoan/logs/faers-random-forest-%j.out
#SBATCH -e /share/home/ycg_luanjingjie/baoan/logs/faers-random-forest-%j.err

set -euo pipefail

# ===== 运行 FAERS Random Forest 全量调参代码 =====
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
mkdir -p "${FAERS_ML_OUTPUT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${PROJECT_DIR}"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date '+%F %T')"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(which python)"
python --version
echo "============================================================"

COMMON_ARGS=(
  --feature-version v2
  --period-token 2004_2025
  --cohort all
  --target-col is_fall
  --train-end-year 2023
  --valid-year 2024
  --test-year 2025
  --train-sample-n 0
  --cv-folds 3
  --search-mode full
  --bootstrap-iterations 100
  --random-state 42
)

for feature_set in core enhanced; do
  echo
  echo "------------------------------------------------------------"
  echo "Running Random Forest, feature_set=${feature_set}"
  echo "Start: $(date '+%F %T')"
  echo "------------------------------------------------------------"
  python ml_project/02_random_forest.py "${COMMON_ARGS[@]}" --feature-set "${feature_set}"
  echo "Finished Random Forest, feature_set=${feature_set}: $(date '+%F %T')"
done

echo "All Random Forest jobs finished: $(date '+%F %T')"
