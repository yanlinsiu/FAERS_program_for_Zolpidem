#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --partition=cpu
#SBATCH --job-name=FAERS_TUNE
#SBATCH --array=0-5%4
#SBATCH -o /share/home/ycg_luanjingjie/baoan/logs/faers-tune-%A_%a.out
#SBATCH -e /share/home/ycg_luanjingjie/baoan/logs/faers-tune-%A_%a.err

set -euo pipefail

PROJECT_DIR="/share/home/ycg_luanjingjie/program_FAERS"
CONDA_SH="/share/home/ycg_luanjingjie/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="faers-ml"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export FAERS_ML_OUTPUT_ROOT="${PROJECT_DIR}/runs/mainline_2026-05-20/OUTPUT_ML"

mkdir -p /share/home/ycg_luanjingjie/baoan/logs
mkdir -p "${FAERS_ML_OUTPUT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

TASKS=(
  "ml_project/01_logistic_regression.py core"
  "ml_project/01_logistic_regression.py enhanced"
  "ml_project/02_random_forest.py core"
  "ml_project/02_random_forest.py enhanced"
  "ml_project/03_xgboost.py core"
  "ml_project/03_xgboost.py enhanced"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
SCRIPT="$(echo "${TASK}" | awk '{print $1}')"
FEATURE_SET="$(echo "${TASK}" | awk '{print $2}')"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date '+%F %T')"
echo "Project dir: ${PROJECT_DIR}"
echo "Script: ${SCRIPT}"
echo "Feature set: ${FEATURE_SET}"
echo "Python: $(which python)"
python --version
echo "============================================================"

python - <<'PY'
import importlib

required = ["pandas", "pyarrow", "sklearn", "xgboost", "numpy", "scipy", "statsmodels"]
missing = []
for name in required:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if missing:
    print("Missing Python packages:")
    for item in missing:
        print(f"  - {item}")
    raise SystemExit(1)
PY

python "${SCRIPT}" \
  --feature-version v2 \
  --period-token 2004_2025 \
  --cohort all \
  --target-col is_fall \
  --train-end-year 2023 \
  --valid-year 2024 \
  --test-year 2025 \
  --feature-set "${FEATURE_SET}" \
  --train-sample-n 0 \
  --cv-folds 3 \
  --search-mode full \
  --bootstrap-iterations 100 \
  --random-state 42

echo "============================================================"
echo "Finished time: $(date '+%F %T')"
echo "Output root: ${FAERS_ML_OUTPUT_ROOT}"
echo "============================================================"
