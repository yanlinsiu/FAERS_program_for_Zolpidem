#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --time=7-00:00:00
#SBATCH --job-name=SERIOUS_ML
#SBATCH --array=0-11
#SBATCH -o /share/home/ycg_luanjingjie/baoan/logs/serious-ml-%A_%a.out
#SBATCH -e /share/home/ycg_luanjingjie/baoan/logs/serious-ml-%A_%a.err

set -euo pipefail

PROJECT_DIR="/share/home/ycg_luanjingjie/program_FAERS/sedative_hypnotic_fall_credibility"
CONDA_SH="/share/home/ycg_luanjingjie/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="faers-ml"
LOG_DIR="/share/home/ycg_luanjingjie/baoan/logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${LOG_DIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

TASKS=(
  "logistic_regression serious_any"
  "logistic_regression serious_hospitalization"
  "logistic_regression serious_death"
  "logistic_regression serious_life_threatening"
  "random_forest serious_any"
  "random_forest serious_hospitalization"
  "random_forest serious_death"
  "random_forest serious_life_threatening"
  "xgboost serious_any"
  "xgboost serious_hospitalization"
  "xgboost serious_death"
  "xgboost serious_life_threatening"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
MODEL="$(echo "${TASK}" | awk '{print $1}')"
TARGET="$(echo "${TASK}" | awk '{print $2}')"

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date '+%F %T')"
echo "Project dir: ${PROJECT_DIR}"
echo "Model: ${MODEL}"
echo "Target: ${TARGET}"
echo "Python: $(which python)"
python --version
echo "============================================================"

python - <<'PY'
import importlib

required = ["pandas", "pyarrow", "sklearn", "xgboost", "numpy", "matplotlib"]
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

python src/13_ml_serious_outcome_prediction.py \
  --target "${TARGET}" \
  --models "${MODEL}" \
  --scope sedative_fall \
  --n-estimators 300 \
  --search-iter 20 \
  --cv-folds 3 \
  --tune-scoring average_precision \
  --shap-sample 1000 \
  --random-state 20260608

echo "============================================================"
echo "Finished time: $(date '+%F %T')"
echo "Model: ${MODEL}"
echo "Target: ${TARGET}"
echo "============================================================"

