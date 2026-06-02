#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --job-name=FAERS_ML
#SBATCH -o /share/home/ycg_luanjingjie/baoan/logs/faers-ml-%j.out
#SBATCH -e /share/home/ycg_luanjingjie/baoan/logs/faers-ml-%j.err

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

echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Start time: $(date '+%F %T')"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(which python)"
python --version
echo "============================================================"

python - <<'PY'
import importlib

required = ["pandas", "pyarrow", "sklearn", "xgboost", "numpy", "scipy"]
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

COMMON_ARGS=(
  --feature-version v2
  --period-token 2004_2025
  --cohort all
  --target-col is_fall
  --train-end-year 2023
  --valid-year 2024
  --test-year 2025
  --cv-folds 3
  --search-mode fast
  --bootstrap-iterations 500
  --random-state 42
)

run_model() {
  local script="$1"
  local feature_set="$2"
  local sample_n="$3"

  echo
  echo "------------------------------------------------------------"
  echo "Running ${script}, feature_set=${feature_set}, sample_n=${sample_n}"
  echo "Start: $(date '+%F %T')"
  echo "------------------------------------------------------------"

  python "${script}" "${COMMON_ARGS[@]}" \
    --feature-set "${feature_set}" \
    --train-sample-n "${sample_n}"

  echo "Finished ${script}, feature_set=${feature_set}: $(date '+%F %T')"
}

for feature_set in core enhanced; do
  run_model "ml_project/01_logistic_regression.py" "${feature_set}" 400000
  run_model "ml_project/02_random_forest.py" "${feature_set}" 400000
  run_model "ml_project/03_xgboost.py" "${feature_set}" 400000
done

echo
echo "================================================------------"
echo "All FAERS ML models finished."
echo "End time: $(date '+%F %T')"
echo "Output root: ${FAERS_ML_OUTPUT_ROOT}"
echo "================================================------------"
