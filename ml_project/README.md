# Machine Learning Workspace

This folder is separated from the existing FAERS signal-analysis pipeline.

## Layout

- `01_logistic_regression.py`: Logistic Regression baseline
- `02_random_forest.py`: Random Forest baseline
- `03_xgboost.py`: XGBoost baseline
- `ml_common.py`: shared loading, preprocessing, splitting, and output helpers

## Default behavior

- Reads from `OUTPUT_GLOBAL/datasets`
- Writes all model results to `OUTPUT_ML`
- Uses a time split by default:
  - train: `<= 2023`
  - validation: `2024`
  - test: `2025`
- Default target: `is_fall`

## Evaluation workflow

- Draw an optional random sample from the training period for faster fitting
- Run `5-fold` cross-validation inside the training period only
- Refit on the sampled training set
- Calibrate validation scores with Platt scaling
- Pick the operating threshold on `2024` with the Youden index
- Report final metrics on `2025`
- Add bootstrap `95%` confidence intervals on the test set

## Main outputs per run

- `cv_metrics.csv`: fold-level cross-validation metrics
- `validation_roc_curve.csv` and `test_roc_curve.csv`: ROC curve points
- `validation_calibration_curve.csv` and `test_calibration_curve.csv`: calibration curve data
- `test_bootstrap_metrics.csv`: bootstrap confidence intervals
- `validation_predictions.csv` and `test_predictions.csv`: raw and calibrated probabilities
- `metrics.json`: split info, threshold selection, and final metrics

## Run examples

```powershell
.\.venv\Scripts\python .\ml_project\01_logistic_regression.py
.\.venv\Scripts\python .\ml_project\02_random_forest.py
.\.venv\Scripts\python .\ml_project\03_xgboost.py
```

For a different target:

```powershell
.\.venv\Scripts\python .\ml_project\01_logistic_regression.py --target-col serious
.\.venv\Scripts\python .\ml_project\02_random_forest.py --target-col has_fall_related_broad
```
