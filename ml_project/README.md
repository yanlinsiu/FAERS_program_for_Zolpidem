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
