# Logistic Regression model card

## Task
- Predict `is_fall` from the FAERS global case-level bundle.
- Use the model as a research ranking layer on top of the existing signal detection workflow.

## Data
- Signal file: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\features_v2\datasets\ml_feature_v2_2004_2025.parquet`
- Feature file: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\features_v2\datasets\ml_feature_v2_2004_2025.parquet`
- Period token: `2004_2025`
- Cohort: `all`

## Time split
- Train: years <= 2023
- Validation: 2024
- Test: 2025

## Search
- Search mode: `fast`
- Search strategy: `grid`
- Refit metric: `average_precision`
- Candidate count: `10`

## Selected parameters
```json
{
  "C": 0.3,
  "class_weight": null,
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": 0.0,
  "max_iter": 10000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "elasticnet",
  "random_state": 42,
  "solver": "saga",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

## Final metrics
- Validation average precision: `0.0953759339164145`
- Validation ROC-AUC: `0.7633680245436882`
- Test average precision: `0.10685418646162129`
- Test ROC-AUC: `0.7614389936807734`
- Test Brier score: `0.030280569665211428`

## Feature highlights
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.8671, odds_ratio=2.3800
- Positive association: bool__event_date_known coefficient=0.6858, odds_ratio=1.9854
- Positive association: categorical__rept_cod_EXP coefficient=0.6325, odds_ratio=1.8822
- Positive association: bool__indi_soc_musculoskeletal_and_connective_tissue_disorders coefficient=0.5910, odds_ratio=1.8059
- Positive association: categorical__reporter_country_FRANCE coefficient=0.5360, odds_ratio=1.7091
- Negative association: bool__indi_soc_infections_and_infestations coefficient=-0.6724, odds_ratio=0.5105
- Negative association: categorical__rept_cod_PER coefficient=-0.5945, odds_ratio=0.5519
- Negative association: categorical__occr_country_IT coefficient=-0.5092, odds_ratio=0.6010
- Negative association: categorical__reporter_country_CN coefficient=-0.4573, odds_ratio=0.6330
- Negative association: categorical__occr_country_CN coefficient=-0.3951, odds_ratio=0.6736

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- Logistic regression is the main narrative model because it is easier to explain in a research report.
- Odds ratios here come from model coefficients and should be interpreted as predictive associations only.
