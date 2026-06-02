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
  "C": 3.0,
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
- Validation average precision: `0.14568647641735685`
- Validation ROC-AUC: `0.7992617275545655`
- Test average precision: `0.16156909834741817`
- Test ROC-AUC: `0.7961052792246873`
- Test Brier score: `0.029359389662976552`

## Feature highlights
- Positive association: bool__pheno_gait_balance_motor coefficient=1.3533, odds_ratio=3.8701
- Positive association: bool__pheno_dizziness_vertigo_syncope coefficient=1.0212, odds_ratio=2.7766
- Positive association: bool__pheno_consciousness_cognition coefficient=0.9128, odds_ratio=2.4913
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.7163, odds_ratio=2.0469
- Positive association: bool__event_date_known coefficient=0.6840, odds_ratio=1.9817
- Negative association: bool__indi_soc_infections_and_infestations coefficient=-0.6893, odds_ratio=0.5019
- Negative association: categorical__rept_cod_PER coefficient=-0.5999, odds_ratio=0.5488
- Negative association: categorical__occr_country_IT coefficient=-0.5038, odds_ratio=0.6042
- Negative association: categorical__reporter_country_IT coefficient=-0.4487, odds_ratio=0.6384
- Negative association: bool__has_end_dt coefficient=-0.3910, odds_ratio=0.6764

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- Logistic regression is the main narrative model because it is easier to explain in a research report.
- Odds ratios here come from model coefficients and should be interpreted as predictive associations only.
