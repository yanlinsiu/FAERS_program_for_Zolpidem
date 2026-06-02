# Random Forest model card

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
- Candidate count: `12`

## Selected parameters
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "class_weight": "balanced_subsample",
  "criterion": "gini",
  "max_depth": 20,
  "max_features": "sqrt",
  "max_leaf_nodes": null,
  "max_samples": null,
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 10,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": null,
  "n_estimators": 200,
  "n_jobs": -1,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

## Final metrics
- Validation average precision: `0.15239298156275097`
- Validation ROC-AUC: `0.8063136827843592`
- Test average precision: `0.1639955401347521`
- Test ROC-AUC: `0.8008241673153567`
- Test Brier score: `0.028347550930586805`

## Feature highlights
- categorical__rept_cod_PER: 0.0624
- bool__pheno_gait_balance_motor: 0.0616
- categorical__rept_cod_EXP: 0.0498
- bool__pheno_consciousness_cognition: 0.0472
- numeric__age_years: 0.0471
- bool__pheno_dizziness_vertigo_syncope: 0.0467
- bool__indi_soc_nervous_system_disorders: 0.0312
- numeric__year: 0.0249
- bool__event_date_known: 0.0232
- categorical__age_group_65-74: 0.0222

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- Random Forest is used as a nonlinear benchmark against the main logistic regression model.
- Feature importance here is impurity-based and should be read as a rough ranking, not a causal explanation.
