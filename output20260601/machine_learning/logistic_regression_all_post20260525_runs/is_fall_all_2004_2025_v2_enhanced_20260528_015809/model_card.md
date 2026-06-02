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
- Search mode: `none`
- Search strategy: `none`
- Refit metric: `average_precision`
- Candidate count: `None`

## Selected parameters
```json
{
  "C": 1.0,
  "class_weight": null,
  "dual": false,
  "fit_intercept": true,
  "intercept_scaling": 1,
  "l1_ratio": 0.0,
  "max_iter": 10000,
  "multi_class": "deprecated",
  "n_jobs": null,
  "penalty": "l2",
  "random_state": 42,
  "solver": "saga",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

## Final metrics
- Validation average precision: `0.12067435675376335`
- Validation ROC-AUC: `0.7840208591929079`
- Test average precision: `0.1174415848235896`
- Test ROC-AUC: `0.781334429630283`
- Test Brier score: `0.033049850899107336`

## Feature highlights
- Positive association: bool__pheno_gait_balance_motor coefficient=0.9817, odds_ratio=2.6689
- Positive association: bool__pheno_dizziness_vertigo_syncope coefficient=0.9618, odds_ratio=2.6163
- Positive association: bool__event_date_known coefficient=0.9264, odds_ratio=2.5254
- Positive association: bool__pheno_consciousness_cognition coefficient=0.8455, odds_ratio=2.3291
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.7483, odds_ratio=2.1135
- Negative association: categorical__rept_cod_PER coefficient=-0.5994, odds_ratio=0.5491
- Negative association: bool__has_end_dt coefficient=-0.3885, odds_ratio=0.6781
- Negative association: bool__indi_pain coefficient=-0.3028, odds_ratio=0.7387
- Negative association: bool__is_antiepileptic coefficient=-0.2982, odds_ratio=0.7422
- Negative association: categorical__e_sub_Y coefficient=-0.2123, odds_ratio=0.8087

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- Logistic regression is the main narrative model because it is easier to explain in a research report.
- Odds ratios here come from model coefficients and should be interpreted as predictive associations only.
