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
  "penalty": "elasticnet",
  "random_state": 42,
  "solver": "saga",
  "tol": 0.0001,
  "verbose": 0,
  "warm_start": false
}
```

## Final metrics
- Validation average precision: `0.08309620245603302`
- Validation ROC-AUC: `0.7478461438835005`
- Test average precision: `0.08962203001271148`
- Test ROC-AUC: `0.7487887267167016`
- Test Brier score: `0.03315273347269471`

## Feature highlights
- Positive association: bool__event_date_known coefficient=0.9103, odds_ratio=2.4851
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.8298, odds_ratio=2.2928
- Positive association: categorical__rept_cod_EXP coefficient=0.5668, odds_ratio=1.7627
- Positive association: bool__indi_soc_musculoskeletal_and_connective_tissue_disorders coefficient=0.5546, odds_ratio=1.7413
- Positive association: bool__indi_soc_vascular_disorders coefficient=0.3386, odds_ratio=1.4030
- Negative association: categorical__rept_cod_PER coefficient=-0.6198, odds_ratio=0.5380
- Negative association: bool__has_end_dt coefficient=-0.3706, odds_ratio=0.6904
- Negative association: bool__is_antiepileptic coefficient=-0.2684, odds_ratio=0.7646
- Negative association: bool__indi_pain coefficient=-0.2374, odds_ratio=0.7887
- Negative association: bool__indi_soc_neoplasms_benign_malignant_and_unspecified_incl_cysts_and_polyps coefficient=-0.2305, odds_ratio=0.7942

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- Logistic regression is the main narrative model because it is easier to explain in a research report.
- Odds ratios here come from model coefficients and should be interpreted as predictive associations only.
