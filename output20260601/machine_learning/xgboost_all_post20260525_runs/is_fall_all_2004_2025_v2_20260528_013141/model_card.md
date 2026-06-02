# XGBoost model card

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
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": null,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": false,
  "eval_metric": "logloss",
  "feature_types": null,
  "feature_weights": null,
  "gamma": null,
  "grow_policy": null,
  "importance_type": null,
  "interaction_constraints": null,
  "learning_rate": null,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": null,
  "max_leaves": null,
  "min_child_weight": null,
  "missing": null,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": null,
  "n_jobs": -1,
  "num_parallel_tree": null,
  "objective": "binary:logistic",
  "random_state": 42,
  "reg_alpha": null,
  "reg_lambda": null,
  "sampling_method": null,
  "scale_pos_weight": 30.25,
  "subsample": null,
  "tree_method": "hist",
  "validate_parameters": null,
  "verbosity": null
}
```

## Final metrics
- Validation average precision: `0.03894441407895646`
- Validation ROC-AUC: `0.589948747743424`
- Test average precision: `0.05072154593397983`
- Test ROC-AUC: `0.611262040900011`
- Test Brier score: `0.03039716242499976`

## Feature highlights
- bool__is_opioid: 0.1029
- categorical__cns_coprescription_bucket_0: 0.0865
- categorical__reporter_country_FRANCE: 0.0560
- bool__indi_depression: 0.0498
- bool__indi_anxiety: 0.0497
- categorical__reporter_country_AR: 0.0338
- categorical__reporter_country_GB: 0.0306
- bool__indi_insomnia: 0.0284
- categorical__reporter_country_JP: 0.0273
- bool__has_end_dt: 0.0228

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- XGBoost is the strongest nonlinear benchmark in this repository, but it remains an auxiliary model.
- The positive-class weight is derived from the training period only, so tuning stays leakage-safe.
