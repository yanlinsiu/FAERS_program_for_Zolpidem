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
- Search mode: `fast`
- Search strategy: `random`
- Refit metric: `average_precision`
- Candidate count: `12`

## Selected parameters
```json
{
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 0.8,
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
  "learning_rate": 0.03,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 5,
  "max_leaves": null,
  "min_child_weight": 5,
  "missing": null,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 600,
  "n_jobs": -1,
  "num_parallel_tree": null,
  "objective": "binary:logistic",
  "random_state": 42,
  "reg_alpha": null,
  "reg_lambda": 5.0,
  "sampling_method": null,
  "scale_pos_weight": 29.947775628626694,
  "subsample": 0.8,
  "tree_method": "hist",
  "validate_parameters": null,
  "verbosity": null
}
```

## Final metrics
- Validation average precision: `0.15724132627091283`
- Validation ROC-AUC: `0.8138123668515073`
- Test average precision: `0.17329531670588438`
- Test ROC-AUC: `0.8053429212055943`
- Test Brier score: `0.02835256731871333`

## Feature highlights
- bool__pheno_gait_balance_motor: 0.0600
- categorical__rept_cod_PER: 0.0470
- categorical__cns_coprescription_bucket_0: 0.0365
- bool__pheno_consciousness_cognition: 0.0348
- bool__pheno_dizziness_vertigo_syncope: 0.0325
- categorical__rept_cod_EXP: 0.0278
- bool__indi_soc_nervous_system_disorders: 0.0182
- bool__indi_soc_musculoskeletal_and_connective_tissue_disorders: 0.0165
- bool__event_date_known: 0.0141
- categorical__sex_clean_F: 0.0119

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- XGBoost is the strongest nonlinear benchmark in this repository, but it remains an auxiliary model.
- The positive-class weight is derived from the training period only, so tuning stays leakage-safe.
