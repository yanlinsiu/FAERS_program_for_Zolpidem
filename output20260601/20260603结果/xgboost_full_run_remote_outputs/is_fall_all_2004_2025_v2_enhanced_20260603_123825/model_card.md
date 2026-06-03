# XGBoost model card

## Task
- Predict `is_fall` from the FAERS global case-level bundle.
- Use the model as a research ranking layer on top of the existing signal detection workflow.

## Data
- Signal file: `/share/home/ycg_luanjingjie/program_FAERS/runs/mainline_2026-05-20/OUTPUT_ML/features_v2/datasets/ml_feature_v2_2004_2025.parquet`
- Feature file: `/share/home/ycg_luanjingjie/program_FAERS/runs/mainline_2026-05-20/OUTPUT_ML/features_v2/datasets/ml_feature_v2_2004_2025.parquet`
- Period token: `2004_2025`
- Cohort: `all`

## Time split
- Train: years <= 2023
- Validation: 2024
- Test: 2025

## Search
- Search mode: `full`
- Search strategy: `random`
- Refit metric: `average_precision`
- Candidate count: `60`

## Selected parameters
```json
{
  "base_score": null,
  "booster": null,
  "callbacks": null,
  "colsample_bylevel": null,
  "colsample_bynode": null,
  "colsample_bytree": 0.9,
  "device": null,
  "early_stopping_rounds": null,
  "enable_categorical": false,
  "eval_metric": "logloss",
  "feature_types": null,
  "feature_weights": null,
  "gamma": 0.0,
  "grow_policy": null,
  "importance_type": null,
  "interaction_constraints": null,
  "learning_rate": 0.08,
  "max_bin": null,
  "max_cat_threshold": null,
  "max_cat_to_onehot": null,
  "max_delta_step": null,
  "max_depth": 6,
  "max_leaves": null,
  "min_child_weight": 3,
  "missing": null,
  "monotone_constraints": null,
  "multi_strategy": null,
  "n_estimators": 600,
  "n_jobs": 4,
  "num_parallel_tree": null,
  "objective": "binary:logistic",
  "random_state": 42,
  "reg_alpha": 1.0,
  "reg_lambda": 1.0,
  "sampling_method": null,
  "scale_pos_weight": 5.0,
  "subsample": 0.7,
  "tree_method": "hist",
  "validate_parameters": null,
  "verbosity": null
}
```

## Final metrics
- Validation average precision: `0.17533811962219262`
- Validation ROC-AUC: `0.8222802346675813`
- Test average precision: `0.18825657640019225`
- Test ROC-AUC: `0.8150023012763901`
- Test Brier score: `0.02818345390831909`

## Feature highlights
- bool__pheno_gait_balance_motor: 0.0766
- categorical__rept_cod_PER: 0.0359
- bool__pheno_consciousness_cognition: 0.0322
- categorical__rept_cod_EXP: 0.0312
- bool__pheno_dizziness_vertigo_syncope: 0.0271
- categorical__cns_coprescription_bucket_0: 0.0240
- bool__indi_soc_nervous_system_disorders: 0.0170
- bool__indi_soc_infections_and_infestations: 0.0152
- categorical__occr_country_CA: 0.0150
- categorical__reporter_country_CN: 0.0139

## Limitations
- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.
- The output reflects reporting patterns in FAERS, not causal drug effects.
- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.

## Notes
- XGBoost is the strongest nonlinear benchmark in this repository, but it remains an auxiliary model.
- The positive-class weight is derived from the training period only, so tuning stays leakage-safe.
