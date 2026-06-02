# XGBoost interpretation summary

## Run snapshot
- Model: `xgboost`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\xgboost\is_fall_all_2004_2025_v2_core_20260529_113116`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0250`
- Validation Youden index: `0.4261`

## Final metrics
- Validation: AP=0.1057, ROC-AUC=0.7791, Brier=0.0251, Recall=0.7297, Precision=0.0622
- Test: AP=0.1142, ROC-AUC=0.7692, Brier=0.0292, Recall=0.7346, Precision=0.0675

## Best tuning parameters
```json
{
  "model__subsample": 0.8,
  "model__reg_lambda": 5.0,
  "model__n_estimators": 400,
  "model__min_child_weight": 1,
  "model__max_depth": 5,
  "model__learning_rate": 0.05,
  "model__colsample_bytree": 1.0
}
```

## Main feature signals
- categorical__rept_cod_PER: 0.0708
- categorical__cns_coprescription_bucket_0: 0.0494
- bool__indi_soc_nervous_system_disorders: 0.0275
- categorical__rept_cod_EXP: 0.0224
- categorical__occr_country_CA: 0.0177
- bool__indi_soc_musculoskeletal_and_connective_tissue_disorders: 0.0170
- bool__event_date_known: 0.0157
- bool__indi_soc_infections_and_infestations: 0.0148
- bool__indi_soc_neoplasms_benign_malignant_and_unspecified_incl_cysts_and_polyps: 0.0120
- categorical__sex_clean_F: 0.0112

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.
