# XGBoost interpretation summary

## Run snapshot
- Model: `xgboost`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\xgboost\is_fall_all_2004_2025_v2_enhanced_20260529_124326`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0215`
- Validation Youden index: `0.4754`

## Final metrics
- Validation: AP=0.1572, ROC-AUC=0.8138, Brier=0.0244, Recall=0.7681, Precision=0.0676
- Test: AP=0.1733, ROC-AUC=0.8053, Brier=0.0284, Recall=0.7778, Precision=0.0731

## Best tuning parameters
```json
{
  "model__subsample": 0.8,
  "model__reg_lambda": 5.0,
  "model__n_estimators": 600,
  "model__min_child_weight": 5,
  "model__max_depth": 5,
  "model__learning_rate": 0.03,
  "model__colsample_bytree": 0.8
}
```

## Main feature signals
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

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.
