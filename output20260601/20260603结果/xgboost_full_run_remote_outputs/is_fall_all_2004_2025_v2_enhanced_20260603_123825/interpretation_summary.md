# XGBoost interpretation summary

## Run snapshot
- Model: `xgboost`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `/share/home/ycg_luanjingjie/program_FAERS/runs/mainline_2026-05-20/OUTPUT_ML/xgboost/is_fall_all_2004_2025_v2_enhanced_20260603_123825`

## Data split
- Train used: `3,511,708` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0197`
- Validation Youden index: `0.4893`

## Final metrics
- Validation: AP=0.1753, ROC-AUC=0.8223, Brier=0.0243, Recall=0.7477, Precision=0.0740
- Test: AP=0.1883, ROC-AUC=0.8150, Brier=0.0282, Recall=0.7589, Precision=0.0791

## Best tuning parameters
```json
{
  "model__subsample": 0.7,
  "model__scale_pos_weight": 5.0,
  "model__reg_lambda": 1.0,
  "model__reg_alpha": 1.0,
  "model__n_estimators": 600,
  "model__min_child_weight": 3,
  "model__max_depth": 6,
  "model__learning_rate": 0.08,
  "model__gamma": 0.0,
  "model__colsample_bytree": 0.9
}
```

## Main feature signals
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

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.
