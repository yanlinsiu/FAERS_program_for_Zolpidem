# XGBoost interpretation summary

## Run snapshot
- Model: `xgboost`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `/share/home/ycg_luanjingjie/program_FAERS/runs/mainline_2026-05-20/OUTPUT_ML/xgboost/is_fall_all_2004_2025_v2_core_20260603_025119`

## Data split
- Train used: `3,511,708` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0206`
- Validation Youden index: `0.4413`

## Final metrics
- Validation: AP=0.1297, ROC-AUC=0.7921, Brier=0.0249, Recall=0.7325, Precision=0.0649
- Test: AP=0.1376, ROC-AUC=0.7827, Brier=0.0290, Recall=0.7437, Precision=0.0699

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
- categorical__rept_cod_EXP: 0.0497
- categorical__rept_cod_PER: 0.0372
- categorical__cns_coprescription_bucket_0: 0.0316
- bool__indi_soc_nervous_system_disorders: 0.0263
- categorical__occr_country_CA: 0.0206
- bool__has_ss_drug: 0.0166
- bool__indi_soc_infections_and_infestations: 0.0157
- categorical__reporter_country_CN: 0.0150
- bool__indi_soc_musculoskeletal_and_connective_tissue_disorders: 0.0142
- categorical__sex_clean_M: 0.0139

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.
