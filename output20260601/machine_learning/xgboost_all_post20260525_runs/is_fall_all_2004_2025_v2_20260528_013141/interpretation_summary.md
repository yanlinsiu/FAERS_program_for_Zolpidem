# XGBoost interpretation summary

## Run snapshot
- Model: `xgboost`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\xgboost\is_fall_all_2004_2025_v2_20260528_013141`

## Data split
- Train used: `1,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0254`
- Validation Youden index: `0.1293`

## Final metrics
- Validation: AP=0.0389, ROC-AUC=0.5899, Brier=0.0261, Recall=0.5715, Precision=0.0345
- Test: AP=0.0507, ROC-AUC=0.6113, Brier=0.0304, Recall=0.6151, Precision=0.0417

## Best tuning parameters
```json
{}
```

## Main feature signals
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

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.
