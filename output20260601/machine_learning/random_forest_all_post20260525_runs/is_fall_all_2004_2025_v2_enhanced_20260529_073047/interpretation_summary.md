# Random Forest interpretation summary

## Run snapshot
- Model: `random_forest`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\random_forest\is_fall_all_2004_2025_v2_enhanced_20260529_073047`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0250`
- Validation Youden index: `0.4601`

## Final metrics
- Validation: AP=0.1524, ROC-AUC=0.8063, Brier=0.0244, Recall=0.7414, Precision=0.0678
- Test: AP=0.1640, ROC-AUC=0.8008, Brier=0.0283, Recall=0.7581, Precision=0.0749

## Best tuning parameters
```json
{
  "model__class_weight": "balanced_subsample",
  "model__max_depth": 20,
  "model__max_features": "sqrt",
  "model__min_samples_leaf": 10,
  "model__n_estimators": 200
}
```

## Main feature signals
- categorical__rept_cod_PER: 0.0624
- bool__pheno_gait_balance_motor: 0.0616
- categorical__rept_cod_EXP: 0.0498
- bool__pheno_consciousness_cognition: 0.0472
- numeric__age_years: 0.0471
- bool__pheno_dizziness_vertigo_syncope: 0.0467
- bool__indi_soc_nervous_system_disorders: 0.0312
- numeric__year: 0.0249
- bool__event_date_known: 0.0232
- categorical__age_group_65-74: 0.0222

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Random Forest importance is useful for rough ranking, but correlated features can split importance between each other.
