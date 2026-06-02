# Random Forest interpretation summary

## Run snapshot
- Model: `random_forest`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\random_forest\is_fall_all_2004_2025_v2_core_20260529_054847`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0245`
- Validation Youden index: `0.4118`

## Final metrics
- Validation: AP=0.1029, ROC-AUC=0.7739, Brier=0.0251, Recall=0.7331, Precision=0.0593
- Test: AP=0.1121, ROC-AUC=0.7665, Brier=0.0292, Recall=0.7431, Precision=0.0656

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
- categorical__rept_cod_PER: 0.0643
- categorical__rept_cod_EXP: 0.0619
- numeric__age_years: 0.0554
- bool__indi_soc_nervous_system_disorders: 0.0382
- numeric__year: 0.0313
- categorical__age_group_65-74: 0.0272
- bool__event_date_known: 0.0268
- numeric__log_drug_n: 0.0246
- numeric__drug_n: 0.0234
- numeric__therapy_record_n: 0.0227

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Random Forest importance is useful for rough ranking, but correlated features can split importance between each other.
