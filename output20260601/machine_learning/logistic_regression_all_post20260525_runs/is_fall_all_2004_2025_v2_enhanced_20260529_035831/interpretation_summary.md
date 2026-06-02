# Logistic Regression interpretation summary

## Run snapshot
- Model: `logistic_regression`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\logistic_regression\is_fall_all_2004_2025_v2_enhanced_20260529_035831`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0229`
- Validation Youden index: `0.4545`

## Final metrics
- Validation: AP=0.1457, ROC-AUC=0.7993, Brier=0.0253, Recall=0.7192, Precision=0.0698
- Test: AP=0.1616, ROC-AUC=0.7961, Brier=0.0294, Recall=0.7310, Precision=0.0780

## Best tuning parameters
```json
{
  "model__C": 3.0,
  "model__class_weight": null,
  "model__l1_ratio": 0.0,
  "model__penalty": "elasticnet"
}
```

## Main feature signals
- Positive association: bool__pheno_gait_balance_motor coefficient=1.3533, odds_ratio=3.8701
- Positive association: bool__pheno_dizziness_vertigo_syncope coefficient=1.0212, odds_ratio=2.7766
- Positive association: bool__pheno_consciousness_cognition coefficient=0.9128, odds_ratio=2.4913
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.7163, odds_ratio=2.0469
- Positive association: bool__event_date_known coefficient=0.6840, odds_ratio=1.9817
- Negative association: bool__indi_soc_infections_and_infestations coefficient=-0.6893, odds_ratio=0.5019
- Negative association: categorical__rept_cod_PER coefficient=-0.5999, odds_ratio=0.5488
- Negative association: categorical__occr_country_IT coefficient=-0.5038, odds_ratio=0.6042
- Negative association: categorical__reporter_country_IT coefficient=-0.4487, odds_ratio=0.6384
- Negative association: bool__has_end_dt coefficient=-0.3910, odds_ratio=0.6764

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Positive coefficients mean the model gives higher predicted probability when that encoded feature is present or larger.
- Negative coefficients mean the model gives lower predicted probability in the same predictive sense.
