# Logistic Regression interpretation summary

## Run snapshot
- Model: `logistic_regression`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\logistic_regression\is_fall_all_2004_2025_v2_enhanced_20260528_015809`

## Data split
- Train used: `20,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0239`
- Validation Youden index: `0.4335`

## Final metrics
- Validation: AP=0.1207, ROC-AUC=0.7840, Brier=0.0257, Recall=0.6967, Precision=0.0681
- Test: AP=0.1174, ROC-AUC=0.7813, Brier=0.0330, Recall=0.7227, Precision=0.0738

## Best tuning parameters
```json
{}
```

## Main feature signals
- Positive association: bool__pheno_gait_balance_motor coefficient=0.9817, odds_ratio=2.6689
- Positive association: bool__pheno_dizziness_vertigo_syncope coefficient=0.9618, odds_ratio=2.6163
- Positive association: bool__event_date_known coefficient=0.9264, odds_ratio=2.5254
- Positive association: bool__pheno_consciousness_cognition coefficient=0.8455, odds_ratio=2.3291
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.7483, odds_ratio=2.1135
- Negative association: categorical__rept_cod_PER coefficient=-0.5994, odds_ratio=0.5491
- Negative association: bool__has_end_dt coefficient=-0.3885, odds_ratio=0.6781
- Negative association: bool__indi_pain coefficient=-0.3028, odds_ratio=0.7387
- Negative association: bool__is_antiepileptic coefficient=-0.2982, odds_ratio=0.7422
- Negative association: categorical__e_sub_Y coefficient=-0.2123, odds_ratio=0.8087

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Positive coefficients mean the model gives higher predicted probability when that encoded feature is present or larger.
- Negative coefficients mean the model gives lower predicted probability in the same predictive sense.
