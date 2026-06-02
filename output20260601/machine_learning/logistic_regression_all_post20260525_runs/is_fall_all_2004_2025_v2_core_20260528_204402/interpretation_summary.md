# Logistic Regression interpretation summary

## Run snapshot
- Model: `logistic_regression`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\logistic_regression\is_fall_all_2004_2025_v2_core_20260528_204402`

## Data split
- Train used: `400,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0230`
- Validation Youden index: `0.3967`

## Final metrics
- Validation: AP=0.0954, ROC-AUC=0.7634, Brier=0.0258, Recall=0.7248, Precision=0.0575
- Test: AP=0.1069, ROC-AUC=0.7614, Brier=0.0303, Recall=0.7358, Precision=0.0653

## Best tuning parameters
```json
{
  "model__C": 0.3,
  "model__class_weight": null,
  "model__l1_ratio": 0.0,
  "model__penalty": "elasticnet"
}
```

## Main feature signals
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.8671, odds_ratio=2.3800
- Positive association: bool__event_date_known coefficient=0.6858, odds_ratio=1.9854
- Positive association: categorical__rept_cod_EXP coefficient=0.6325, odds_ratio=1.8822
- Positive association: bool__indi_soc_musculoskeletal_and_connective_tissue_disorders coefficient=0.5910, odds_ratio=1.8059
- Positive association: categorical__reporter_country_FRANCE coefficient=0.5360, odds_ratio=1.7091
- Negative association: bool__indi_soc_infections_and_infestations coefficient=-0.6724, odds_ratio=0.5105
- Negative association: categorical__rept_cod_PER coefficient=-0.5945, odds_ratio=0.5519
- Negative association: categorical__occr_country_IT coefficient=-0.5092, odds_ratio=0.6010
- Negative association: categorical__reporter_country_CN coefficient=-0.4573, odds_ratio=0.6330
- Negative association: categorical__occr_country_CN coefficient=-0.3951, odds_ratio=0.6736

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Positive coefficients mean the model gives higher predicted probability when that encoded feature is present or larger.
- Negative coefficients mean the model gives lower predicted probability in the same predictive sense.
