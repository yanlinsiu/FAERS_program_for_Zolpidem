# Logistic Regression interpretation summary

## Run snapshot
- Model: `logistic_regression`
- Target: `is_fall`
- Feature version: `v2`
- Cohort: `all`
- Period token: `2004_2025`
- Output directory: `D:\program_FAERS\runs\mainline_2026-05-20\OUTPUT_ML\logistic_regression\is_fall_all_2004_2025_v2_core_20260528_015926`

## Data split
- Train used: `20,000` rows
- Validation: `313,722` rows
- Test: `329,593` rows

## Selected threshold
- Threshold: `0.0254`
- Validation Youden index: `0.3761`

## Final metrics
- Validation: AP=0.0831, ROC-AUC=0.7478, Brier=0.0262, Recall=0.6597, Precision=0.0604
- Test: AP=0.0896, ROC-AUC=0.7488, Brier=0.0332, Recall=0.6942, Precision=0.0671

## Best tuning parameters
```json
{}
```

## Main feature signals
- Positive association: bool__event_date_known coefficient=0.9103, odds_ratio=2.4851
- Positive association: bool__indi_soc_nervous_system_disorders coefficient=0.8298, odds_ratio=2.2928
- Positive association: categorical__rept_cod_EXP coefficient=0.5668, odds_ratio=1.7627
- Positive association: bool__indi_soc_musculoskeletal_and_connective_tissue_disorders coefficient=0.5546, odds_ratio=1.7413
- Positive association: bool__indi_soc_vascular_disorders coefficient=0.3386, odds_ratio=1.4030
- Negative association: categorical__rept_cod_PER coefficient=-0.6198, odds_ratio=0.5380
- Negative association: bool__has_end_dt coefficient=-0.3706, odds_ratio=0.6904
- Negative association: bool__is_antiepileptic coefficient=-0.2684, odds_ratio=0.7646
- Negative association: bool__indi_pain coefficient=-0.2374, odds_ratio=0.7887
- Negative association: bool__indi_soc_neoplasms_benign_malignant_and_unspecified_incl_cysts_and_polyps coefficient=-0.2305, odds_ratio=0.7942

## Plain-language caution
- These are prediction signals from FAERS reporting patterns, not causal effects.
- Use them as a ranking and explanation layer, then interpret with the main signal analysis.

## Notes
- Positive coefficients mean the model gives higher predicted probability when that encoded feature is present or larger.
- Negative coefficients mean the model gives lower predicted probability in the same predictive sense.
