from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
GLOBAL_OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "analysis_v2"

MIN_EXPOSED_OUTCOME_CASES = 5
MIN_EXPOSED_CASES = 50


@dataclass(frozen=True)
class OutcomeSpec:
    name: str
    column: str
    label: str


@dataclass(frozen=True)
class SignalSpec:
    analysis: str
    tier: str
    exposure_column: str
    suspect_column: str
    group_column: str
    outcome_names: tuple[str, ...]
    comparison: str
    exclude_group: str | None = "both_zolpidem_and_other_zdrug"


@dataclass(frozen=True)
class GroupComparisonSpec:
    analysis: str
    tier: str
    group_column: str
    exposed_value: str
    reference_value: str
    outcome_names: tuple[str, ...]
    comparison: str


@dataclass(frozen=True)
class FeatureSpec:
    column: str
    value: object
    domain: str
    label: str


@dataclass(frozen=True)
class AdjustmentModelSpec:
    name: str
    label: str
    covariates: tuple[str, ...]


OUTCOME_SPECS: tuple[OutcomeSpec, ...] = (
    OutcomeSpec("strict_fall", "is_fall_narrow", "OCMQ-compatible narrow fall event"),
)

OUTCOMES_BY_NAME = {spec.name: spec for spec in OUTCOME_SPECS}

SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec(
        analysis="primary_ps_ss",
        tier="primary",
        exposure_column="is_zolpidem_suspect",
        suspect_column="suspect_role_any",
        group_column="target_drug_group",
        outcome_names=("strict_fall",),
        comparison="zolpidem_suspect_vs_all_other_suspect_drugs_excluding_mixed_zdrug_cases",
    ),
    SignalSpec(
        analysis="sensitivity_ps_only",
        tier="sensitivity",
        exposure_column="is_zolpidem_suspect_ps",
        suspect_column="suspect_role_any_ps",
        group_column="target_drug_group_ps",
        outcome_names=("strict_fall",),
        comparison="zolpidem_primary_suspect_vs_all_other_primary_suspect_drugs_excluding_mixed_zdrug_cases",
    ),
)

GROUP_COMPARISON_SPECS: tuple[GroupComparisonSpec, ...] = (
    GroupComparisonSpec(
        analysis="comparative_ps_ss",
        tier="sensitivity",
        group_column="target_drug_group",
        exposed_value="zolpidem_only",
        reference_value="other_zdrug_only",
        outcome_names=("strict_fall",),
        comparison="zolpidem_only_vs_other_zdrug_only",
    ),
    GroupComparisonSpec(
        analysis="comparative_ps_only",
        tier="sensitivity",
        group_column="target_drug_group_ps",
        exposed_value="zolpidem_only",
        reference_value="other_zdrug_only",
        outcome_names=("strict_fall",),
        comparison="zolpidem_only_vs_other_zdrug_only_primary_suspect_only",
    ),
)

EXPLORATORY_SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec(
        analysis="primary_ps_ss",
        tier="exploratory",
        exposure_column="is_zolpidem_suspect",
        suspect_column="suspect_role_any",
        group_column="target_drug_group",
        outcome_names=("strict_fall",),
        comparison="feature_positive_vs_feature_negative_among_zolpidem_suspect_cases",
    ),
    SignalSpec(
        analysis="sensitivity_ps_only",
        tier="exploratory",
        exposure_column="is_zolpidem_suspect_ps",
        suspect_column="suspect_role_any_ps",
        group_column="target_drug_group_ps",
        outcome_names=("strict_fall",),
        comparison="feature_positive_vs_feature_negative_among_zolpidem_primary_suspect_cases",
    ),
)

FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("age_group", "65-74", "demographic", "Age 65-74"),
    FeatureSpec("age_group", "75-84", "demographic", "Age 75-84"),
    FeatureSpec("age_group", ">=85", "demographic", "Age >=85"),
    FeatureSpec("sex_clean", "F", "demographic", "Female"),
    FeatureSpec("sex_clean", "M", "demographic", "Male"),
    FeatureSpec("polypharmacy_5", True, "medication_burden", "Polypharmacy >=5"),
    FeatureSpec("is_benzo", True, "co_medication", "Benzodiazepine co-report"),
    FeatureSpec("is_antidepressant", True, "co_medication", "Antidepressant co-report"),
    FeatureSpec("is_antipsychotic", True, "co_medication", "Antipsychotic co-report"),
    FeatureSpec("is_opioid", True, "co_medication", "Opioid co-report"),
    FeatureSpec("is_antiepileptic", True, "co_medication", "Antiepileptic co-report"),
)

CORE_ADJUSTMENT_COLUMNS: tuple[str, ...] = (
    "age_group",
    "sex_clean",
    "year",
    "quarter",
    "polypharmacy_5",
    "distinct_drug_n",
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
)

EXTENDED_ADJUSTMENT_COLUMNS: tuple[str, ...] = (
    *CORE_ADJUSTMENT_COLUMNS,
    "reporter_country",
    "rept_cod",
    "e_sub",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
    "indi_dizziness_vertigo",
)

ADJUSTMENT_MODEL_SPECS: tuple[AdjustmentModelSpec, ...] = (
    AdjustmentModelSpec(
        name="core_clinical_adjusted",
        label="Core clinical adjustment model",
        covariates=CORE_ADJUSTMENT_COLUMNS,
    ),
    AdjustmentModelSpec(
        name="extended_report_indication_adjusted",
        label="Extended report-indication adjustment model",
        covariates=EXTENDED_ADJUSTMENT_COLUMNS,
    ),
)

CATEGORICAL_ADJUSTMENT_COLUMNS: tuple[str, ...] = (
    "age_group",
    "sex_clean",
    "quarter",
    "reporter_country",
    "occr_country",
    "rept_cod",
    "e_sub",
    "rpsr_cod",
)

NUMERIC_ADJUSTMENT_COLUMNS: tuple[str, ...] = (
    "year",
    "age_years",
    "drug_n",
    "distinct_drug_n",
    "indi_n",
    "distinct_indi_n",
)

BOOL_COLUMNS: tuple[str, ...] = (
    "is_fall_narrow",
    "is_fall_broad",
    "is_zolpidem_any",
    "is_zolpidem_suspect",
    "is_zolpidem_suspect_ps",
    "is_other_zdrug_suspect",
    "is_other_zdrug_suspect_ps",
    "suspect_role_any",
    "suspect_role_any_ps",
    "serious",
    "is_zolpidem",
    "is_zaleplon",
    "is_zopiclone",
    "is_eszopiclone",
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
    "polypharmacy_5",
    "polypharmacy",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
    "indi_dizziness_vertigo",
)
