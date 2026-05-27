from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

try:
    from .adjusted_models import _fit_logit, _prepare_model_frame
    from .config import (
        ADJUSTMENT_MODEL_SPECS,
        EXTENDED_ADJUSTMENT_COLUMNS,
        MIN_EXPOSED_CASES,
        MIN_EXPOSED_OUTCOME_CASES,
        AdjustmentModelSpec,
        SignalSpec,
    )
except ImportError:
    from adjusted_models import _fit_logit, _prepare_model_frame
    from config import (
        ADJUSTMENT_MODEL_SPECS,
        EXTENDED_ADJUSTMENT_COLUMNS,
        MIN_EXPOSED_CASES,
        MIN_EXPOSED_OUTCOME_CASES,
        AdjustmentModelSpec,
        SignalSpec,
    )


OUTCOME_COLUMN = "is_fall"
OUTCOME_NAME = "fall_event"
OUTCOME_DEFINITION = "Fall event"
PRIMARY_EXPOSURE = "is_zolpidem_suspect"
PRIMARY_SUSPECT_COLUMN = "suspect_role_any"
PRIMARY_GROUP_COLUMN = "target_drug_group"
PRIMARY_EXCLUDE_GROUP = "both_zolpidem_and_other_zdrug"
DEFAULT_MODEL_NAME = "extended_report_indication_adjusted"
CNS_COMEDICATION_COLUMNS = (
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
)


@dataclass(frozen=True)
class SensitivityScenario:
    section: str
    analysis: str
    label: str
    filter_func: Callable[[pd.DataFrame], pd.Series]
    exposure_column: str = PRIMARY_EXPOSURE
    suspect_column: str = PRIMARY_SUSPECT_COLUMN
    group_column: str = PRIMARY_GROUP_COLUMN
    exclude_group: str | None = PRIMARY_EXCLUDE_GROUP
    covariates_to_drop: tuple[str, ...] = ()
    model_names: tuple[str, ...] = (DEFAULT_MODEL_NAME,)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def _base_filter(df: pd.DataFrame, scenario: SensitivityScenario) -> pd.Series:
    mask = _bool_series(df, scenario.suspect_column)
    if scenario.exclude_group and scenario.group_column in df.columns:
        mask &= df[scenario.group_column].ne(scenario.exclude_group)
    return mask


def _model_spec_by_name(name: str, drop_covariates: tuple[str, ...]) -> AdjustmentModelSpec:
    model_spec = next(spec for spec in ADJUSTMENT_MODEL_SPECS if spec.name == name)
    if not drop_covariates:
        return model_spec
    drop_set = set(drop_covariates)
    return AdjustmentModelSpec(
        name=model_spec.name,
        label=model_spec.label,
        covariates=tuple(col for col in model_spec.covariates if col not in drop_set),
    )


def _signal_spec(scenario: SensitivityScenario) -> SignalSpec:
    return SignalSpec(
        analysis=scenario.analysis,
        tier="sensitivity_adjusted",
        exposure_column=scenario.exposure_column,
        suspect_column=scenario.suspect_column,
        group_column=scenario.group_column,
        outcome_names=("fall_event",),
        comparison=scenario.label,
        exclude_group=scenario.exclude_group,
    )


def _empty_result_row(
    scenario: SensitivityScenario,
    model_spec: AdjustmentModelSpec,
    n_cases: int,
    n_outcome: int,
    n_exposed: int,
    n_exposed_outcome: int,
    skip_reason: str,
) -> dict[str, object]:
    return {
        "section": scenario.section,
        "analysis": scenario.analysis,
        "comparison": scenario.label,
        "outcome_name": OUTCOME_NAME,
        "outcome_definition": OUTCOME_DEFINITION,
        "exposure": scenario.exposure_column,
        "model": model_spec.name,
        "model_label": model_spec.label,
        "n_cases": n_cases,
        "n_outcome": n_outcome,
        "n_exposed": n_exposed,
        "n_exposed_outcome": n_exposed_outcome,
        "adjusted_ror": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "optimization_success": False,
        "optimization_message": "skipped",
        "status": "skipped",
        "skip_reason": skip_reason,
        "used_covariates": "",
    }


def _summarize_scenario(df: pd.DataFrame, scenario: SensitivityScenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    scenario_df = df[scenario.filter_func(df)].copy()
    spec = _signal_spec(scenario)

    for model_name in scenario.model_names:
        model_spec = _model_spec_by_name(model_name, scenario.covariates_to_drop)
        model_label = model_spec.label

        base_mask = _base_filter(scenario_df, scenario)
        model_base = scenario_df[base_mask].copy()
        y = _bool_series(model_base, OUTCOME_COLUMN)
        exposed = _bool_series(model_base, scenario.exposure_column)
        n_cases = int(len(model_base))
        n_outcome = int(y.sum())
        n_exposed = int(exposed.sum())
        n_exposed_outcome = int((y & exposed).sum())
        skip_reason = ""

        if n_cases == 0:
            skip_reason = "no_cases_after_filter"
        elif n_outcome == 0 or n_outcome == n_cases:
            skip_reason = "outcome_has_no_variation"
        elif n_exposed == 0 or n_exposed == n_cases:
            skip_reason = "exposure_has_no_variation"
        elif n_exposed < MIN_EXPOSED_CASES:
            skip_reason = "too_few_exposed_cases"
        elif n_exposed_outcome < MIN_EXPOSED_OUTCOME_CASES:
            skip_reason = "too_few_exposed_outcome_cases"

        if skip_reason:
            row = _empty_result_row(
                scenario,
                model_spec,
                n_cases,
                n_outcome,
                n_exposed,
                n_exposed_outcome,
                skip_reason,
            )
            rows.append(row)
            qc_rows.append({**row, "section": "sensitivity_adjusted_qc"})
            continue

        try:
            design, fit_y, _caseids, used_covariates = _prepare_model_frame(
                scenario_df, spec, OUTCOME_COLUMN, model_spec
            )
            model_df, diagnostics = _fit_logit(design, fit_y)
            exposure_row = model_df[model_df["term"].eq(scenario.exposure_column)].iloc[0]
            row = {
                "section": scenario.section,
                "analysis": scenario.analysis,
                "comparison": scenario.label,
                "outcome_name": OUTCOME_NAME,
                "outcome_definition": OUTCOME_DEFINITION,
                "exposure": scenario.exposure_column,
                "model": model_spec.name,
                "model_label": model_label,
                "n_cases": int(len(fit_y)),
                "n_outcome": int(fit_y.sum()),
                "n_exposed": int(design[scenario.exposure_column].sum()),
                "n_exposed_outcome": n_exposed_outcome,
                "adjusted_ror": float(exposure_row["adjusted_reporting_odds_ratio"]),
                "ci_low": float(exposure_row["ci_low"]),
                "ci_high": float(exposure_row["ci_high"]),
                "p_value": float(exposure_row["p_value"]),
                "optimization_success": diagnostics["optimization_success"],
                "optimization_message": diagnostics["optimization_message"],
                "status": "fit",
                "skip_reason": "",
                "used_covariates": ";".join(used_covariates),
            }
        except Exception as exc:
            row = _empty_result_row(
                scenario,
                model_spec,
                n_cases,
                n_outcome,
                n_exposed,
                n_exposed_outcome,
                f"fit_failed_{type(exc).__name__}",
            )
            row["optimization_message"] = str(exc)

        rows.append(row)
        qc_rows.append(
            {
                "section": "sensitivity_adjusted_qc",
                "analysis": row["analysis"],
                "comparison": row["comparison"],
                "outcome_name": row["outcome_name"],
                "exposure": row["exposure"],
                "model": row["model"],
                "model_label": row["model_label"],
                "n_cases": row["n_cases"],
                "n_outcome": row["n_outcome"],
                "n_exposed": row["n_exposed"],
                "n_exposed_outcome": row["n_exposed_outcome"],
                "optimization_success": row["optimization_success"],
                "optimization_message": row["optimization_message"],
                "status": row["status"],
                "skip_reason": row["skip_reason"],
                "used_covariates": row["used_covariates"],
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def _all_rows(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def _col_eq(col: str, value: object) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[col].eq(value) if col in df.columns else pd.Series(False, index=df.index)


def _col_ne(col: str, value: object) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[col].ne(value) if col in df.columns else pd.Series(False, index=df.index)


def _make_scenarios(df: pd.DataFrame) -> list[SensitivityScenario]:
    scenarios: list[SensitivityScenario] = [
        SensitivityScenario(
            section="exposure",
            analysis="exposure_ps_ss",
            label="Zolpidem primary or secondary suspect",
            filter_func=_all_rows,
            model_names=("core_clinical_adjusted", DEFAULT_MODEL_NAME),
        ),
        SensitivityScenario(
            section="exposure",
            analysis="exposure_ps_only",
            label="Zolpidem primary suspect only",
            filter_func=_all_rows,
            exposure_column="is_zolpidem_suspect_ps",
            suspect_column="suspect_role_any_ps",
            group_column="target_drug_group_ps",
            model_names=("core_clinical_adjusted", DEFAULT_MODEL_NAME),
        ),
        SensitivityScenario(
            section="indication",
            analysis="indication_any_indi",
            label="Cases with at least one indication record",
            filter_func=lambda frame: pd.to_numeric(frame["indi_n"], errors="coerce").fillna(0).gt(0),
        ),
        SensitivityScenario(
            section="indication",
            analysis="indication_insomnia",
            label="Cases with insomnia indication",
            filter_func=lambda frame: _bool_series(frame, "indi_insomnia"),
            covariates_to_drop=("indi_insomnia",),
        ),
        SensitivityScenario(
            section="indication",
            analysis="indication_exclude_dizziness_vertigo",
            label="Excluding dizziness or vertigo indication",
            filter_func=lambda frame: ~_bool_series(frame, "indi_dizziness_vertigo"),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="reporting_e_sub_y",
            label="Electronic submission reports",
            filter_func=_col_eq("e_sub", "Y"),
            covariates_to_drop=("e_sub",),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="reporting_e_sub_n",
            label="Non-electronic submission reports",
            filter_func=_col_eq("e_sub", "N"),
            covariates_to_drop=("e_sub",),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="country_us",
            label="Reporter country US",
            filter_func=lambda frame: frame["reporter_country"].astype(str).str.upper().isin(
                {"US", "UNITED STATES"}
            ),
            covariates_to_drop=("reporter_country",),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="country_non_us",
            label="Reporter country non-US",
            filter_func=lambda frame: ~frame["reporter_country"].astype(str).str.upper().isin(
                {"US", "UNITED STATES"}
            ),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="time_2004_2012",
            label="Report years 2004-2012",
            filter_func=lambda frame: pd.to_numeric(frame["year"], errors="coerce").between(2004, 2012),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="time_2013_2019",
            label="Report years 2013-2019",
            filter_func=lambda frame: pd.to_numeric(frame["year"], errors="coerce").between(2013, 2019),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="time_2020_2025",
            label="Report years 2020-2025",
            filter_func=lambda frame: pd.to_numeric(frame["year"], errors="coerce").between(2020, 2025),
        ),
        SensitivityScenario(
            section="reporting_country_time",
            analysis="time_exclude_2025",
            label="Report years 2004-2024",
            filter_func=lambda frame: pd.to_numeric(frame["year"], errors="coerce").between(2004, 2024),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="age_65_74",
            label="Age 65-74",
            filter_func=_col_eq("age_group", "65-74"),
            covariates_to_drop=("age_group",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="age_75_84",
            label="Age 75-84",
            filter_func=_col_eq("age_group", "75-84"),
            covariates_to_drop=("age_group",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="age_85_plus",
            label="Age >=85",
            filter_func=_col_eq("age_group", ">=85"),
            covariates_to_drop=("age_group",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="comedication_exclude_benzo",
            label="Excluding benzodiazepine co-reports",
            filter_func=lambda frame: ~_bool_series(frame, "is_benzo"),
            covariates_to_drop=("is_benzo",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="comedication_exclude_antidepressant",
            label="Excluding antidepressant co-reports",
            filter_func=lambda frame: ~_bool_series(frame, "is_antidepressant"),
            covariates_to_drop=("is_antidepressant",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="comedication_exclude_opioid",
            label="Excluding opioid co-reports",
            filter_func=lambda frame: ~_bool_series(frame, "is_opioid"),
            covariates_to_drop=("is_opioid",),
        ),
        SensitivityScenario(
            section="age_comedication",
            analysis="comedication_exclude_any_cns",
            label="Excluding any selected CNS high-risk co-report",
            filter_func=lambda frame: ~pd.concat(
                [_bool_series(frame, col) for col in CNS_COMEDICATION_COLUMNS], axis=1
            ).any(axis=1),
            covariates_to_drop=CNS_COMEDICATION_COLUMNS,
        ),
    ]

    base_mask = _base_filter(df, scenarios[0])
    common_rept_codes = (
        df.loc[base_mask, "rept_cod"].value_counts(dropna=False).head(5).index.tolist()
        if "rept_cod" in df.columns
        else []
    )
    for code in common_rept_codes:
        scenarios.append(
            SensitivityScenario(
                section="reporting_country_time",
                analysis=f"reporting_rept_cod_{code}",
                label=f"Report type {code}",
                filter_func=_col_eq("rept_cod", code),
                covariates_to_drop=("rept_cod",),
            )
        )

    top_countries = (
        df.loc[base_mask, "reporter_country"].value_counts(dropna=False).head(5).index.tolist()
        if "reporter_country" in df.columns
        else []
    )
    for country in top_countries:
        scenarios.append(
            SensitivityScenario(
                section="reporting_country_time",
                analysis=f"country_top_{str(country).replace(' ', '_')}",
                label=f"Reporter country {country}",
                filter_func=_col_eq("reporter_country", country),
                covariates_to_drop=("reporter_country",),
            )
        )

    return scenarios


def build_sensitivity_adjusted_analysis(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    qc_frames: list[pd.DataFrame] = []

    for scenario in _make_scenarios(df):
        result_df, qc_df = _summarize_scenario(df, scenario)
        result_frames.append(result_df)
        qc_frames.append(qc_df)

    all_summary = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    qc = pd.concat(qc_frames, ignore_index=True) if qc_frames else pd.DataFrame()
    tables = {
        "sensitivity_exposure": all_summary[all_summary["section"].eq("exposure")].copy(),
        "sensitivity_indication": all_summary[all_summary["section"].eq("indication")].copy(),
        "sensitivity_reporting_country_time": all_summary[
            all_summary["section"].eq("reporting_country_time")
        ].copy(),
        "sensitivity_age_comedication": all_summary[
            all_summary["section"].eq("age_comedication")
        ].copy(),
        "sensitivity_all_summary": all_summary,
    }
    return tables, all_summary, qc

