from __future__ import annotations

import pandas as pd


PRIMARY_COLUMNS = [
    "analysis_tier",
    "analysis",
    "comparison",
    "outcome_name",
    "a",
    "b",
    "c",
    "d",
    "exposed_n",
    "reporting_rate_exposed",
    "reporting_rate_unexposed",
    "ror",
    "ror_ci_low",
    "ror_ci_high",
    "prr",
    "prr_ci_low",
    "prr_ci_high",
    "ic",
    "ic025",
    "ebgm",
    "eb05",
    "p_value",
    "stability_status",
    "is_stable_signal",
    "conclusion",
]


EXPLORATORY_COLUMNS = [
    "analysis_tier",
    "analysis",
    "comparison",
    "outcome_name",
    "feature_domain",
    "feature_name",
    "feature_label",
    "a",
    "exposed_n",
    "reporting_rate_exposed",
    "reporting_rate_unexposed",
    "ror",
    "ror_ci_low",
    "ror_ci_high",
    "p_value",
    "fdr_q_value",
    "fdr_significant",
    "stability_status",
    "is_stable_signal",
    "conclusion",
]


def build_primary_summary(primary_df: pd.DataFrame, adjusted_df: pd.DataFrame) -> pd.DataFrame:
    summary = primary_df[[col for col in PRIMARY_COLUMNS if col in primary_df.columns]].copy()
    exposure_terms = adjusted_df[adjusted_df["is_exposure_term"].fillna(False)].copy()
    exposure_terms = exposure_terms[exposure_terms["analysis_tier"].isin(["primary", "sensitivity"])]
    if exposure_terms.empty:
        return summary

    adjusted_pivot = exposure_terms.pivot_table(
        index=["analysis_tier", "analysis", "comparison", "outcome_name"],
        columns="model",
        values=["adjusted_reporting_odds_ratio", "ci_low", "ci_high", "p_value"],
        aggfunc="first",
    )
    adjusted_pivot.columns = [f"{model}_{metric}" for metric, model in adjusted_pivot.columns]
    adjusted_pivot = adjusted_pivot.reset_index()
    return summary.merge(
        adjusted_pivot,
        on=["analysis_tier", "analysis", "comparison", "outcome_name"],
        how="left",
    )


def build_exploratory_summary(exploratory_df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in EXPLORATORY_COLUMNS if col in exploratory_df.columns]
    summary = exploratory_df[columns].copy()
    sort_cols = [col for col in ["analysis", "outcome_name", "fdr_q_value", "ror"] if col in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols, ascending=[True, True, True, False][: len(sort_cols)])
    return summary
