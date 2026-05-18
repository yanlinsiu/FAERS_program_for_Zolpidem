from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    BASE_ADJUSTMENT_COLUMNS,
    OUTCOMES_BY_NAME,
    SERIOUS_ADJUSTMENT_COLUMNS,
    SIGNAL_SPECS,
    SignalSpec,
)


def _prepare_model_frame(
    df: pd.DataFrame,
    spec: SignalSpec,
    outcome_col: str,
    covariates: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    subset = df[df[spec.suspect_column].fillna(False).astype(bool)].copy()
    if spec.exclude_group:
        subset = subset[subset[spec.group_column] != spec.exclude_group].copy()
    subset = subset.dropna(subset=[outcome_col]).copy()

    y = subset[outcome_col].fillna(False).astype(int)
    exposure = subset[spec.exposure_column].fillna(False).astype(int).rename(spec.exposure_column)
    design = pd.DataFrame(index=subset.index)
    design[spec.exposure_column] = exposure.astype(float)

    if "year" in covariates:
        design["year_centered"] = pd.to_numeric(subset["year"], errors="coerce").fillna(subset["year"].median()) - float(
            pd.to_numeric(subset["year"], errors="coerce").median()
        )

    for col in covariates:
        if col == "year":
            continue
        if col in ["age_group", "sex_clean", "quarter"]:
            dummies = pd.get_dummies(
                subset[col].where(subset[col].notna(), "unknown").astype(str),
                prefix=col,
                drop_first=True,
                dtype=float,
            )
            design = pd.concat([design, dummies], axis=1)
        else:
            design[col] = subset[col].fillna(False).astype(bool).astype(float)

    non_constant_cols = [col for col in design.columns if design[col].nunique(dropna=False) > 1]
    design = design[non_constant_cols].copy()
    if spec.exposure_column not in design.columns:
        raise ValueError(f"Exposure column is constant in adjusted model: {spec.exposure_column}")
    design = sm.add_constant(design, has_constant="add")
    return design, y, subset["caseid"].astype(str)


def _fit_logit(design: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = sm.Logit(y.to_numpy(dtype=float), design.to_numpy(dtype=float))
    try:
        result = model.fit(disp=False, maxiter=200)
        converged = bool(result.mle_retvals.get("converged", False))
        message = "converged" if converged else "did_not_converge"
        params = result.params
        conf = result.conf_int(alpha=0.05)
        p_values = result.pvalues
        standard_errors = result.bse
    except Exception as exc:
        regularized = model.fit_regularized(disp=False, maxiter=200, alpha=1e-6)
        converged = False
        message = f"fallback_fit_regularized_after_{type(exc).__name__}"
        params = regularized.params
        standard_errors = np.full_like(params, np.nan, dtype=float)
        conf = np.column_stack([np.full_like(params, np.nan), np.full_like(params, np.nan)])
        p_values = np.full_like(params, np.nan, dtype=float)

    rows = pd.DataFrame(
        {
            "term": design.columns,
            "coefficient": params,
            "std_error": standard_errors,
            "p_value": p_values,
            "adjusted_reporting_odds_ratio": np.exp(np.clip(params, -50, 50)),
            "ci_low": np.exp(np.clip(conf[:, 0], -50, 50)),
            "ci_high": np.exp(np.clip(conf[:, 1], -50, 50)),
        }
    )
    diagnostics = {
        "optimization_success": converged,
        "optimization_message": message,
    }
    return rows, diagnostics


def build_adjusted_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []

    model_sets = (
        ("model_a_base", BASE_ADJUSTMENT_COLUMNS),
        ("model_b_with_serious", SERIOUS_ADJUSTMENT_COLUMNS),
    )
    for spec in SIGNAL_SPECS:
        for outcome_name in spec.outcome_names:
            outcome = OUTCOMES_BY_NAME[outcome_name]
            for model_name, covariates in model_sets:
                design, y, caseids = _prepare_model_frame(df, spec, outcome.column, covariates)
                model_df, diagnostics = _fit_logit(design, y)
                model_df.insert(0, "analysis_tier", spec.tier)
                model_df.insert(1, "analysis", spec.analysis)
                model_df.insert(2, "comparison", spec.comparison)
                model_df.insert(3, "outcome_name", outcome.name)
                model_df.insert(4, "outcome_definition", outcome.label)
                model_df.insert(5, "model", model_name)
                model_df["exposure_term"] = spec.exposure_column
                model_df["is_exposure_term"] = model_df["term"].eq(spec.exposure_column)
                model_df["n_cases"] = int(len(caseids))
                model_df["n_outcome"] = int(y.sum())
                model_df["optimization_success"] = diagnostics["optimization_success"]
                model_df["optimization_message"] = diagnostics["optimization_message"]
                result_frames.append(model_df)
                qc_rows.append(
                    {
                        "section": "adjusted",
                        "analysis_tier": spec.tier,
                        "analysis": spec.analysis,
                        "outcome_name": outcome.name,
                        "model": model_name,
                        "n_cases": int(len(caseids)),
                        "n_outcome": int(y.sum()),
                        "n_exposed": int(design[spec.exposure_column].sum()),
                        "n_terms": int(design.shape[1]),
                        **diagnostics,
                    }
                )

    result_df = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    qc_df = pd.DataFrame(qc_rows)
    return result_df, qc_df
