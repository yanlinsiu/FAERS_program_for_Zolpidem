from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from .config import (
        ADJUSTMENT_MODEL_SPECS,
        CATEGORICAL_ADJUSTMENT_COLUMNS,
        NUMERIC_ADJUSTMENT_COLUMNS,
        OUTCOMES_BY_NAME,
        SIGNAL_SPECS,
        AdjustmentModelSpec,
        SignalSpec,
    )
except ImportError:
    from config import (
        ADJUSTMENT_MODEL_SPECS,
        CATEGORICAL_ADJUSTMENT_COLUMNS,
        NUMERIC_ADJUSTMENT_COLUMNS,
        OUTCOMES_BY_NAME,
        SIGNAL_SPECS,
        AdjustmentModelSpec,
        SignalSpec,
    )


MAX_CATEGORICAL_LEVELS = 25


def _prepare_categorical_series(series: pd.Series) -> pd.Series:
    values = series.where(series.notna(), "unknown").astype(str).str.strip().replace("", "unknown")
    counts = values.value_counts(dropna=False)
    if len(counts) <= MAX_CATEGORICAL_LEVELS:
        return values
    keep = set(counts.head(MAX_CATEGORICAL_LEVELS - 1).index)
    return values.where(values.isin(keep), "OTHER")


def _prepare_model_frame(
    df: pd.DataFrame,
    spec: SignalSpec,
    outcome_col: str,
    model_spec: AdjustmentModelSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, tuple[str, ...]]:
    subset = df[df[spec.suspect_column].fillna(False).astype(bool)].copy()
    if spec.exclude_group:
        subset = subset[subset[spec.group_column] != spec.exclude_group].copy()
    subset = subset.dropna(subset=[outcome_col]).copy()

    y = subset[outcome_col].fillna(False).astype(int)
    exposure = subset[spec.exposure_column].fillna(False).astype(int).rename(spec.exposure_column)
    design = pd.DataFrame(index=subset.index)
    design[spec.exposure_column] = exposure.astype(float)

    used_covariates: list[str] = []
    for col in model_spec.covariates:
        if col not in subset.columns:
            continue
        if col in CATEGORICAL_ADJUSTMENT_COLUMNS:
            values = _prepare_categorical_series(subset[col])
            if values.nunique(dropna=False) <= 1:
                continue
            dummies = pd.get_dummies(values, prefix=col, drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
            used_covariates.append(col)
        elif col in NUMERIC_ADJUSTMENT_COLUMNS:
            values = pd.to_numeric(subset[col], errors="coerce")
            median = values.median()
            if pd.isna(median):
                continue
            filled = values.fillna(median).astype(float)
            std = float(filled.std())
            if std == 0 or np.isnan(std):
                continue
            design[f"{col}_standardized"] = (filled - float(filled.mean())) / std
            used_covariates.append(col)
        else:
            values = subset[col].fillna(False).astype(bool).astype(float)
            if values.nunique(dropna=False) <= 1:
                continue
            design[col] = values
            used_covariates.append(col)

    non_constant_cols = [col for col in design.columns if design[col].nunique(dropna=False) > 1]
    design = design[non_constant_cols].copy()
    if spec.exposure_column not in design.columns:
        raise ValueError(f"Exposure column is constant in adjusted model: {spec.exposure_column}")
    design = sm.add_constant(design, has_constant="add")
    return design, y, subset["caseid"].astype(str), tuple(used_covariates)


def _fit_logit(design: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    model = sm.GLM(
        y.to_numpy(dtype=float),
        design.to_numpy(dtype=float),
        family=sm.families.Binomial(),
    )
    try:
        result = model.fit(maxiter=200, disp=False)
        converged = bool(result.converged)
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

    for spec in SIGNAL_SPECS:
        for outcome_name in spec.outcome_names:
            outcome = OUTCOMES_BY_NAME[outcome_name]
            for model_spec in ADJUSTMENT_MODEL_SPECS:
                design, y, caseids, used_covariates = _prepare_model_frame(
                    df, spec, outcome.column, model_spec
                )
                model_df, diagnostics = _fit_logit(design, y)
                model_df.insert(0, "analysis_tier", spec.tier)
                model_df.insert(1, "analysis", spec.analysis)
                model_df.insert(2, "comparison", spec.comparison)
                model_df.insert(3, "outcome_name", outcome.name)
                model_df.insert(4, "outcome_definition", outcome.label)
                model_df.insert(5, "model", model_spec.name)
                model_df.insert(6, "model_label", model_spec.label)
                model_df["exposure_term"] = spec.exposure_column
                model_df["is_exposure_term"] = model_df["term"].eq(spec.exposure_column)
                model_df["n_cases"] = int(len(caseids))
                model_df["n_outcome"] = int(y.sum())
                model_df["used_covariates"] = ";".join(used_covariates)
                model_df["optimization_success"] = diagnostics["optimization_success"]
                model_df["optimization_message"] = diagnostics["optimization_message"]
                result_frames.append(model_df)
                qc_rows.append(
                    {
                        "section": "adjusted",
                        "analysis_tier": spec.tier,
                        "analysis": spec.analysis,
                        "outcome_name": outcome.name,
                        "model": model_spec.name,
                        "model_label": model_spec.label,
                        "n_cases": int(len(caseids)),
                        "n_outcome": int(y.sum()),
                        "n_exposed": int(design[spec.exposure_column].sum()),
                        "n_terms": int(design.shape[1]),
                        "requested_covariates": ";".join(model_spec.covariates),
                        "used_covariates": ";".join(used_covariates),
                        **diagnostics,
                    }
                )

    result_df = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    qc_df = pd.DataFrame(qc_rows)
    return result_df, qc_df
