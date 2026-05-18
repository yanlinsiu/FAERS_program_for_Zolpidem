from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import digamma
from scipy.stats import chi2, gamma

from config import MIN_EXPOSED_CASES, MIN_EXPOSED_OUTCOME_CASES


def two_by_two_counts(exposed: pd.Series, outcome: pd.Series) -> dict[str, int]:
    exp = exposed.fillna(False).astype(bool)
    out = outcome.fillna(False).astype(bool)
    a = int((exp & out).sum())
    b = int((exp & ~out).sum())
    c = int((~exp & out).sum())
    d = int((~exp & ~out).sum())
    return {"a": a, "b": b, "c": c, "d": d}


def _safe_exp(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return float(math.exp(max(min(value, 50.0), -50.0)))


def _safe_log2(value: float | None) -> float | None:
    if value is None or value <= 0 or math.isnan(value):
        return None
    return float(math.log(value, 2))


def _wald_ci(log_estimate: float, se: float) -> tuple[float | None, float | None]:
    if math.isnan(log_estimate) or math.isnan(se) or math.isinf(se):
        return (None, None)
    return (
        _safe_exp(log_estimate - 1.96 * se),
        _safe_exp(log_estimate + 1.96 * se),
    )


def _gamma_interval(
    shape: float,
    rate: float,
    lower_prob: float,
    upper_prob: float,
) -> tuple[float | None, float | None]:
    if shape <= 0 or rate <= 0:
        return (None, None)
    lower = float(gamma.ppf(lower_prob, a=shape, scale=1.0 / rate))
    upper = float(gamma.ppf(upper_prob, a=shape, scale=1.0 / rate))
    if math.isnan(lower) or math.isnan(upper):
        return (None, None)
    return (lower, upper)


def _ic_metrics(observed: int, expected: float) -> dict[str, float | None]:
    shape = observed + 0.5
    rate = expected + 0.5
    low, high = _gamma_interval(shape, rate, 0.025, 0.975)
    return {
        "ic": _safe_log2(shape / rate),
        "ic025": _safe_log2(low),
        "ic975": _safe_log2(high),
    }


def _ebgm_metrics(observed: int, expected: float) -> dict[str, float | None]:
    shape = observed + 1.0
    rate = expected + 1.0
    low, high = _gamma_interval(shape, rate, 0.05, 0.95)
    return {
        "ebgm": float(math.exp(float(digamma(shape))) / rate) if rate > 0 else None,
        "eb05": low,
        "eb95": high,
    }


def signal_metrics(a: int, b: int, c: int, d: int) -> dict[str, Any]:
    n = a + b + c + d
    exposed_n = a + b
    unexposed_n = c + d
    expected = ((a + b) * (a + c) / n) if n else 0.0

    if n > 0:
        correction = 0.5 if any(cell == 0 for cell in [a, b, c, d]) else 0.0
        a_eff, b_eff, c_eff, d_eff = a + correction, b + correction, c + correction, d + correction
        ror = (a_eff * d_eff) / (b_eff * c_eff)
        se_log_ror = math.sqrt((1 / a_eff) + (1 / b_eff) + (1 / c_eff) + (1 / d_eff))
        ror_ci_low, ror_ci_high = _wald_ci(math.log(ror), se_log_ror)
        prr = (a_eff / (a_eff + b_eff)) / (c_eff / (c_eff + d_eff))
        se_log_prr = math.sqrt(
            (1 / a_eff)
            - (1 / (a_eff + b_eff))
            + (1 / c_eff)
            - (1 / (c_eff + d_eff))
        )
        prr_ci_low, prr_ci_high = _wald_ci(math.log(prr), se_log_prr)
    else:
        ror = prr = ror_ci_low = ror_ci_high = prr_ci_low = prr_ci_high = None

    chi_square_yates = 0.0
    if exposed_n and unexposed_n and (a + c) and (b + d):
        numerator = abs((a * d) - (b * c)) - (n / 2)
        chi_square_yates = n * max(numerator, 0) ** 2 / (
            exposed_n * unexposed_n * (a + c) * (b + d)
        )
    p_value = float(chi2.sf(chi_square_yates, df=1)) if n else None
    ic = _ic_metrics(a, expected)
    ebgm = _ebgm_metrics(a, expected)

    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "n": n,
        "exposed_n": exposed_n,
        "unexposed_n": unexposed_n,
        "reporting_rate_exposed": a / exposed_n if exposed_n else None,
        "reporting_rate_unexposed": c / unexposed_n if unexposed_n else None,
        "ror": ror,
        "ror_ci_low": ror_ci_low,
        "ror_ci_high": ror_ci_high,
        "prr": prr,
        "prr_ci_low": prr_ci_low,
        "prr_ci_high": prr_ci_high,
        "chi_square_yates": chi_square_yates,
        "p_value": p_value,
        "expected_a": expected,
        **ic,
        **ebgm,
        "signal_flag_mhra": bool(a >= 3 and prr is not None and prr >= 2 and chi_square_yates >= 4),
        "signal_flag_ror": bool(ror_ci_low is not None and ror_ci_low > 1),
        "signal_flag_ic": bool(ic["ic025"] is not None and ic["ic025"] > 0),
        "signal_flag_ebgm": bool(a >= 3 and ebgm["eb05"] is not None and ebgm["eb05"] >= 2),
    }


def add_signal_classification(df: pd.DataFrame, apply_stability_gate: bool) -> pd.DataFrame:
    result = df.copy()
    signal_cols = ["signal_flag_mhra", "signal_flag_ror", "signal_flag_ic", "signal_flag_ebgm"]
    for col in signal_cols:
        if col not in result.columns:
            result[col] = False
    result["raw_signal_detected"] = result[signal_cols].fillna(False).any(axis=1)
    result["stability_status"] = np.where(
        (result["a"] >= MIN_EXPOSED_OUTCOME_CASES) & (result["exposed_n"] >= MIN_EXPOSED_CASES),
        "stable",
        "unstable",
    )
    if apply_stability_gate:
        result["is_stable_signal"] = result["raw_signal_detected"] & result["stability_status"].eq("stable")
    else:
        result["is_stable_signal"] = result["raw_signal_detected"]
    result["conclusion"] = np.where(result["is_stable_signal"], "signal_detected", "no_stable_signal")
    return result


def apply_bh_fdr(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    result = df.copy()
    result["fdr_q_value"] = pd.NA
    result["fdr_significant"] = False
    valid = result[p_col].notna()
    if not valid.any():
        return result

    p_values = result.loc[valid, p_col].astype(float).to_numpy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    cumulative = 1.0
    for idx in range(m - 1, -1, -1):
        cumulative = min(cumulative, ranked[idx] * m / (idx + 1))
        adjusted[idx] = cumulative
    q_values = np.empty(m, dtype=float)
    q_values[order] = np.clip(adjusted, 0.0, 1.0)
    result.loc[valid, "fdr_q_value"] = q_values
    result.loc[valid, "fdr_significant"] = q_values < 0.05
    return result


def feature_mask(df: pd.DataFrame, column: str, value: object) -> pd.Series:
    if isinstance(value, bool):
        return df[column].fillna(False).astype(bool).eq(value)
    return df[column].astype(str).str.strip().eq(str(value))
