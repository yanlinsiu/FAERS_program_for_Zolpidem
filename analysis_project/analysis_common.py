from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd
from scipy.special import digamma
from scipy.stats import gamma

DEFAULT_SIGNAL_ROOT = Path(r"D:\program_FAERS\OUTPUT")
DEFAULT_ANALYSIS_ROOT = Path(r"D:\program_FAERS\OUTPUT\analysis")

OUTCOME_SPECS = [
    {
        "outcome_name": "strict_fall",
        "outcome_col": "is_fall_narrow",
        "outcome_label": "Narrow fall definition",
    },
    {
        "outcome_name": "broad_fall",
        "outcome_col": "is_fall_broad",
        "outcome_label": "Broad fall-related definition",
    },
]

STRATUM_SPECS = [
    ("age_group", "65-74", "Age 65-74"),
    ("age_group", "75-84", "Age 75-84"),
    ("age_group", ">=85", "Age >=85"),
    ("sex_clean", "F", "Female"),
    ("sex_clean", "M", "Male"),
    ("serious", True, "Serious outcome"),
    ("polypharmacy_5", True, "Polypharmacy >=5"),
]


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir else DEFAULT_ANALYSIS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_input_path(path_value: str | Path | None, default_path: Path) -> Path:
    return Path(path_value) if path_value else default_path


def list_signal_files(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> list[Path]:
    if signal_file:
        return [Path(signal_file)]

    root = _normalize_input_path(signal_root, DEFAULT_SIGNAL_ROOT)
    if root.is_file():
        return [root]
    return sorted(root.glob("signal_dataset_*.parquet"))


def list_feature_files(
    signal_root: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> list[Path]:
    if feature_file:
        return [Path(feature_file)]

    root = _normalize_input_path(signal_root, DEFAULT_SIGNAL_ROOT)
    if root.is_file():
        return [root]
    return sorted(root.glob("drug_feature_*_case.parquet"))


def _extract_period_from_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("signal_dataset_"):
        return stem.replace("signal_dataset_", "")
    if stem.startswith("drug_feature_") and stem.endswith("_case"):
        return stem.replace("drug_feature_", "").replace("_case", "")
    return stem


def load_signal_dataset(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> pd.DataFrame:
    files = list_signal_files(signal_root=signal_root, signal_file=signal_file)
    if not files:
        raise FileNotFoundError("No signal_dataset_*.parquet files found.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_parquet(file_path)
        df = df.copy()
        df["dataset_period"] = _extract_period_from_name(file_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["caseid"] = combined["caseid"].astype(str).str.strip()
    combined = combined[combined["caseid"] != ""].copy()
    if "is_fall_narrow" not in combined.columns and "is_fall" in combined.columns:
        combined["is_fall_narrow"] = combined["is_fall"].fillna(False).astype(bool)
    if "is_fall_broad" not in combined.columns and "is_fall_narrow" in combined.columns:
        combined["is_fall_broad"] = combined["is_fall_narrow"].fillna(False).astype(bool)
    if "serious" in combined.columns:
        combined["serious"] = combined["serious"].fillna(False).astype(bool)
    return combined


def load_feature_dataset(
    signal_root: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> pd.DataFrame:
    files = list_feature_files(signal_root=signal_root, feature_file=feature_file)
    if not files:
        raise FileNotFoundError("No drug_feature_*_case.parquet files found.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_parquet(file_path)
        df = df.copy()
        df["dataset_period"] = _extract_period_from_name(file_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["caseid"] = combined["caseid"].astype(str).str.strip()
    combined = combined[combined["caseid"] != ""].copy()
    return combined


def merge_signal_and_feature(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> pd.DataFrame:
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)
    feature_df = load_feature_dataset(signal_root=signal_root, feature_file=feature_file)
    merged = signal_df.merge(feature_df, on=["caseid", "dataset_period"], how="left")

    feature_bool_cols = [
        "is_zolpidem",
        "is_zolpidem_any",
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
        "serious",
        "is_fall_narrow",
        "is_fall_broad",
    ]
    for col in feature_bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    for col in ["drug_n", "distinct_drug_n"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    return merged


def feature_mask(df: pd.DataFrame, column: str, value) -> pd.Series:
    if isinstance(value, bool):
        return df[column].fillna(False).astype(bool).eq(value)
    return df[column].astype(str).str.strip().eq(str(value))


def summarize_missing(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            summary[f"missing_{col}"] = -1
            continue
        missing_mask = df[col].isna()
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            missing_mask = missing_mask | (df[col].astype(str).str.strip() == "")
        summary[f"missing_{col}"] = int(missing_mask.sum())
    return summary


def two_by_two_counts(exposed: pd.Series, outcome: pd.Series) -> dict[str, int]:
    exp = exposed.fillna(False).astype(bool)
    out = outcome.fillna(False).astype(bool)
    a = int((exp & out).sum())
    b = int((exp & ~out).sum())
    c = int((~exp & out).sum())
    d = int((~exp & ~out).sum())
    return {"a": a, "b": b, "c": c, "d": d}


def _wald_ci_from_log_estimate(log_estimate: float, se: float) -> tuple[float | None, float | None]:
    if math.isnan(log_estimate) or math.isnan(se) or math.isinf(se):
        return (None, None)
    lower = math.exp(log_estimate - 1.96 * se)
    upper = math.exp(log_estimate + 1.96 * se)
    return (lower, upper)


def _safe_log2(value: float | None) -> float | None:
    if value is None or value <= 0 or math.isnan(value):
        return None
    return math.log(value, 2)


def _gamma_quantile_scores(
    shape: float,
    rate: float,
    lower_prob: float,
    upper_prob: float,
) -> tuple[float | None, float | None]:
    if shape <= 0 or rate <= 0:
        return (None, None)
    scale = 1.0 / rate
    lower = float(gamma.ppf(lower_prob, a=shape, scale=scale))
    upper = float(gamma.ppf(upper_prob, a=shape, scale=scale))
    if math.isnan(lower) or math.isnan(upper):
        return (None, None)
    return (lower, upper)


def _ic_from_observed_expected(
    observed: int,
    expected: float,
    shrinkage: float = 0.5,
    credibility_level: float = 0.95,
) -> dict[str, float | None]:
    alpha = 1.0 - credibility_level
    lower_prob = alpha / 2.0
    upper_prob = 1.0 - (alpha / 2.0)

    shape = float(observed) + float(shrinkage)
    rate = float(expected) + float(shrinkage)
    if shape <= 0 or rate <= 0:
        return {
            "ic": None,
            "ic025": None,
            "ic975": None,
        }

    ic = _safe_log2(shape / rate)
    posterior_low, posterior_high = _gamma_quantile_scores(
        shape=shape,
        rate=rate,
        lower_prob=lower_prob,
        upper_prob=upper_prob,
    )
    return {
        "ic": ic,
        "ic025": _safe_log2(posterior_low),
        "ic975": _safe_log2(posterior_high),
    }


def _ebgm_from_observed_expected(
    observed: int,
    expected: float,
    prior_shape: float = 1.0,
    prior_rate: float = 1.0,
    credibility_level: float = 0.90,
) -> dict[str, float | None]:
    alpha = 1.0 - credibility_level
    lower_prob = alpha / 2.0
    upper_prob = 1.0 - (alpha / 2.0)

    shape = float(observed) + float(prior_shape)
    rate = float(expected) + float(prior_rate)
    if shape <= 0 or rate <= 0:
        return {
            "ebgm": None,
            "eb05": None,
            "eb95": None,
        }

    ebgm = math.exp(float(digamma(shape))) / rate
    eb05, eb95 = _gamma_quantile_scores(
        shape=shape,
        rate=rate,
        lower_prob=lower_prob,
        upper_prob=upper_prob,
    )
    return {
        "ebgm": ebgm,
        "eb05": eb05,
        "eb95": eb95,
    }


def ror_prr_from_counts(a: int, b: int, c: int, d: int) -> dict[str, float | int | None]:
    n = a + b + c + d
    result: dict[str, float | int | None] = {"a": a, "b": b, "c": c, "d": d, "n": n}

    result["reporting_rate_exposed"] = a / (a + b) if (a + b) else None
    result["reporting_rate_unexposed"] = c / (c + d) if (c + d) else None
    expected = ((a + b) * (a + c) / n) if n else 0.0

    if n > 0:
        # Apply Haldane-Anscombe continuity correction when any cell is zero.
        # This avoids dropping otherwise informative signals (e.g., b=0 or c=0).
        use_continuity_correction = any(x == 0 for x in [a, b, c, d])
        correction = 0.5 if use_continuity_correction else 0.0
        a_eff = a + correction
        b_eff = b + correction
        c_eff = c + correction
        d_eff = d + correction

        ror = (a_eff * d_eff) / (b_eff * c_eff)
        se_log_ror = math.sqrt((1 / a_eff) + (1 / b_eff) + (1 / c_eff) + (1 / d_eff))
        ror_ci_low, ror_ci_high = _wald_ci_from_log_estimate(math.log(ror), se_log_ror)

        prr = (a_eff / (a_eff + b_eff)) / (c_eff / (c_eff + d_eff))
        se_log_prr = math.sqrt((1 / a_eff) - (1 / (a_eff + b_eff)) + (1 / c_eff) - (1 / (c_eff + d_eff)))
        prr_ci_low, prr_ci_high = _wald_ci_from_log_estimate(math.log(prr), se_log_prr)
    else:
        ror = None
        ror_ci_low = None
        ror_ci_high = None
        prr = None
        prr_ci_low = None
        prr_ci_high = None

    chi_square_yates = 0.0
    if (a + b) and (c + d) and (a + c) and (b + d):
        numerator = abs((a * d) - (b * c)) - (n / 2)
        chi_square_yates = n * max(numerator, 0) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))

    ic_metrics = _ic_from_observed_expected(observed=a, expected=expected)
    ebgm_metrics = _ebgm_from_observed_expected(observed=a, expected=expected)

    result.update(
        {
            "ror": ror,
            "ror_ci_low": ror_ci_low,
            "ror_ci_high": ror_ci_high,
            "prr": prr,
            "prr_ci_low": prr_ci_low,
            "prr_ci_high": prr_ci_high,
            "chi_square_yates": chi_square_yates,
            "expected_a": expected,
            **ic_metrics,
            **ebgm_metrics,
            "signal_flag_mhra": bool(a >= 3 and prr is not None and prr >= 2 and chi_square_yates >= 4),
            "signal_flag_ror": bool(ror_ci_low is not None and ror_ci_low > 1),
            "signal_flag_ic": bool(ic_metrics["ic025"] is not None and ic_metrics["ic025"] > 0),
            "signal_flag_ebgm": bool(a >= 3 and ebgm_metrics["eb05"] is not None and ebgm_metrics["eb05"] >= 2),
        }
    )
    return result


def describe_signal(metrics: dict[str, float | int | None]) -> str:
    if (
        metrics["signal_flag_ror"]
        or metrics["signal_flag_mhra"]
        or metrics.get("signal_flag_ic", False)
        or metrics.get("signal_flag_ebgm", False)
    ):
        return "signal_detected"
    return "no_clear_signal"


def format_metric(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


def build_stratified_rows(
    df: pd.DataFrame,
    analysis_name: str,
    exposure_col: str,
    outcome_col: str,
    outcome_name: str,
    outcome_label: str,
    stratum_specs: list[tuple[str, object, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    specs = STRATUM_SPECS if stratum_specs is None else stratum_specs
    for column, value, label in specs:
        if column not in df.columns:
            continue
        mask = feature_mask(df, column, value)
        subset = df[mask].copy()
        if subset.empty:
            continue
        counts = two_by_two_counts(subset[exposure_col], subset[outcome_col])
        metrics = ror_prr_from_counts(**counts)
        rows.append(
            {
                "analysis": analysis_name,
                "outcome_name": outcome_name,
                "outcome_label": outcome_label,
                "stratum_col": column,
                "stratum_value": value,
                "stratum_label": label,
                "n_in_stratum": int(len(subset)),
                "n_outcome": int(subset[outcome_col].fillna(False).astype(bool).sum()),
                "conclusion": describe_signal(metrics),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def make_overall_qc(df: pd.DataFrame, label: str, exposure_col: str, outcome_col: str) -> pd.DataFrame:
    qc = {
        "analysis": label,
        "outcome_col": outcome_col,
        "n_total": int(len(df)),
        "n_exposed": int(df[exposure_col].fillna(False).astype(bool).sum()),
        "n_unexposed": int((~df[exposure_col].fillna(False).astype(bool)).sum()),
        "n_outcome": int(df[outcome_col].fillna(False).astype(bool).sum()),
        "n_exposed_outcome": int((df[exposure_col].fillna(False).astype(bool) & df[outcome_col].fillna(False).astype(bool)).sum()),
    }
    qc.update(summarize_missing(df, ["age_group", "sex_clean", "serious", "fall_pt_list"]))
    return pd.DataFrame([qc])


def save_tables(result_df: pd.DataFrame, qc_df: pd.DataFrame, result_path: Path, qc_path: Path) -> None:
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")
