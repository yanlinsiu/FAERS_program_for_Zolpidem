from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_SIGNAL_ROOT = Path(r"D:\program_FAERS\OUTPUT")
DEFAULT_ANALYSIS_ROOT = Path(r"D:\program_FAERS\OUTPUT\analysis")

OUTCOME_SPECS = [
    {
        "outcome_name": "strict_fall",
        "outcome_col": "is_fall",
        "outcome_label": "Strict fall definition (PT=FALL/FALLS)",
    },
    {
        "outcome_name": "broad_fall",
        "outcome_col": "has_fall_related_broad",
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


def list_signal_files(signal_root: str | Path | None = None) -> list[Path]:
    root = Path(signal_root) if signal_root else DEFAULT_SIGNAL_ROOT
    return sorted(root.glob("signal_dataset_*.parquet"))


def list_feature_files(signal_root: str | Path | None = None) -> list[Path]:
    root = Path(signal_root) if signal_root else DEFAULT_SIGNAL_ROOT
    return sorted(root.glob("drug_feature_*_case.parquet"))


def _extract_period_from_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("signal_dataset_"):
        return stem.replace("signal_dataset_", "")
    if stem.startswith("drug_feature_") and stem.endswith("_case"):
        return stem.replace("drug_feature_", "").replace("_case", "")
    return stem


def load_signal_dataset(signal_root: str | Path | None = None) -> pd.DataFrame:
    files = list_signal_files(signal_root)
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
    if "has_fall_related_broad" not in combined.columns and "is_fall" in combined.columns:
        combined["has_fall_related_broad"] = combined["is_fall"].fillna(False).astype(bool)
    if "serious" in combined.columns:
        combined["serious"] = combined["serious"].fillna(False).astype(bool)
    return combined


def load_feature_dataset(signal_root: str | Path | None = None) -> pd.DataFrame:
    files = list_feature_files(signal_root)
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


def merge_signal_and_feature(signal_root: str | Path | None = None) -> pd.DataFrame:
    signal_df = load_signal_dataset(signal_root)
    feature_df = load_feature_dataset(signal_root)
    merged = signal_df.merge(feature_df, on=["caseid", "dataset_period"], how="left")

    feature_bool_cols = [
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
        "serious",
        "has_fall_related_broad",
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


def ror_prr_from_counts(a: int, b: int, c: int, d: int) -> dict[str, float | int | None]:
    n = a + b + c + d
    result: dict[str, float | int | None] = {"a": a, "b": b, "c": c, "d": d, "n": n}

    result["reporting_rate_exposed"] = a / (a + b) if (a + b) else None
    result["reporting_rate_unexposed"] = c / (c + d) if (c + d) else None

    if all(x > 0 for x in [a, b, c, d]):
        ror = (a * d) / (b * c)
        se_log_ror = math.sqrt((1 / a) + (1 / b) + (1 / c) + (1 / d))
        ror_ci_low, ror_ci_high = _wald_ci_from_log_estimate(math.log(ror), se_log_ror)
        prr = (a / (a + b)) / (c / (c + d)) if (a + b) and (c + d) and c > 0 else None
        se_log_prr = math.sqrt((1 / a) - (1 / (a + b)) + (1 / c) - (1 / (c + d)))
        prr_ci_low, prr_ci_high = _wald_ci_from_log_estimate(math.log(prr), se_log_prr) if prr else (None, None)
    else:
        ror = None
        ror_ci_low = None
        ror_ci_high = None
        prr = None
        prr_ci_low = None
        prr_ci_high = None

    expected = ((a + b) * (a + c) / n) if n else 0.0
    chi_square_yates = 0.0
    if (a + b) and (c + d) and (a + c) and (b + d):
        numerator = abs((a * d) - (b * c)) - (n / 2)
        chi_square_yates = n * max(numerator, 0) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))

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
            "signal_flag_mhra": bool(a >= 3 and prr is not None and prr >= 2 and chi_square_yates >= 4),
            "signal_flag_ror": bool(ror_ci_low is not None and ror_ci_low > 1),
        }
    )
    return result


def describe_signal(metrics: dict[str, float | int | None]) -> str:
    if metrics["signal_flag_ror"] or metrics["signal_flag_mhra"]:
        return "存在不成比例性信号"
    return "未见明确不成比例性信号"


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
