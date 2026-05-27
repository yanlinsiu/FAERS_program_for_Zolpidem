from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import DatasetBundle, resolve_signal_feature_bundle

try:
    from .config import BOOL_COLUMNS, GLOBAL_DATASET_DIR
except ImportError:
    from config import BOOL_COLUMNS, GLOBAL_DATASET_DIR


def resolve_dataset_bundle(
    dataset_dir: str | Path = GLOBAL_DATASET_DIR,
    period_token: str | None = None,
) -> DatasetBundle:
    return resolve_signal_feature_bundle(dataset_dir=dataset_dir, period_token=period_token)


def _normalize_caseid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "caseid" not in df.columns:
        raise ValueError(f"{label} is missing required column: caseid")
    normalized = df.copy()
    normalized["caseid"] = normalized["caseid"].where(
        normalized["caseid"].notna(), ""
    ).astype(str).str.strip()
    normalized = normalized[normalized["caseid"] != ""].copy()
    duplicates = int(normalized["caseid"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{label} contains duplicate caseid values: {duplicates}")
    return normalized


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "polypharmacy_5" not in normalized.columns and "polypharmacy" in normalized.columns:
        normalized["polypharmacy_5"] = normalized["polypharmacy"]
    if "polypharmacy" not in normalized.columns and "polypharmacy_5" in normalized.columns:
        normalized["polypharmacy"] = normalized["polypharmacy_5"]

    for col in BOOL_COLUMNS:
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna(False).astype(bool)

    for col in ["year", "drug_n", "distinct_drug_n"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in ["age_group", "sex_clean", "quarter", "target_drug_group", "target_drug_group_ps"]:
        if col in normalized.columns:
            normalized[col] = (
                normalized[col]
                .where(normalized[col].notna(), "unknown")
                .astype(str)
                .str.strip()
                .replace("", "unknown")
            )
    return normalized


def load_analysis_frame(bundle: DatasetBundle) -> pd.DataFrame:
    signal_df = _normalize_caseid(pd.read_parquet(bundle.signal_file), "signal dataset")
    feature_df = _normalize_caseid(pd.read_parquet(bundle.feature_file), "feature dataset")

    merged = signal_df.merge(
        feature_df,
        on="caseid",
        how="left",
        suffixes=("", "_feature"),
        indicator=True,
    )
    missing_feature_rows = int(merged["_merge"].eq("left_only").sum())
    if missing_feature_rows:
        raise ValueError(f"Merged dataset has missing feature rows: {missing_feature_rows}")
    merged = merged.drop(columns=["_merge"])
    merged = _normalize_columns(merged)
    return merged
