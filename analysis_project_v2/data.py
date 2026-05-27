from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import DatasetBundle, resolve_signal_feature_bundle
from common.schema_checks import validate_feature_schema, validate_signal_schema

try:
    from .config import BOOL_COLUMNS, CATEGORICAL_ADJUSTMENT_COLUMNS, GLOBAL_DATASET_DIR
except ImportError:
    from config import BOOL_COLUMNS, CATEGORICAL_ADJUSTMENT_COLUMNS, GLOBAL_DATASET_DIR


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
    if "polypharmacy" not in normalized.columns and "polypharmacy_5" in normalized.columns:
        normalized["polypharmacy"] = normalized["polypharmacy_5"]

    for col in BOOL_COLUMNS:
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna(False).astype(bool)

    for col in ["year", "age_years", "drug_n", "distinct_drug_n", "indi_n", "distinct_indi_n"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    categorical_cols = {
        "target_drug_group",
        "target_drug_group_ps",
        *CATEGORICAL_ADJUSTMENT_COLUMNS,
    }
    for col in categorical_cols:
        if col in normalized.columns:
            normalized[col] = (
                normalized[col]
                .where(normalized[col].notna(), "unknown")
                .astype(str)
                .str.strip()
                .replace("", "unknown")
            )
    return normalized


def _resolve_ml_feature_v2_file(period_token: str) -> Path | None:
    feature_v2_file = (
        PROJECT_ROOT
        / "OUTPUT_ML"
        / "features_v2"
        / "datasets"
        / f"ml_feature_v2_{period_token}.parquet"
    )
    return feature_v2_file if feature_v2_file.exists() else None


def load_analysis_frame(bundle: DatasetBundle) -> pd.DataFrame:
    signal_df = _normalize_caseid(pd.read_parquet(bundle.signal_file), "signal dataset")
    feature_df = _normalize_caseid(pd.read_parquet(bundle.feature_file), "feature dataset")
    validate_signal_schema(signal_df)
    validate_feature_schema(feature_df)

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

    feature_v2_file = _resolve_ml_feature_v2_file(bundle.period_token)
    if feature_v2_file is not None:
        feature_v2_df = _normalize_caseid(pd.read_parquet(feature_v2_file), "ML feature v2 dataset")
        extra_cols = [col for col in feature_v2_df.columns if col == "caseid" or col not in merged.columns]
        if len(extra_cols) > 1:
            merged = merged.merge(
                feature_v2_df[extra_cols],
                on="caseid",
                how="left",
                indicator="_feature_v2_merge",
            )
            missing_feature_v2_rows = int(merged["_feature_v2_merge"].eq("left_only").sum())
            if missing_feature_v2_rows:
                raise ValueError(
                    f"Merged dataset has missing ML feature v2 rows: {missing_feature_v2_rows}"
                )
            merged = merged.drop(columns=["_feature_v2_merge"])

    merged = _normalize_columns(merged)
    return merged
