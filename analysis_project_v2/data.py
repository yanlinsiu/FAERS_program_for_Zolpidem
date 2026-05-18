from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import BOOL_COLUMNS, GLOBAL_DATASET_DIR


@dataclass(frozen=True)
class DatasetBundle:
    period_token: str
    signal_file: Path
    feature_file: Path


def _extract_token(path: Path, prefix: str, suffix: str = "") -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected dataset file name: {path.name}")
    token = stem[len(prefix) :]
    if suffix:
        if not token.endswith(suffix):
            raise ValueError(f"Unexpected dataset file name: {path.name}")
        token = token[: -len(suffix)]
    return token


def _token_sort_key(token: str) -> tuple[int, int, int, str]:
    parts = token.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        start_year = int(parts[0])
        end_year = int(parts[1])
        return (end_year - start_year, end_year, -start_year, token)
    return (0, 0, 0, token)


def resolve_dataset_bundle(
    dataset_dir: str | Path = GLOBAL_DATASET_DIR,
    period_token: str | None = None,
) -> DatasetBundle:
    dataset_path = Path(dataset_dir)
    signal_files = sorted(dataset_path.glob("signal_dataset_*.parquet"))
    feature_files = sorted(dataset_path.glob("drug_feature_*_case.parquet"))
    if not signal_files:
        raise FileNotFoundError(f"No signal_dataset_*.parquet files found in {dataset_path}")
    if not feature_files:
        raise FileNotFoundError(f"No drug_feature_*_case.parquet files found in {dataset_path}")

    signal_by_token = {_extract_token(path, "signal_dataset_"): path for path in signal_files}
    feature_by_token = {
        _extract_token(path, "drug_feature_", "_case"): path for path in feature_files
    }
    shared_tokens = sorted(set(signal_by_token) & set(feature_by_token))
    if not shared_tokens:
        raise RuntimeError(f"No matching signal/feature bundle found in {dataset_path}")

    selected_token = period_token or max(shared_tokens, key=_token_sort_key)
    if selected_token not in signal_by_token or selected_token not in feature_by_token:
        raise FileNotFoundError(f"Period token not found in {dataset_path}: {selected_token}")

    return DatasetBundle(
        period_token=selected_token,
        signal_file=signal_by_token[selected_token],
        feature_file=feature_by_token[selected_token],
    )


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
    if "is_fall_broad" not in normalized.columns and "is_fall_narrow" in normalized.columns:
        normalized["is_fall_broad"] = normalized["is_fall_narrow"]
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
