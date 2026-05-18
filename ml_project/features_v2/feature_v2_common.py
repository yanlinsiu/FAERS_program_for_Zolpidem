from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAERS_PROJECT_DIR = PROJECT_ROOT / "faers_project"
if str(FAERS_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(FAERS_PROJECT_DIR))

from config import RAW_ROOT  # noqa: E402
from utils import (  # noqa: E402
    apply_demo_demographic_criteria,
    attach_caseid_from_demo,
    build_file_path,
    deduplicate_demo_records,
    exclude_deleted_caseids,
    load_retained_demo_primaryids,
    read_faers_txt,
)


OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
OUTPUT_ML_ROOT = PROJECT_ROOT / "OUTPUT_ML"
FEATURE_V2_ROOT = OUTPUT_ML_ROOT / "features_v2"
AUDIT_DIR = FEATURE_V2_ROOT / "audit"
LOOKUP_DIR = FEATURE_V2_ROOT / "lookup"
QUARTERLY_DIR = FEATURE_V2_ROOT / "quarterly"
DATASET_DIR = FEATURE_V2_ROOT / "datasets"
QC_DIR = FEATURE_V2_ROOT / "qc"
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"

TABLES = ("DEMO", "DRUG", "INDI", "RPSR", "THER")
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


def ensure_feature_v2_dirs() -> None:
    for directory in [AUDIT_DIR, LOOKUP_DIR, QUARTERLY_DIR, DATASET_DIR, QC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def iter_quarters(start_year: int, end_year: int) -> Iterable[tuple[int, str]]:
    for year in range(int(start_year), int(end_year) + 1):
        for quarter in QUARTERS:
            try:
                build_file_path(RAW_ROOT, year, quarter, "DEMO")
            except FileNotFoundError:
                continue
            yield year, quarter


def period_token(start_year: int, end_year: int) -> str:
    return f"{int(start_year)}_{int(end_year)}"


def quarter_token(year: int, quarter: str) -> str:
    return f"{int(year)}{str(quarter).lower()}"


def clean_caseid(series: pd.Series) -> pd.Series:
    return series.where(series.notna(), "").astype(str).str.strip()


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.where(series.notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_meddra_term(series: pd.Series) -> pd.Series:
    return (
        normalize_text(series)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def read_raw_table(year: int, quarter: str, table_name: str) -> pd.DataFrame:
    path = build_file_path(RAW_ROOT, year, quarter, table_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing {table_name} file: {path}")
    return read_faers_txt(path, dataset_name=table_name)


def output_quarter_dir(year: int) -> Path:
    return OUTPUT_ROOT / str(int(year)) / "quarterly"


def processed_quarter_file(year: int, quarter: str, stem: str) -> Path:
    return output_quarter_dir(year) / f"{stem}_{quarter_token(year, quarter)}.parquet"


def load_clean_demo(year: int, quarter: str) -> pd.DataFrame:
    df = read_raw_table(year, quarter, "DEMO")
    df, _, _ = exclude_deleted_caseids(df, RAW_ROOT, year, quarter)
    df = deduplicate_demo_records(df)
    df = apply_demo_demographic_criteria(df)
    return df


def load_clean_case_table(year: int, quarter: str, table_name: str) -> pd.DataFrame:
    df = read_raw_table(year, quarter, table_name)
    df = attach_caseid_from_demo(df, RAW_ROOT, year, quarter, output_root=output_quarter_dir(year))
    df["caseid"] = clean_caseid(df["caseid"])
    if "primaryid" in df.columns:
        df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
        retained = load_retained_demo_primaryids(
            RAW_ROOT, year, quarter, output_root=output_quarter_dir(year)
        )
        df = df[df["primaryid"].isin(retained)].copy()
    return df[df["caseid"] != ""].copy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def summarize_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = [
        {
            "dataset": dataset,
            "metric": "n_rows",
            "value": int(len(df)),
        },
        {
            "dataset": dataset,
            "metric": "unique_caseid",
            "value": int(df["caseid"].nunique()) if "caseid" in df.columns else None,
        },
        {
            "dataset": dataset,
            "metric": "duplicate_caseid_rows",
            "value": int(df.duplicated("caseid").sum()) if "caseid" in df.columns else None,
        },
    ]
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        blank = 0
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            blank = int(df[column].fillna("").astype(str).str.strip().eq("").sum())
        rows.append(
            {
                "dataset": dataset,
                "column": column,
                "n_rows": int(len(df)),
                "missing_n": missing,
                "blank_n": blank,
                "missing_or_blank_rate": (missing + blank) / len(df) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def concat_existing(paths: list[Path]) -> pd.DataFrame:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in existing), ignore_index=True)


def latest_meddra_excel() -> Path:
    candidates = sorted(PROJECT_ROOT.glob("MedDRA*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No MedDRA*.xlsx file found in project root.")
    return candidates[0]


def boundary_pattern(terms: list[str]) -> str:
    escaped = sorted({re.escape(term) for term in terms if term}, key=len, reverse=True)
    if not escaped:
        return r"a^"
    return rf"(?<![A-Z0-9])(?:{'|'.join(escaped)})(?![A-Z0-9])"
