from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("FAERS_CLEAN_OUTPUT_ROOT", PROJECT_ROOT / "OUTPUT"))
OUTPUT_ML_ROOT = Path(os.environ.get("FAERS_ML_OUTPUT_ROOT", PROJECT_ROOT / "OUTPUT_ML"))
FEATURE_V2_ROOT = OUTPUT_ML_ROOT / "features_v2"
AUDIT_DIR = FEATURE_V2_ROOT / "audit"
LOOKUP_DIR = FEATURE_V2_ROOT / "lookup"
QUARTERLY_DIR = FEATURE_V2_ROOT / "quarterly"
DATASET_DIR = FEATURE_V2_ROOT / "datasets"
QC_DIR = FEATURE_V2_ROOT / "qc"
GLOBAL_DATASET_DIR = Path(
    os.environ.get("FAERS_GLOBAL_DATASET_DIR", PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets")
)

TABLES = ("DEMO", "DRUG", "INDI", "RPSR", "THER")
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


def ensure_feature_v2_dirs() -> None:
    for directory in [AUDIT_DIR, LOOKUP_DIR, QUARTERLY_DIR, DATASET_DIR, QC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def iter_quarters(start_year: int, end_year: int) -> Iterable[tuple[int, str]]:
    for year in range(int(start_year), int(end_year) + 1):
        for quarter in QUARTERS:
            if not processed_case_file(year, quarter, "DEMO").exists():
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


def output_quarter_dir(year: int) -> Path:
    return OUTPUT_ROOT / str(int(year)) / "quarterly"


def processed_quarter_file(year: int, quarter: str, stem: str) -> Path:
    return output_quarter_dir(year) / f"{stem}_{quarter_token(year, quarter)}.parquet"


def processed_case_file(year: int, quarter: str, table_name: str) -> Path:
    table_key = str(table_name).upper()
    stem_by_table = {
        "DEMO": "case_base_dataset",
        "DRUG": "drug",
        "INDI": "indi",
        "RPSR": "rpsr",
        "THER": "ther",
    }
    if table_key not in stem_by_table:
        raise ValueError(f"Unsupported processed case table: {table_name}")
    suffix = "_case" if table_key in {"INDI", "RPSR", "THER"} else ""
    return output_quarter_dir(year) / f"{stem_by_table[table_key]}_{quarter_token(year, quarter)}{suffix}.parquet"


def require_processed_case_file(year: int, quarter: str, table_name: str) -> Path:
    path = processed_case_file(year, quarter, table_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned {table_name} input: {path}. "
            "Run faers_project/year_batch_runner.py first so ML-v2 reads only main cleaned outputs."
        )
    return path


def load_clean_demo(year: int, quarter: str) -> pd.DataFrame:
    return pd.read_parquet(require_processed_case_file(year, quarter, "DEMO"))


def load_clean_case_table(year: int, quarter: str, table_name: str) -> pd.DataFrame:
    df = pd.read_parquet(require_processed_case_file(year, quarter, table_name))
    df["caseid"] = clean_caseid(df["caseid"])
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
