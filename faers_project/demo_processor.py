from pathlib import Path

import pandas as pd

from config import RAW_ROOT
from utils import (
    apply_demo_demographic_criteria,
    build_file_path,
    deduplicate_demo_records,
    read_faers_txt,
)


CASE_BASE_COLUMNS = [
    "caseid",
    "primaryid",
    "fda_dt",
    "age_years",
    "age_group",
    "sex_clean",
    "serious",
    "year",
    "quarter",
    "year_quarter",
]


def _build_case_base_dataset(df: pd.DataFrame, year: int, quarter: str) -> pd.DataFrame:
    quarter_upper = str(quarter).upper()
    year_int = int(year)

    case_base_df = pd.DataFrame(index=df.index)
    case_base_df["caseid"] = (
        df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    )
    case_base_df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    case_base_df["fda_dt"] = pd.to_datetime(df["fda_dt"], errors="coerce")
    case_base_df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    case_base_df["age_group"] = (
        df["age_group"].where(df["age_group"].notna(), "").astype(str).str.strip()
    )
    case_base_df["sex_clean"] = (
        df["sex_clean"].where(df["sex_clean"].notna(), "").astype(str).str.strip()
    )

    if "serious" in df.columns:
        case_base_df["serious"] = df["serious"]
    else:
        case_base_df["serious"] = pd.NA

    case_base_df["year"] = year_int
    case_base_df["quarter"] = quarter_upper
    case_base_df["year_quarter"] = f"{year_int}{quarter_upper}"

    case_base_df = case_base_df[CASE_BASE_COLUMNS]
    case_base_df = case_base_df[case_base_df["caseid"] != ""].copy()
    case_base_df = case_base_df.drop_duplicates(subset="caseid", keep="last")

    return case_base_df


def process_demo(year, quarter, output_root):
    file_path = build_file_path(RAW_ROOT, year, quarter, "DEMO")
    print(f"processing file: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    df = read_faers_txt(file_path)
    print("raw rows:", len(df))

    df = deduplicate_demo_records(df)
    print("rows after DEMO dedup:", len(df))

    pre_filter_n = len(df)
    df = apply_demo_demographic_criteria(df)
    print("rows after demographic criteria:", len(df))
    print("rows removed by criteria:", pre_filter_n - len(df))

    case_base_df = _build_case_base_dataset(df, year=year, quarter=quarter)
    print("case_base_dataset rows:", len(case_base_df))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"case_base_dataset_{year}{str(quarter).lower()}.parquet"
    case_base_df.to_parquet(output_file, index=False)

    # Keep legacy filename for downstream continuity.
    legacy_output_file = output_root / f"demo_{year}{str(quarter).lower()}.parquet"
    case_base_df.to_parquet(legacy_output_file, index=False)

    print(f"saved: {output_file}")
    print(f"saved (legacy): {legacy_output_file}")
    return case_base_df
