from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RAW_ROOT
from utils import (
    attach_caseid_from_demo,
    build_file_path,
    load_retained_demo_primaryids,
    read_faers_txt,
)


CASE_LINKED_TABLES: dict[str, dict[str, object]] = {
    "INDI": {
        "output_stem": "indi",
        "columns": ["primaryid", "caseid", "indi_pt"],
        "text_columns": ["indi_pt"],
    },
    "RPSR": {
        "output_stem": "rpsr",
        "columns": ["primaryid", "caseid", "rpsr_cod"],
        "text_columns": ["rpsr_cod"],
    },
    "THER": {
        "output_stem": "ther",
        "columns": ["primaryid", "caseid", "start_dt", "end_dt", "dur", "dur_cod"],
        "text_columns": ["start_dt", "end_dt", "dur", "dur_cod"],
    },
}


def _empty_case_linked_frame(table_name: str) -> pd.DataFrame:
    spec = CASE_LINKED_TABLES[table_name]
    return pd.DataFrame(columns=list(spec["columns"]))


def process_case_linked_table(
    year: int,
    quarter: str,
    output_root: str | Path,
    table_name: str,
) -> pd.DataFrame:
    table_name = table_name.upper()
    if table_name not in CASE_LINKED_TABLES:
        raise ValueError(f"Unsupported case-linked table: {table_name}")

    spec = CASE_LINKED_TABLES[table_name]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"{spec['output_stem']}_{year}{str(quarter).lower()}_case.parquet"

    file_path = build_file_path(RAW_ROOT, year, quarter, table_name)
    if not file_path.exists():
        empty_df = _empty_case_linked_frame(table_name)
        empty_df.to_parquet(output_file, index=False)
        print(f"{table_name} file not found; saved empty dataset: {output_file}")
        return empty_df

    df = read_faers_txt(file_path, dataset_name=table_name)
    df = attach_caseid_from_demo(df, RAW_ROOT, year, quarter, output_root=output_root)

    for column in spec["columns"]:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[list(spec["columns"])].copy()
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()

    retained_primaryids = load_retained_demo_primaryids(
        RAW_ROOT, year, quarter, output_root=output_root
    )
    df = df[df["primaryid"].isin(retained_primaryids)]
    df = df[df["caseid"] != ""].copy()

    for column in spec["text_columns"]:
        df[column] = df[column].where(df[column].notna(), "").astype(str).str.strip()

    value_columns = [column for column in spec["columns"] if column not in {"primaryid", "caseid"}]
    if value_columns:
        has_value = pd.Series(False, index=df.index)
        for column in value_columns:
            has_value = has_value | df[column].astype(str).str.strip().ne("")
        df = df[has_value].copy()

    df.to_parquet(output_file, index=False)
    print(f"{table_name} rows: {len(df)}")
    print(f"saved: {output_file}")
    return df


def process_indi(year: int, quarter: str, output_root: str | Path) -> pd.DataFrame:
    return process_case_linked_table(year, quarter, output_root, "INDI")


def process_rpsr(year: int, quarter: str, output_root: str | Path) -> pd.DataFrame:
    return process_case_linked_table(year, quarter, output_root, "RPSR")


def process_ther(year: int, quarter: str, output_root: str | Path) -> pd.DataFrame:
    return process_case_linked_table(year, quarter, output_root, "THER")
