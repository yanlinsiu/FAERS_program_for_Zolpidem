from pathlib import Path

import pandas as pd


def read_faers_txt(file_path):
    """Read FAERS ASCII file with standard settings."""
    # Read raw FAERS columns as strings first so large identifier-like fields
    # (for example lot numbers / NDA numbers) are not inferred as oversized ints.
    df = pd.read_csv(
        file_path,
        sep="$",
        encoding="latin1",
        low_memory=False,
        dtype=str,
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_file_path(raw_root, year, quarter, table_name):
    """Build canonical FAERS raw file path."""
    year_str = str(year)
    quarter_upper = str(quarter).upper()
    table_upper = str(table_name).upper()
    year_short = year_str[-2:]
    filename = f"{table_upper}{year_short}{quarter_upper}.txt"
    return Path(raw_root) / year_str / quarter_upper / "ASCII" / filename


def deduplicate_demo_records(df):
    """Keep one latest DEMO record for each caseid."""
    required_cols = ["caseid", "primaryid", "fda_dt", "caseversion"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DEMO missing required columns: {missing_cols}")

    out_df = df.copy()
    out_df["caseid"] = out_df["caseid"].where(out_df["caseid"].notna(), "").astype(str).str.strip()
    out_df = out_df[out_df["caseid"] != ""].copy()

    out_df["primaryid"] = pd.to_numeric(out_df["primaryid"], errors="coerce")
    out_df["caseversion"] = pd.to_numeric(out_df["caseversion"], errors="coerce")
    out_df["fda_dt"] = pd.to_datetime(out_df["fda_dt"], format="%Y%m%d", errors="coerce")

    out_df = out_df.sort_values(
        by=["caseid", "fda_dt", "caseversion", "primaryid"],
        ascending=[True, True, True, True],
    )
    out_df = out_df.drop_duplicates(subset="caseid", keep="last")
    return out_df


def apply_demo_demographic_criteria(df):
    """
    Standardize age/sex and keep elderly cases only.
    Keeps rows where 65 <= age_years <= 120.
    """
    required_age_cols = ["age", "age_cod"]
    missing_age_cols = [col for col in required_age_cols if col not in df.columns]
    if missing_age_cols:
        raise ValueError(f"DEMO missing age columns: {missing_age_cols}")

    out_df = df.copy()
    age_value = pd.to_numeric(out_df["age"], errors="coerce")
    age_unit = (
        out_df["age_cod"].where(out_df["age_cod"].notna(), "").astype(str).str.strip().str.upper()
    )

    age_years = pd.Series(float("nan"), index=out_df.index)
    age_years.loc[age_unit == "YR"] = age_value.loc[age_unit == "YR"]
    age_years.loc[age_unit == "MON"] = age_value.loc[age_unit == "MON"] / 12
    age_years.loc[age_unit == "WK"] = age_value.loc[age_unit == "WK"] / 52
    age_years.loc[age_unit == "DY"] = age_value.loc[age_unit == "DY"] / 365
    age_years.loc[age_unit == "HR"] = age_value.loc[age_unit == "HR"] / (24 * 365)
    age_years.loc[age_unit == "DEC"] = age_value.loc[age_unit == "DEC"] * 10
    out_df["age_years"] = age_years

    out_df = out_df[out_df["age_years"].between(65, 120, inclusive="both")].copy()

    out_df["age_group"] = pd.cut(
        out_df["age_years"],
        bins=[65, 75, 85, float("inf")],
        labels=["65-74", "75-84", ">=85"],
        right=False,
    ).astype(str)

    sex_raw = pd.Series("", index=out_df.index, dtype="object")
    if "sex" in out_df.columns:
        sex_raw = out_df["sex"].where(out_df["sex"].notna(), "").astype(str).str.strip().str.upper()

    if "gndr_cod" in out_df.columns:
        gndr_raw = (
            out_df["gndr_cod"]
            .where(out_df["gndr_cod"].notna(), "")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        sex_raw = sex_raw.where(sex_raw.isin(["M", "F"]), gndr_raw)

    out_df["sex_clean"] = sex_raw.map({"M": "M", "F": "F"}).fillna("UNK")
    return out_df


def load_retained_demo_primaryids(raw_root, year, quarter, output_root=None):
    """
    Return retained primaryid set using processed DEMO/case_base output if available.
    Fallback reads and processes raw DEMO.
    """
    if output_root is not None:
        output_root = Path(output_root)
        quarter_lower = str(quarter).lower()
        preferred_file = output_root / f"case_base_dataset_{year}{quarter_lower}.parquet"
        legacy_file = output_root / f"demo_{year}{quarter_lower}.parquet"

        for file_path in [preferred_file, legacy_file]:
            if not file_path.exists():
                continue
            demo_df = pd.read_parquet(file_path)
            if "primaryid" not in demo_df.columns:
                raise ValueError(
                    f"DEMO result missing required column ['primaryid']: {file_path}"
                )
            primaryid = pd.to_numeric(demo_df["primaryid"], errors="coerce")
            return set(primaryid.dropna())

    demo_file = build_file_path(raw_root, year, quarter, "DEMO")
    if not demo_file.exists():
        raise FileNotFoundError(f"file not found: {demo_file}")

    demo_df = read_faers_txt(demo_file)
    demo_df = deduplicate_demo_records(demo_df)
    demo_df = apply_demo_demographic_criteria(demo_df)
    primaryid = pd.to_numeric(demo_df["primaryid"], errors="coerce")
    return set(primaryid.dropna())


def iter_quarters(start_year, start_quarter, end_year=None, end_quarter=None):
    """Yield (year, quarter) pairs from start to end, inclusive."""
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    quarter_index = {quarter: i for i, quarter in enumerate(quarter_order)}

    start_year = int(start_year)
    start_quarter = str(start_quarter).upper()
    end_year = start_year if end_year is None else int(end_year)
    end_quarter = start_quarter if end_quarter is None else str(end_quarter).upper()

    if start_quarter not in quarter_index or end_quarter not in quarter_index:
        raise ValueError("quarter must be one of Q1, Q2, Q3, Q4")

    start_key = (start_year, quarter_index[start_quarter])
    end_key = (end_year, quarter_index[end_quarter])
    if start_key > end_key:
        raise ValueError("start quarter must be earlier than or equal to end quarter")

    year = start_year
    quarter_pos = quarter_index[start_quarter]
    while (year, quarter_pos) <= end_key:
        yield year, quarter_order[quarter_pos]
        quarter_pos += 1
        if quarter_pos == len(quarter_order):
            year += 1
            quarter_pos = 0
