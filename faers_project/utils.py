from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from pandas.errors import ParserError


COLUMN_ALIASES = {
    "demo": {
        "isr": "primaryid",
        "case": "caseid",
        "i_f_cod": "i_f_code",
        "gndr_cod": "sex",
    },
    "drug": {
        "isr": "primaryid",
        "lot_num": "lot_nbr",
    },
    "indi": {
        "isr": "primaryid",
        "drug_seq": "indi_drug_seq",
    },
    "outc": {
        "isr": "primaryid",
        "outc_code": "outc_cod",
    },
    "reac": {
        "isr": "primaryid",
        "reac_pt": "pt",
    },
    "rpsr": {
        "isr": "primaryid",
    },
    "ther": {
        "isr": "primaryid",
        "drug_seq": "dsg_drug_seq",
    },
}

OPTIONAL_DEFAULT_COLUMNS = {
    "demo": ["caseversion", "fda_dt", "age", "age_cod", "sex", "gndr_cod", "serious"],
    "drug": ["primaryid", "caseid", "prod_ai", "role_cod"],
    "outc": ["primaryid", "caseid", "outc_cod"],
    "reac": ["primaryid", "caseid", "pt", "reac_pt"],
}

FALLBACK_MERGE_INDEX = {
    "demo": 9,
}


def clean_column_name(column_name: str) -> str:
    cleaned = str(column_name).strip().lower().lstrip("\ufeff")
    if cleaned.startswith("ï»¿"):
        cleaned = cleaned[3:]
    return cleaned


def resolve_raw_root(raw_root) -> Path:
    raw_root = Path(raw_root)
    if any((raw_root / str(year)).is_dir() for year in range(2004, 2026)):
        return raw_root

    data_dir = raw_root / "data"
    if any((data_dir / str(year)).is_dir() for year in range(2004, 2026)):
        return data_dir

    return raw_root


def resolve_quarter_dir(raw_root, year, quarter) -> Path:
    resolved_root = resolve_raw_root(raw_root)
    quarter_upper = str(quarter).upper()
    year_dir = resolved_root / str(year)
    quarter_dir = year_dir / quarter_upper
    if quarter_dir.exists():
        return quarter_dir

    raise FileNotFoundError(f"quarter directory not found: {quarter_dir}")


def build_file_path(raw_root, year, quarter, table_name):
    """Return the actual FAERS TXT file path, handling ascii/ASCII and filename case changes."""
    quarter_dir = resolve_quarter_dir(raw_root, year, quarter)
    quarter_upper = str(quarter).upper()
    year_short = str(year)[-2:]
    expected_name = f"{str(table_name).upper()}{year_short}{quarter_upper}.txt"

    exact_matches = [path for path in quarter_dir.rglob("*.txt") if path.name.upper() == expected_name.upper()]
    if exact_matches:
        return sorted(exact_matches, key=lambda item: str(item))[0]

    fallback_matches = [
        path
        for path in quarter_dir.rglob("*.txt")
        if path.stem.upper().startswith(str(table_name).upper()) and quarter_upper.upper() in path.stem.upper()
    ]
    if fallback_matches:
        return sorted(fallback_matches, key=lambda item: str(item))[0]

    candidate = quarter_dir / "ASCII" / expected_name
    return candidate


def normalize_faers_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    dataset_key = str(dataset_name).lower()
    out_df = df.copy()
    out_df.columns = [clean_column_name(column) for column in out_df.columns]

    rename_map = {}
    for source_name, target_name in COLUMN_ALIASES.get(dataset_key, {}).items():
        if source_name in out_df.columns and target_name not in out_df.columns:
            rename_map[source_name] = target_name
    if rename_map:
        out_df = out_df.rename(columns=rename_map)

    for column_name in OPTIONAL_DEFAULT_COLUMNS.get(dataset_key, []):
        if column_name not in out_df.columns:
            out_df[column_name] = pd.NA

    return out_df


def ensure_required_columns(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{table_name} missing required columns after normalization: {missing_cols}")


def _read_faers_txt_with_fallback(file_path, dataset_name: str | None = None) -> pd.DataFrame:
    dataset_key = str(dataset_name or Path(file_path).stem[:4]).lower()
    merge_index = FALLBACK_MERGE_INDEX.get(dataset_key)

    with Path(file_path).open("r", encoding="latin1", newline="") as handle:
        header_line = handle.readline().rstrip("\r\n")
        header = [clean_column_name(column) for column in header_line.split("$")]
        expected_len = len(header)

        rows: list[list[str]] = []
        adjusted_rows = 0
        padded_rows = 0

        for line_number, raw_line in enumerate(handle, start=2):
            line = raw_line.rstrip("\r\n")
            parts = line.split("$")

            if (
                len(parts) > expected_len
                and parts[-1] == ""
                and (len(parts) - 1) % expected_len == 0
                and (len(parts) - 1) // expected_len > 1
            ):
                body = parts[:-1]
                chunk_size = expected_len
                chunk_count = len(body) // chunk_size
                for chunk_index in range(chunk_count):
                    rows.append(body[chunk_index * chunk_size : (chunk_index + 1) * chunk_size])
                adjusted_rows += 1
                continue

            while len(parts) > expected_len and parts[-1] == "":
                parts.pop()
                adjusted_rows += 1

            if len(parts) > expected_len and len(parts) % expected_len == 0:
                chunk_size = expected_len
                chunk_count = len(parts) // chunk_size
                for chunk_index in range(chunk_count):
                    rows.append(parts[chunk_index * chunk_size : (chunk_index + 1) * chunk_size])
                adjusted_rows += 1
                continue

            if len(parts) > expected_len:
                if merge_index is None or merge_index >= len(parts):
                    raise ParserError(
                        f"Unable to repair {file_path} at line {line_number}: "
                        f"expected {expected_len} fields, saw {len(parts)}"
                    )
                extra_count = len(parts) - expected_len
                merged_value = "$".join(parts[merge_index : merge_index + extra_count + 1])
                parts = (
                    parts[:merge_index]
                    + [merged_value]
                    + parts[merge_index + extra_count + 1 :]
                )
                adjusted_rows += 1

            if len(parts) < expected_len:
                parts = parts + [""] * (expected_len - len(parts))
                padded_rows += 1

            rows.append(parts)

    df = pd.DataFrame(rows, columns=header, dtype="object")
    if adjusted_rows or padded_rows:
        print(
            f"Fallback parser repaired {Path(file_path).name}: "
            f"adjusted_rows={adjusted_rows}, padded_rows={padded_rows}"
        )
    return df


def read_faers_txt(file_path, dataset_name: str | None = None):
    """Read FAERS ASCII file with standard settings and normalize known schema aliases."""
    try:
        df = pd.read_csv(
            file_path,
            sep="$",
            encoding="latin1",
            low_memory=False,
            dtype=str,
            index_col=False,
        )
    except ParserError:
        df = _read_faers_txt_with_fallback(file_path, dataset_name=dataset_name)
    dataset_key = dataset_name or Path(file_path).stem[:4]
    df = normalize_faers_columns(df, dataset_key)
    return df


def load_standardized_demo(raw_root, year, quarter) -> pd.DataFrame:
    demo_file = build_file_path(raw_root, year, quarter, "DEMO")
    if not demo_file.exists():
        raise FileNotFoundError(f"file not found: {demo_file}")

    demo_df = read_faers_txt(demo_file, dataset_name="DEMO")
    ensure_required_columns(
        demo_df,
        ["primaryid", "caseid", "fda_dt", "caseversion", "age", "age_cod"],
        "DEMO",
    )

    demo_df["caseid"] = demo_df["caseid"].where(demo_df["caseid"].notna(), "").astype(str).str.strip()
    demo_df["primaryid"] = pd.to_numeric(demo_df["primaryid"], errors="coerce")
    demo_df["caseversion"] = pd.to_numeric(demo_df["caseversion"], errors="coerce")
    demo_df["fda_dt"] = pd.to_datetime(demo_df["fda_dt"], format="%Y%m%d", errors="coerce")
    return demo_df


def _list_deleted_case_files(raw_root, year, quarter) -> list[Path]:
    quarter_dir = resolve_quarter_dir(raw_root, year, quarter)
    deleted_dirs = [name for name in quarter_dir.iterdir() if name.is_dir() and "DELETE" in name.name.upper()]

    files: list[Path] = []
    for directory in deleted_dirs:
        files.extend(sorted(directory.glob("*.txt")))
    return files


def load_deleted_caseids(raw_root, year, quarter) -> set[str]:
    deleted_caseids: set[str] = set()
    digit_pattern = re.compile(r"^\d+$")
    for file_path in _list_deleted_case_files(raw_root, year, quarter):
        with file_path.open("r", encoding="latin1", errors="ignore") as handle:
            for raw_line in handle:
                token = raw_line.strip()
                if not token:
                    continue
                token = token.split("$", 1)[0].strip()
                if digit_pattern.fullmatch(token):
                    deleted_caseids.add(token)
    return deleted_caseids


def exclude_deleted_caseids(
    df: pd.DataFrame,
    raw_root,
    year,
    quarter,
    caseid_col: str = "caseid",
) -> tuple[pd.DataFrame, int, int]:
    if caseid_col not in df.columns:
        return df.copy(), 0, 0

    deleted_caseids = load_deleted_caseids(raw_root, year, quarter)
    if not deleted_caseids:
        return df.copy(), 0, 0

    out_df = df.copy()
    caseid_norm = out_df[caseid_col].where(out_df[caseid_col].notna(), "").astype(str).str.strip()
    remove_mask = caseid_norm.isin(deleted_caseids)
    removed_rows = int(remove_mask.sum())
    removed_caseids = int(caseid_norm[remove_mask].nunique())
    out_df = out_df.loc[~remove_mask].copy()
    return out_df, removed_rows, removed_caseids


def deduplicate_demo_records(df):
    """Keep one latest DEMO record for each caseid."""
    ensure_required_columns(df, ["caseid", "primaryid", "fda_dt", "caseversion"], "DEMO")

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
    ensure_required_columns(df, ["age", "age_cod"], "DEMO")

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


def load_demo_case_lookup(raw_root, year, quarter, output_root=None) -> pd.DataFrame:
    """
    Return deduplicated DEMO mapping between primaryid and caseid for a quarter.
    """
    if output_root is not None:
        output_root = Path(output_root)
        preferred_file = output_root / f"case_base_dataset_{year}{str(quarter).lower()}.parquet"
        if preferred_file.exists():
            demo_df = pd.read_parquet(preferred_file)
            ensure_required_columns(demo_df, ["primaryid", "caseid"], "DEMO output")
            lookup_df = demo_df[["primaryid", "caseid"]].copy()
            lookup_df["primaryid"] = pd.to_numeric(lookup_df["primaryid"], errors="coerce")
            lookup_df["caseid"] = lookup_df["caseid"].where(lookup_df["caseid"].notna(), "").astype(str).str.strip()
            lookup_df = lookup_df.dropna(subset=["primaryid"])
            lookup_df = lookup_df[lookup_df["caseid"] != ""].drop_duplicates(subset=["primaryid"], keep="last")
            return lookup_df

    demo_df = load_standardized_demo(raw_root, year, quarter)
    demo_df, _, _ = exclude_deleted_caseids(demo_df, raw_root, year, quarter)
    demo_df = deduplicate_demo_records(demo_df)
    lookup_df = demo_df[["primaryid", "caseid"]].copy()
    lookup_df = lookup_df.dropna(subset=["primaryid"])
    lookup_df = lookup_df[lookup_df["caseid"] != ""].drop_duplicates(subset=["primaryid"], keep="last")
    return lookup_df


def attach_caseid_from_demo(df: pd.DataFrame, raw_root, year, quarter, output_root=None) -> pd.DataFrame:
    """
    Fill missing caseid values from DEMO using primaryid, which is required for early FAERS quarters.
    """
    out_df = df.copy()
    if "primaryid" not in out_df.columns:
        raise ValueError("Cannot attach caseid from DEMO without primaryid column.")

    if "caseid" not in out_df.columns:
        out_df["caseid"] = pd.NA

    out_df["primaryid"] = pd.to_numeric(out_df["primaryid"], errors="coerce")
    out_df["caseid"] = out_df["caseid"].where(out_df["caseid"].notna(), "").astype(str).str.strip()
    missing_caseid_mask = out_df["caseid"] == ""
    if not missing_caseid_mask.any():
        return out_df

    lookup_df = load_demo_case_lookup(raw_root, year, quarter, output_root=output_root)
    lookup_df = lookup_df.rename(columns={"caseid": "demo_caseid"})
    merged_df = out_df.merge(lookup_df, on="primaryid", how="left")
    merged_df["caseid"] = merged_df["caseid"].where(merged_df["caseid"] != "", merged_df["demo_caseid"])
    merged_df["caseid"] = (
        merged_df["caseid"].where(merged_df["caseid"].notna(), "").astype(str).str.strip()
    )
    return merged_df.drop(columns=["demo_caseid"])


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
            ensure_required_columns(demo_df, ["primaryid"], "DEMO output")
            primaryid = pd.to_numeric(demo_df["primaryid"], errors="coerce")
            return set(primaryid.dropna())

    demo_df = load_standardized_demo(raw_root, year, quarter)
    demo_df, _, _ = exclude_deleted_caseids(demo_df, raw_root, year, quarter)
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
