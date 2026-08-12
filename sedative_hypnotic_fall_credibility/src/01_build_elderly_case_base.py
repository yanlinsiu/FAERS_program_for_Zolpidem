from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FAERS_ROOT = PROJECT_DIR.parent
DEFAULT_DEMO_GLOB = (
    FAERS_ROOT
    / "archive_old_outputs"
    / "results_before_20260525"
    / "OUTPUT_COUNTRY"
    / "*"
    / "demo_*.parquet"
)
DEFAULT_ML_FEATURE = (
    FAERS_ROOT
    / "runs"
    / "mainline_2026-05-20"
    / "OUTPUT_ML"
    / "features_v2"
    / "datasets"
    / "ml_feature_v2_2004_2025.parquet"
)
DEFAULT_OUT = PROJECT_DIR / "outputs" / "intermediate" / "01_elderly_case_base.parquet"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "01_elderly_case_base_qc.csv"

DEMO_COLUMNS = [
    "caseid",
    "primaryid",
    "age_years",
    "age_group",
    "sex_clean",
    "reporter_country",
    "occr_country",
    "year",
    "quarter",
]


def regulatory_period(year: object) -> str:
    if pd.isna(year):
        return "unknown"
    year_int = int(year)
    if 2004 <= year_int <= 2012:
        return "2004-2012"
    if 2013 <= year_int <= 2018:
        return "2013-2018"
    if 2019 <= year_int <= 2025:
        return "2019-2025"
    return "outside_2004_2025"


def normalize_age_group(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("years", "").replace("year", "")
    if text in {"65-74", "65_74"}:
        return "65-74"
    if text in {"75-84", "75_84"}:
        return "75-84"
    if text in {"85+", ">=85", "85andover", "85andover", ">85"}:
        return ">=85"
    return str(value).strip()


def derive_age_group_3(row: pd.Series) -> str:
    age = row.get("age_years")
    if pd.notna(age):
        age_float = float(age)
        if 65 <= age_float <= 74:
            return "65-74"
        if 75 <= age_float <= 84:
            return "75-84"
        if 85 <= age_float <= 120:
            return ">=85"
    return normalize_age_group(row.get("age_group"))


def is_elderly_row(row: pd.Series) -> bool:
    age = row.get("age_years")
    if pd.notna(age):
        return 65 <= float(age) <= 120
    return normalize_age_group(row.get("age_group")) in {"65-74", "75-84", ">=85"}


def country_group(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "unknown"
    text = str(value).strip().upper()
    if text in {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        return "US"
    return "non-US"


def find_demo_files(pattern: Path) -> list[Path]:
    files = [Path(p) for p in glob.glob(str(pattern))]
    annual = []
    for path in files:
        if re.fullmatch(r"demo_\d{4}\.parquet", path.name, flags=re.I):
            annual.append(path)
    return sorted(annual)


def read_demo_files(files: list[Path]) -> pd.DataFrame:
    chunks = []
    for path in files:
        print(f"Reading {path}")
        chunk = pd.read_parquet(path, columns=DEMO_COLUMNS)
        chunk["source_file"] = str(path)
        chunks.append(chunk)
    if not chunks:
        raise FileNotFoundError("No annual demo parquet files found.")
    return pd.concat(chunks, ignore_index=True)


def add_reporting_fields(base: pd.DataFrame, ml_feature_path: Path) -> pd.DataFrame:
    base = base.copy()
    base["rept_cod"] = pd.NA
    base["e_sub"] = pd.NA
    if not ml_feature_path.exists():
        return base

    ml = pd.read_parquet(ml_feature_path, columns=["caseid", "rept_cod", "e_sub"])
    ml["caseid"] = ml["caseid"].astype(str)
    ml = ml.drop_duplicates("caseid", keep="first")

    merged = base.merge(ml, on="caseid", how="left", suffixes=("", "_ml"))
    merged["rept_cod"] = merged["rept_cod_ml"]
    merged["e_sub"] = merged["e_sub_ml"]
    return merged.drop(columns=["rept_cod_ml", "e_sub_ml"])


def qc_rows(raw: pd.DataFrame, elderly: pd.DataFrame, output: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "elderly_case_base", "metric": metric, "value": value, "note": note})

    add("demo_rows_raw", len(raw))
    add("demo_unique_caseid_raw", raw["caseid"].astype(str).nunique())
    add("elderly_rows_before_caseid_dedup", len(elderly))
    add("elderly_unique_caseid_before_dedup", elderly["caseid"].astype(str).nunique())
    add("elderly_rows_final", len(output))
    add("duplicate_caseid_final", int(output["caseid"].duplicated().sum()))
    add("missing_caseid_final", int(output["caseid"].isna().sum()))
    add("missing_year_final", int(output["year"].isna().sum()))
    add("missing_age_years_final", int(output["age_years"].isna().sum()))
    add("missing_age_group_3_final", int(output["age_group_3"].eq("").sum()))
    add("sex_m_or_f_final", int(output["sex_clean"].isin(["M", "F"]).sum()))
    add("sex_not_m_or_f_final", int((~output["sex_clean"].isin(["M", "F"])).sum()))
    add("missing_rept_cod_final", int(output["rept_cod"].isna().sum()), "Filled from ml_feature_v2 when available.")
    add("missing_e_sub_final", int(output["e_sub"].isna().sum()), "Filled from ml_feature_v2 when available.")

    for group, count in output["age_group_3"].value_counts(dropna=False).sort_index().items():
        add(f"age_group_3__{group}", int(count))
    for period, count in output["regulatory_period"].value_counts(dropna=False).sort_index().items():
        add(f"regulatory_period__{period}", int(count))
    for sex, count in output["sex_clean"].value_counts(dropna=False).sort_index().items():
        add(f"sex_clean__{sex}", int(count))
    for group, count in output["country_group"].value_counts(dropna=False).sort_index().items():
        add(f"country_group__{group}", int(count))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-glob", type=Path, default=DEFAULT_DEMO_GLOB)
    parser.add_argument("--ml-feature", type=Path, default=DEFAULT_ML_FEATURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    demo_files = find_demo_files(args.demo_glob)
    raw = read_demo_files(demo_files)
    raw["caseid"] = raw["caseid"].astype(str)
    raw["primaryid"] = raw["primaryid"].astype(str)
    raw["age_years"] = pd.to_numeric(raw["age_years"], errors="coerce")
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw["quarter"] = pd.to_numeric(raw["quarter"], errors="coerce").astype("Int64")
    raw["sex_clean"] = raw["sex_clean"].fillna("").astype(str).str.strip().str.upper()

    elderly_mask = raw.apply(is_elderly_row, axis=1)
    elderly = raw.loc[elderly_mask].copy()
    elderly["age_group_3"] = elderly.apply(derive_age_group_3, axis=1)
    elderly["regulatory_period"] = elderly["year"].map(regulatory_period)

    country_source = elderly["reporter_country"].where(
        elderly["reporter_country"].notna() & elderly["reporter_country"].astype(str).str.strip().ne(""),
        elderly["occr_country"],
    )
    elderly["country_group"] = country_source.map(country_group)

    elderly = elderly.sort_values(["caseid", "year", "quarter", "primaryid"], kind="mergesort")
    elderly = elderly.drop_duplicates("caseid", keep="first")
    elderly = elderly.rename(columns={"primaryid": "primaryid_example"})

    output_columns = [
        "caseid",
        "primaryid_example",
        "year",
        "quarter",
        "regulatory_period",
        "age_years",
        "age_group_3",
        "sex_clean",
        "reporter_country",
        "occr_country",
        "country_group",
    ]
    output = elderly[output_columns].copy()
    output = add_reporting_fields(output, args.ml_feature)
    output = output[
        [
            "caseid",
            "primaryid_example",
            "year",
            "quarter",
            "regulatory_period",
            "age_years",
            "age_group_3",
            "sex_clean",
            "reporter_country",
            "occr_country",
            "country_group",
            "rept_cod",
            "e_sub",
        ]
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.out, index=False)
    pd.DataFrame(qc_rows(raw, raw.loc[elderly_mask], output)).to_csv(args.qc_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.qc_out}")
    print(f"Final elderly cases: {len(output):,}")


if __name__ == "__main__":
    main()
