from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
FAERS_ROOT = PROJECT_DIR.parent
DEFAULT_RAW_DIR = FAERS_ROOT / "data"
DEFAULT_PROCESSED_DIR = (
    FAERS_ROOT
    / "archive_old_outputs"
    / "results_before_20260525"
    / "OUTPUT_COUNTRY"
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
DEFAULT_OUT = PROJECT_DIR / "outputs" / "qc" / "00_input_data_manifest.csv"
DEFAULT_SUMMARY_OUT = PROJECT_DIR / "outputs" / "qc" / "00_input_data_manifest_summary.csv"

RAW_PATTERN = re.compile(r"([A-Z]+)(\d{2})Q([1-4])\.TXT$", re.I)
ANNUAL_PARQUET_PATTERN = re.compile(
    r"(demo|drug|reac|outc|outcome_dataset|drug_feature|drug_exposure|drug_feature_dataset)_(\d{4})(?:_case)?\.parquet$",
    re.I,
)
QUARTERLY_PARQUET_PATTERN = re.compile(
    r"(demo|drug|reac|outc|outcome_dataset|drug_feature|drug_exposure|drug_feature_dataset)_(\d{4})q([1-4])(?:_case)?\.parquet$",
    re.I,
)

KEY_FIELDS = {
    "demo": {"caseid", "primaryid", "age_years", "age_group", "sex_clean", "year", "quarter"},
    "drug": {"caseid", "primaryid", "role_cod", "drugname", "prod_ai"},
    "reac": {"caseid"},
    "outc": {"caseid"},
    "outcome_dataset": {"caseid"},
    "drug_feature": {"caseid", "drug_n", "distinct_drug_n"},
    "drug_exposure": {"caseid"},
    "drug_feature_dataset": {"caseid"},
    "ml_feature_v2": {
        "caseid",
        "age_years",
        "age_group",
        "sex_clean",
        "year",
        "quarter",
        "is_fall",
        "serious",
        "indi_insomnia",
        "drug_n",
        "distinct_drug_n",
    },
}


def year_from_two_digits(value: str) -> int:
    year = int(value)
    return 2000 + year if year < 50 else 1900 + year


def parquet_info(path: Path, data_role: str) -> dict[str, object]:
    metadata = pq.read_metadata(path)
    schema = pq.read_schema(path)
    fields = list(schema.names)
    field_set = set(fields)
    required = KEY_FIELDS.get(data_role, {"caseid"})
    missing = sorted(required - field_set)
    return {
        "n_rows": metadata.num_rows,
        "n_columns": len(fields),
        "columns": ";".join(fields),
        "key_fields_checked": ";".join(sorted(required)),
        "missing_key_fields": ";".join(missing),
        "has_required_key_fields": len(missing) == 0,
    }


def raw_ascii_rows(raw_dir: Path) -> list[dict[str, object]]:
    rows = []
    if not raw_dir.exists():
        return rows

    for path in raw_dir.rglob("*.TXT"):
        match = RAW_PATTERN.match(path.name)
        if not match:
            continue
        table, yy, quarter = match.groups()
        rows.append(
            {
                "source_family": "raw_ascii",
                "data_role": table.lower(),
                "year": year_from_two_digits(yy),
                "quarter": int(quarter),
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "n_rows": pd.NA,
                "n_columns": pd.NA,
                "columns": "",
                "key_fields_checked": "",
                "missing_key_fields": "",
                "has_required_key_fields": pd.NA,
                "note": "Raw FAERS ASCII; keep as fallback and audit source.",
            }
        )
    return rows


def processed_parquet_rows(processed_dir: Path) -> list[dict[str, object]]:
    rows = []
    if not processed_dir.exists():
        return rows

    for path in processed_dir.rglob("*.parquet"):
        source_family = ""
        data_role = ""
        year: int | None = None
        quarter: int | str = ""

        quarterly = QUARTERLY_PARQUET_PATTERN.match(path.name)
        annual = ANNUAL_PARQUET_PATTERN.match(path.name)
        if quarterly:
            data_role, year_text, quarter_text = quarterly.groups()
            source_family = "old_quarterly_parquet"
            year = int(year_text)
            quarter = int(quarter_text)
        elif annual and "quarterly" not in {part.lower() for part in path.parts}:
            data_role, year_text = annual.groups()
            source_family = "old_annual_parquet"
            year = int(year_text)
        else:
            continue

        info = parquet_info(path, data_role.lower())
        rows.append(
            {
                "source_family": source_family,
                "data_role": data_role.lower(),
                "year": year,
                "quarter": quarter,
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                **info,
                "note": "Preferred processed input for this project.",
            }
        )
    return rows


def full_period_rows(ml_feature_path: Path) -> list[dict[str, object]]:
    if not ml_feature_path.exists():
        return [
            {
                "source_family": "full_period_parquet",
                "data_role": "ml_feature_v2",
                "year": "",
                "quarter": "",
                "path": str(ml_feature_path),
                "file_size_bytes": pd.NA,
                "n_rows": pd.NA,
                "n_columns": pd.NA,
                "columns": "",
                "key_fields_checked": ";".join(sorted(KEY_FIELDS["ml_feature_v2"])),
                "missing_key_fields": "FILE_NOT_FOUND",
                "has_required_key_fields": False,
                "note": "Expected full-period feature table was not found.",
            }
        ]

    info = parquet_info(ml_feature_path, "ml_feature_v2")
    return [
        {
            "source_family": "full_period_parquet",
            "data_role": "ml_feature_v2",
            "year": "2004-2025",
            "quarter": "",
            "path": str(ml_feature_path),
            "file_size_bytes": ml_feature_path.stat().st_size,
            **info,
            "note": "Useful old full-period table for elderly base, covariates, indications, and existing phenotypes.",
        }
    ]


def build_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        manifest.groupby(["source_family", "data_role"], dropna=False)
        .agg(
            n_files=("path", "size"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            n_rows_total=("n_rows", "sum"),
            files_missing_key_fields=("has_required_key_fields", lambda s: int((s == False).sum())),
        )
        .reset_index()
        .sort_values(["source_family", "data_role"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--ml-feature", type=Path, default=DEFAULT_ML_FEATURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    rows = []
    rows.extend(raw_ascii_rows(args.raw_dir))
    rows.extend(processed_parquet_rows(args.processed_dir))
    rows.extend(full_period_rows(args.ml_feature))

    manifest = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False, encoding="utf-8-sig")

    summary = build_summary(manifest)
    summary.to_csv(args.summary_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
