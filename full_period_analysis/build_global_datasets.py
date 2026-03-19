from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd

from config import GLOBAL_DATASET_DIR, GLOBAL_QC_DIR, OUTPUT_ROOT, RAW_ROOT


YEAR_Q_PATTERN = re.compile(r"(\d{4})q([1-4])", re.IGNORECASE)
PATH_YEAR_Q_PATTERN = re.compile(r"[\\/](20\d{2})[\\/]Q([1-4])[\\/]", re.IGNORECASE)


def discover_quarterly_files(pattern: str) -> list[Path]:
    files: list[Path] = []
    for year_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        quarterly_dir = year_dir / "quarterly"
        if not quarterly_dir.exists():
            continue
        files.extend(sorted(quarterly_dir.glob(pattern)))
    return files


def load_deleted_case_map(start_year: int, end_year: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for year in range(max(start_year, 2019), end_year + 1):
        for quarter in ("Q1", "Q2", "Q3", "Q4"):
            quarter_dir = RAW_ROOT / str(year) / quarter
            if not quarter_dir.exists():
                continue
            deleted_dirs = [d for d in quarter_dir.iterdir() if d.is_dir() and "DELETE" in d.name.upper()]
            for deleted_dir in deleted_dirs:
                for file_path in sorted(deleted_dir.glob("*.txt")):
                    with file_path.open("r", encoding="latin1", errors="ignore") as handle:
                        for line in handle:
                            token = line.strip().split("$", 1)[0].strip()
                            if token.isdigit():
                                records.append({"year": year, "quarter": quarter, "caseid": token})

    if not records:
        return pd.DataFrame(columns=["year", "quarter", "caseid"])

    deleted_df = pd.DataFrame(records).drop_duplicates()
    deleted_df["year"] = pd.to_numeric(deleted_df["year"], errors="coerce").astype("Int64")
    deleted_df["quarter"] = deleted_df["quarter"].astype(str).str.upper().str.strip()
    deleted_df["caseid"] = deleted_df["caseid"].astype(str).str.strip()
    deleted_df = deleted_df.dropna(subset=["year"])
    deleted_df = deleted_df[deleted_df["caseid"] != ""].copy()
    deleted_df["year"] = deleted_df["year"].astype(int)
    return deleted_df


def period_from_feature_path(path: Path) -> str:
    match = YEAR_Q_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse year-quarter token from feature file name: {path}")
    return f"{match.group(1)}Q{match.group(2)}"


def _sql_quoted(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def create_global_case_index(
    con: duckdb.DuckDBPyConnection,
    case_files: list[Path],
    deleted_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> None:
    case_file_list = [file_path.as_posix() for file_path in case_files]
    con.register("deleted_case_map", deleted_df)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE case_base_all AS
        SELECT
            TRIM(CAST(caseid AS VARCHAR)) AS caseid,
            CAST(primaryid AS BIGINT) AS primaryid,
            CAST(fda_dt AS DATE) AS fda_dt,
            CAST(year AS INTEGER) AS year,
            UPPER(TRIM(CAST(quarter AS VARCHAR))) AS quarter
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(year AS INTEGER) BETWEEN ? AND ?
        """,
        [case_file_list, start_year, end_year],
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE case_base_filtered AS
        SELECT c.*
        FROM case_base_all c
        LEFT JOIN deleted_case_map d
            ON c.year = d.year
           AND c.quarter = d.quarter
           AND c.caseid = d.caseid
        WHERE c.caseid <> ''
          AND d.caseid IS NULL
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE global_case_index AS
        WITH ranked AS (
            SELECT
                caseid,
                primaryid,
                fda_dt,
                year,
                quarter,
                ROW_NUMBER() OVER (
                    PARTITION BY caseid
                    ORDER BY
                        fda_dt DESC NULLS LAST,
                        primaryid DESC NULLS LAST,
                        year DESC,
                        CASE quarter
                            WHEN 'Q1' THEN 1
                            WHEN 'Q2' THEN 2
                            WHEN 'Q3' THEN 3
                            WHEN 'Q4' THEN 4
                            ELSE 0
                        END DESC
                ) AS rn
            FROM case_base_filtered
        )
        SELECT
            caseid,
            primaryid,
            fda_dt,
            year,
            quarter,
            CAST(year AS VARCHAR) || quarter AS year_quarter
        FROM ranked
        WHERE rn = 1
        """
    )


def create_global_signal_dataset(
    con: duckdb.DuckDBPyConnection,
    signal_files: list[Path],
    start_year: int,
    end_year: int,
) -> None:
    signal_file_list = [file_path.as_posix() for file_path in signal_files]
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE signal_all AS
        SELECT
            *
        FROM read_parquet(?, union_by_name=true)
        WHERE CAST(year AS INTEGER) BETWEEN ? AND ?
        """,
        [signal_file_list, start_year, end_year],
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE signal_global AS
        SELECT s.*
        FROM signal_all s
        INNER JOIN global_case_index g
            ON TRIM(CAST(s.caseid AS VARCHAR)) = g.caseid
           AND CAST(s.year AS INTEGER) = g.year
           AND UPPER(TRIM(CAST(s.quarter AS VARCHAR))) = g.quarter
        """
    )


def create_global_feature_dataset(
    con: duckdb.DuckDBPyConnection,
    feature_files: list[Path],
) -> None:
    union_sql_parts: list[str] = []
    for feature_file in feature_files:
        period = period_from_feature_path(feature_file)
        union_sql_parts.append(
            "SELECT *, "
            f"'{period}' AS dataset_period "
            f"FROM read_parquet('{_sql_quoted(feature_file)}')"
        )

    if not union_sql_parts:
        raise FileNotFoundError("No quarterly drug_feature_*_case.parquet files found.")

    union_sql = "\nUNION ALL\n".join(union_sql_parts)
    con.execute(f"CREATE OR REPLACE TEMP TABLE feature_all AS {union_sql}")
    con.execute(
        """
        CREATE OR REPLACE TABLE feature_global AS
        SELECT f.*
        FROM feature_all f
        INNER JOIN global_case_index g
            ON TRIM(CAST(f.caseid AS VARCHAR)) = g.caseid
           AND UPPER(TRIM(CAST(f.dataset_period AS VARCHAR))) = g.year_quarter
        """
    )


def write_outputs(
    con: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
) -> tuple[Path, Path, Path, Path]:
    GLOBAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_QC_DIR.mkdir(parents=True, exist_ok=True)

    period_token = f"{start_year}_{end_year}"
    case_index_file = GLOBAL_DATASET_DIR / f"global_case_index_{period_token}.parquet"
    signal_file = GLOBAL_DATASET_DIR / f"signal_dataset_{period_token}.parquet"
    feature_file = GLOBAL_DATASET_DIR / f"drug_feature_{period_token}_case.parquet"
    qc_file = GLOBAL_QC_DIR / f"global_dataset_qc_{period_token}.csv"

    con.execute(f"COPY global_case_index TO '{_sql_quoted(case_index_file)}' (FORMAT PARQUET)")
    con.execute(f"COPY signal_global TO '{_sql_quoted(signal_file)}' (FORMAT PARQUET)")
    con.execute(f"COPY feature_global TO '{_sql_quoted(feature_file)}' (FORMAT PARQUET)")

    qc_df = con.execute(
        """
        SELECT 'global_case_index' AS dataset, COUNT(*) AS n_rows FROM global_case_index
        UNION ALL
        SELECT 'signal_global' AS dataset, COUNT(*) AS n_rows FROM signal_global
        UNION ALL
        SELECT 'feature_global' AS dataset, COUNT(*) AS n_rows FROM feature_global
        """
    ).df()
    qc_df.to_csv(qc_file, index=False, encoding="utf-8-sig")
    return case_index_file, signal_file, feature_file, qc_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 2004-2025 global deduplicated FAERS datasets with DuckDB."
    )
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    case_files = discover_quarterly_files("case_base_dataset_*q*.parquet")
    signal_files = discover_quarterly_files("signal_dataset_*q*.parquet")
    feature_files = discover_quarterly_files("drug_feature_*q*_case.parquet")

    if not case_files:
        raise FileNotFoundError("No quarterly case_base_dataset parquet files found in OUTPUT/*/quarterly.")
    if not signal_files:
        raise FileNotFoundError("No quarterly signal_dataset parquet files found in OUTPUT/*/quarterly.")

    deleted_df = load_deleted_case_map(start_year, end_year)

    con = duckdb.connect()
    try:
        create_global_case_index(con, case_files, deleted_df, start_year, end_year)
        create_global_signal_dataset(con, signal_files, start_year, end_year)
        create_global_feature_dataset(con, feature_files)
        case_file, signal_file, feature_file, qc_file = write_outputs(con, start_year, end_year)
    finally:
        con.close()

    print("Global datasets built successfully.")
    print(f"Deleted-case map rows: {len(deleted_df)}")
    print(f"Case index: {case_file}")
    print(f"Signal dataset: {signal_file}")
    print(f"Feature dataset: {feature_file}")
    print(f"QC summary: {qc_file}")


if __name__ == "__main__":
    main()
