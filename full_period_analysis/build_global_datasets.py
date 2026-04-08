from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

from config import GLOBAL_DATASET_DIR, GLOBAL_QC_DIR, OUTPUT_ROOT


YEAR_Q_PATTERN = re.compile(r"(\d{4})q([1-4])", re.IGNORECASE)


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
    start_year: int,
    end_year: int,
) -> None:
    case_file_list = [file_path.as_posix() for file_path in case_files]
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
        SELECT *
        FROM case_base_all
        WHERE caseid <> ''
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
) -> tuple[Path, Path, Path, Path, Path]:
    GLOBAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_QC_DIR.mkdir(parents=True, exist_ok=True)

    period_token = f"{start_year}_{end_year}"
    case_index_file = GLOBAL_DATASET_DIR / f"global_case_index_{period_token}.parquet"
    signal_file = GLOBAL_DATASET_DIR / f"signal_dataset_{period_token}.parquet"
    feature_file = GLOBAL_DATASET_DIR / f"drug_feature_{period_token}_case.parquet"
    qc_file = GLOBAL_QC_DIR / f"global_dataset_qc_{period_token}.csv"
    qc_summary_file = GLOBAL_QC_DIR / f"global_signal_summary_{period_token}.csv"

    con.execute(f"COPY global_case_index TO '{_sql_quoted(case_index_file)}' (FORMAT PARQUET)")
    con.execute(f"COPY signal_global TO '{_sql_quoted(signal_file)}' (FORMAT PARQUET)")
    con.execute(f"COPY feature_global TO '{_sql_quoted(feature_file)}' (FORMAT PARQUET)")

    qc_df = con.execute(
        """
        SELECT 'case_base_all' AS dataset, COUNT(*) AS n_rows FROM case_base_all
        UNION ALL
        SELECT 'case_base_filtered' AS dataset, COUNT(*) AS n_rows FROM case_base_filtered
        UNION ALL
        SELECT 'global_case_index' AS dataset, COUNT(*) AS n_rows FROM global_case_index
        UNION ALL
        SELECT 'signal_global' AS dataset, COUNT(*) AS n_rows FROM signal_global
        UNION ALL
        SELECT 'feature_global' AS dataset, COUNT(*) AS n_rows FROM feature_global
        """
    ).df()
    qc_df.to_csv(qc_file, index=False, encoding="utf-8-sig")

    signal_summary_df = con.execute(
        """
        WITH metrics AS (
            SELECT 'global_total_cases' AS metric, COUNT(*)::BIGINT AS value FROM global_case_index
            UNION ALL
            SELECT 'signal_dataset_cases', COUNT(*)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'feature_dataset_cases', COUNT(*)::BIGINT FROM feature_global
            UNION ALL
            SELECT 'strict_fall_cases', SUM(CASE WHEN is_fall_narrow THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'broad_fall_cases', SUM(CASE WHEN is_fall_broad THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'suspect_any_cases_ps_ss', SUM(CASE WHEN suspect_role_any THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'suspect_any_cases_ps_only', SUM(CASE WHEN suspect_role_any_ps THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'zolpidem_any_cases', SUM(CASE WHEN is_zolpidem_any THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'zolpidem_suspect_cases_ps_ss', SUM(CASE WHEN is_zolpidem_suspect THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'zolpidem_suspect_cases_ps_only', SUM(CASE WHEN is_zolpidem_suspect_ps THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'other_zdrug_suspect_cases_ps_ss', SUM(CASE WHEN is_other_zdrug_suspect THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'other_zdrug_suspect_cases_ps_only', SUM(CASE WHEN is_other_zdrug_suspect_ps THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_ss_no_suspect_drug', SUM(CASE WHEN target_drug_group = 'no_suspect_drug' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_ss_no_target_zdrug_suspect', SUM(CASE WHEN target_drug_group = 'no_target_zdrug_suspect' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_ss_other_zdrug_only', SUM(CASE WHEN target_drug_group = 'other_zdrug_only' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_ss_zolpidem_only', SUM(CASE WHEN target_drug_group = 'zolpidem_only' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_ss_both_zolpidem_and_other_zdrug', SUM(CASE WHEN target_drug_group = 'both_zolpidem_and_other_zdrug' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_only_no_suspect_drug', SUM(CASE WHEN target_drug_group_ps = 'no_suspect_drug' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_only_no_target_zdrug_suspect', SUM(CASE WHEN target_drug_group_ps = 'no_target_zdrug_suspect' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_only_other_zdrug_only', SUM(CASE WHEN target_drug_group_ps = 'other_zdrug_only' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_only_zolpidem_only', SUM(CASE WHEN target_drug_group_ps = 'zolpidem_only' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
            UNION ALL
            SELECT 'group_ps_only_both_zolpidem_and_other_zdrug', SUM(CASE WHEN target_drug_group_ps = 'both_zolpidem_and_other_zdrug' THEN 1 ELSE 0 END)::BIGINT FROM signal_global
        )
        SELECT metric, value
        FROM metrics
        ORDER BY metric
        """
    ).df()
    signal_summary_df.to_csv(qc_summary_file, index=False, encoding="utf-8-sig")
    return case_index_file, signal_file, feature_file, qc_file, qc_summary_file


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

    con = duckdb.connect()
    try:
        create_global_case_index(con, case_files, start_year, end_year)
        create_global_signal_dataset(con, signal_files, start_year, end_year)
        create_global_feature_dataset(con, feature_files)
        case_file, signal_file, feature_file, qc_file, qc_summary_file = write_outputs(con, start_year, end_year)
    finally:
        con.close()

    print("Global datasets built successfully.")
    print("Global build reused cleaned quarterly datasets only.")
    print(f"Case index: {case_file}")
    print(f"Signal dataset: {signal_file}")
    print(f"Feature dataset: {feature_file}")
    print(f"QC summary: {qc_file}")
    print(f"Signal summary: {qc_summary_file}")


if __name__ == "__main__":
    main()
