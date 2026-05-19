from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb

try:
    from .config import GLOBAL_OUTPUT_ROOT, OUTPUT_ROOT
except ImportError:
    from config import GLOBAL_OUTPUT_ROOT, OUTPUT_ROOT


YEAR_Q_PATTERN = re.compile(r"(\d{4})q([1-4])", re.IGNORECASE)


def discover_quarterly_files(pattern: str, output_root: Path = OUTPUT_ROOT) -> list[Path]:
    files: list[Path] = []
    for year_dir in sorted(output_root.iterdir()):
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
    schema_df = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)",
        [case_file_list],
    ).df()
    available_cols = set(schema_df["column_name"].astype(str).str.lower())
    reporter_country_expr = (
        "UPPER(NULLIF(TRIM(CAST(reporter_country AS VARCHAR)), '')) AS reporter_country"
        if "reporter_country" in available_cols
        else "'UNKNOWN' AS reporter_country"
    )
    occr_country_expr = (
        "UPPER(NULLIF(TRIM(CAST(occr_country AS VARCHAR)), '')) AS occr_country"
        if "occr_country" in available_cols
        else "'UNKNOWN' AS occr_country"
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE case_base_all AS
        SELECT
            TRIM(CAST(caseid AS VARCHAR)) AS caseid,
            CAST(primaryid AS BIGINT) AS primaryid,
            CAST(fda_dt AS DATE) AS fda_dt,
            {reporter_country_expr},
            {occr_country_expr},
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
                reporter_country,
                occr_country,
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
            COALESCE(reporter_country, 'UNKNOWN') AS reporter_country,
            COALESCE(occr_country, 'UNKNOWN') AS occr_country,
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
    global_output_root: Path = GLOBAL_OUTPUT_ROOT,
) -> tuple[Path, Path, Path, Path, Path]:
    global_dataset_dir = global_output_root / "datasets"
    global_qc_dir = global_output_root / "qc"

    global_dataset_dir.mkdir(parents=True, exist_ok=True)
    global_qc_dir.mkdir(parents=True, exist_ok=True)

    period_token = f"{start_year}_{end_year}"
    case_index_file = global_dataset_dir / f"global_case_index_{period_token}.parquet"
    signal_file = global_dataset_dir / f"signal_dataset_{period_token}.parquet"
    feature_file = global_dataset_dir / f"drug_feature_{period_token}_case.parquet"
    qc_file = global_qc_dir / f"global_dataset_qc_{period_token}.csv"
    qc_summary_file = global_qc_dir / f"global_signal_summary_{period_token}.csv"

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


def build_global_datasets(
    start_year: int = 2004,
    end_year: int = 2025,
    input_output_root: Path = OUTPUT_ROOT,
    global_output_root: Path = GLOBAL_OUTPUT_ROOT,
) -> dict[str, Path]:
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    input_output_root = Path(input_output_root).resolve()
    global_output_root = Path(global_output_root).resolve()

    case_files = discover_quarterly_files("case_base_dataset_*q*.parquet", input_output_root)
    signal_files = discover_quarterly_files("signal_dataset_*q*.parquet", input_output_root)
    feature_files = discover_quarterly_files("drug_feature_*q*_case.parquet", input_output_root)

    if not case_files:
        raise FileNotFoundError(
            f"No quarterly case_base_dataset parquet files found in {input_output_root}/*/quarterly."
        )
    if not signal_files:
        raise FileNotFoundError(
            f"No quarterly signal_dataset parquet files found in {input_output_root}/*/quarterly."
        )

    con = duckdb.connect()
    try:
        create_global_case_index(con, case_files, start_year, end_year)
        create_global_signal_dataset(con, signal_files, start_year, end_year)
        create_global_feature_dataset(con, feature_files)
        case_file, signal_file, feature_file, qc_file, qc_summary_file = write_outputs(
            con,
            start_year,
            end_year,
            global_output_root=global_output_root,
        )
    finally:
        con.close()

    return {
        "input_output_root": input_output_root,
        "global_output_root": global_output_root,
        "case_index": case_file,
        "signal_dataset": signal_file,
        "feature_dataset": feature_file,
        "qc_summary": qc_file,
        "signal_summary": qc_summary_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 2004-2025 global deduplicated FAERS datasets with DuckDB."
    )
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--input-output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Cleaned yearly output root to read from. Defaults to OUTPUT.",
    )
    parser.add_argument(
        "--global-output-root",
        type=Path,
        default=GLOBAL_OUTPUT_ROOT,
        help="Global output root to write to. Defaults to OUTPUT_GLOBAL.",
    )
    args = parser.parse_args()

    outputs = build_global_datasets(
        start_year=args.start_year,
        end_year=args.end_year,
        input_output_root=args.input_output_root,
        global_output_root=args.global_output_root,
    )

    print("Global datasets built successfully.")
    print("Global build reused cleaned quarterly datasets only.")
    print(f"Input cleaned output root: {outputs['input_output_root']}")
    print(f"Global output root: {outputs['global_output_root']}")
    print(f"Case index: {outputs['case_index']}")
    print(f"Signal dataset: {outputs['signal_dataset']}")
    print(f"Feature dataset: {outputs['feature_dataset']}")
    print(f"QC summary: {outputs['qc_summary']}")
    print(f"Signal summary: {outputs['signal_summary']}")


if __name__ == "__main__":
    main()
