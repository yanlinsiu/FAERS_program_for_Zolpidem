from __future__ import annotations

import argparse

import pandas as pd

from feature_v2_common import (
    AUDIT_DIR,
    TABLES,
    clean_caseid,
    ensure_feature_v2_dirs,
    iter_quarters,
    period_token,
    read_raw_table,
    write_csv,
)


KEY_COLUMNS = {
    "DEMO": [
        "primaryid",
        "caseid",
        "age",
        "age_cod",
        "wt",
        "wt_cod",
        "rept_cod",
        "e_sub",
        "reporter_country",
        "occr_country",
        "event_dt",
        "fda_dt",
    ],
    "DRUG": ["primaryid", "caseid", "drugname", "prod_ai", "role_cod"],
    "INDI": ["primaryid", "caseid", "indi_pt"],
    "RPSR": ["primaryid", "caseid", "rpsr_cod"],
    "THER": ["primaryid", "caseid", "start_dt", "end_dt", "dur", "dur_cod"],
}


def audit_table(year: int, quarter: str, table_name: str) -> list[dict[str, object]]:
    try:
        df = read_raw_table(year, quarter, table_name)
    except FileNotFoundError:
        return [
            {
                "year": year,
                "quarter": quarter,
                "table": table_name,
                "column": "__file__",
                "exists": False,
                "n_rows": 0,
                "n_caseid": 0,
                "missing_rate": None,
                "blank_rate": None,
            }
        ]

    n_rows = len(df)
    n_caseid = int(clean_caseid(df["caseid"]).ne("").sum()) if "caseid" in df.columns else 0
    rows = []
    for column in KEY_COLUMNS[table_name]:
        exists = column in df.columns
        missing_rate = None
        blank_rate = None
        if exists and n_rows:
            series = df[column]
            missing_rate = float(series.isna().mean())
            blank_rate = float(series.fillna("").astype(str).str.strip().eq("").mean())
        rows.append(
            {
                "year": year,
                "quarter": quarter,
                "table": table_name,
                "column": column,
                "exists": exists,
                "n_rows": n_rows,
                "n_caseid": n_caseid,
                "missing_rate": missing_rate,
                "blank_rate": blank_rate,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ML-v2 source fields.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    rows: list[dict[str, object]] = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        print(f"[audit] {year} {quarter}", flush=True)
        for table_name in TABLES:
            rows.extend(audit_table(year, quarter, table_name))

    audit_df = pd.DataFrame(rows)
    output_path = AUDIT_DIR / f"field_audit_{period_token(args.start_year, args.end_year)}.csv"
    write_csv(audit_df, output_path)
    print(f"Saved field audit to: {output_path}")


if __name__ == "__main__":
    main()
