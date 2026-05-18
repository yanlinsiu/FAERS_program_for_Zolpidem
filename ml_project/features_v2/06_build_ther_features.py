from __future__ import annotations

import argparse

import pandas as pd

from feature_v2_common import (
    QC_DIR,
    QUARTERLY_DIR,
    clean_caseid,
    ensure_feature_v2_dirs,
    iter_quarters,
    load_clean_case_table,
    missingness_table,
    quarter_token,
    summarize_frame,
    write_csv,
    write_parquet,
)


def _known(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def build_ther_features(year: int, quarter: str) -> pd.DataFrame:
    df = load_clean_case_table(year, quarter, "THER")
    for column in ["start_dt", "end_dt", "dur", "dur_cod"]:
        if column not in df.columns:
            df[column] = pd.NA
    df["caseid"] = clean_caseid(df["caseid"])
    df = df[df["caseid"] != ""].copy()
    df["has_start_dt_row"] = _known(df["start_dt"])
    df["has_end_dt_row"] = _known(df["end_dt"])
    df["duration_known_row"] = _known(df["dur"]) & _known(df["dur_cod"])

    grouped = df.groupby("caseid", as_index=False).agg(
        therapy_record_n=("caseid", "size"),
        has_start_dt=("has_start_dt_row", "max"),
        has_end_dt=("has_end_dt_row", "max"),
        duration_known=("duration_known_row", "max"),
    )
    grouped["therapy_record_n"] = grouped["therapy_record_n"].fillna(0).astype(int)
    for column in ["has_start_dt", "has_end_dt", "duration_known"]:
        grouped[column] = grouped[column].fillna(False).astype(bool)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build THER ML-v2 quarterly features.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    qc_frames = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        token = quarter_token(year, quarter)
        print(f"[ther-v2] {token}", flush=True)
        features = build_ther_features(year, quarter)
        write_parquet(features, QUARTERLY_DIR / f"ther_v2_{token}.parquet")
        qc_frames.append(summarize_frame(features, f"ther_v2_{token}"))
        qc_frames.append(missingness_table(features, f"ther_v2_{token}"))

    if qc_frames:
        write_csv(pd.concat(qc_frames, ignore_index=True), QC_DIR / "ther_v2_qc.csv")


if __name__ == "__main__":
    main()
