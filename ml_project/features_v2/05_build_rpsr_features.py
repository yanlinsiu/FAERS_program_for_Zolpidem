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
    normalize_text,
    quarter_token,
    summarize_frame,
    write_csv,
    write_parquet,
)


def build_rpsr_features(year: int, quarter: str) -> pd.DataFrame:
    df = load_clean_case_table(year, quarter, "RPSR")
    if "rpsr_cod" not in df.columns:
        raise ValueError("RPSR missing required column: rpsr_cod")
    df["caseid"] = clean_caseid(df["caseid"])
    df["rpsr_cod"] = normalize_text(df["rpsr_cod"]).replace("", "unknown")
    df = df[df["caseid"] != ""].copy()

    grouped = (
        df[["caseid", "rpsr_cod"]]
        .drop_duplicates()
        .sort_values(["caseid", "rpsr_cod"])
        .groupby("caseid", as_index=False)
        .agg(rpsr_cod=("rpsr_cod", lambda values: "|".join(values)))
    )
    grouped["has_rpsr"] = grouped["rpsr_cod"].ne("")
    grouped["rpsr_cod"] = grouped["rpsr_cod"].replace("", "unknown")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RPSR ML-v2 quarterly features.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    qc_frames = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        token = quarter_token(year, quarter)
        print(f"[rpsr-v2] {token}", flush=True)
        features = build_rpsr_features(year, quarter)
        write_parquet(features, QUARTERLY_DIR / f"rpsr_v2_{token}.parquet")
        qc_frames.append(summarize_frame(features, f"rpsr_v2_{token}"))
        qc_frames.append(missingness_table(features, f"rpsr_v2_{token}"))

    if qc_frames:
        write_csv(pd.concat(qc_frames, ignore_index=True), QC_DIR / "rpsr_v2_qc.csv")


if __name__ == "__main__":
    main()
