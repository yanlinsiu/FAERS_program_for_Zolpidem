from __future__ import annotations

import argparse

import pandas as pd

from feature_v2_common import (
    QC_DIR,
    QUARTERLY_DIR,
    clean_caseid,
    ensure_feature_v2_dirs,
    iter_quarters,
    load_clean_demo,
    missingness_table,
    quarter_token,
    summarize_frame,
    write_csv,
    write_parquet,
)


def build_demo_features(year: int, quarter: str) -> pd.DataFrame:
    df = load_clean_demo(year, quarter)
    out = pd.DataFrame()
    out["caseid"] = clean_caseid(df["caseid"])
    out["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    out["age_group"] = df["age_group"].where(df["age_group"].notna(), "unknown").astype(str).str.strip()
    out["rept_cod"] = df.get("rept_cod", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str).str.strip().str.upper().replace("", "unknown")
    out["e_sub"] = df.get("e_sub", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str).str.strip().str.upper().replace("", "unknown")
    out["reporter_country"] = df.get("reporter_country", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str).str.strip().str.upper().replace("", "unknown")
    out["occr_country"] = df.get("occr_country", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str).str.strip().str.upper().replace("", "unknown")
    event_dt = df.get("event_dt", pd.Series(pd.NA, index=df.index))
    out["event_date_known"] = event_dt.notna() & event_dt.astype(str).str.strip().ne("")
    return out.drop_duplicates(subset="caseid", keep="last")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DEMO ML-v2 quarterly features.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    qc_frames = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        token = quarter_token(year, quarter)
        print(f"[demo-v2] {token}", flush=True)
        features = build_demo_features(year, quarter)
        output_path = QUARTERLY_DIR / f"demo_v2_{token}.parquet"
        write_parquet(features, output_path)
        qc_frames.append(summarize_frame(features, f"demo_v2_{token}"))
        qc_frames.append(missingness_table(features, f"demo_v2_{token}"))

    if qc_frames:
        write_csv(pd.concat(qc_frames, ignore_index=True), QC_DIR / "demo_v2_qc.csv")


if __name__ == "__main__":
    main()
