from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
FAERS_PROJECT_ROOT = PROJECT_ROOT / "faers_project"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(FAERS_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(FAERS_PROJECT_ROOT))

from drug_dictionary import build_zdrug_exposure_terms, normalize_dictionary_term
from feature_v2_common import (
    QC_DIR,
    QUARTERLY_DIR,
    boundary_pattern,
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


ZOLPIDEM_TERMS, OTHER_ZDRUG_TERMS = build_zdrug_exposure_terms()


def build_drug_role_features(year: int, quarter: str) -> pd.DataFrame:
    df = load_clean_case_table(year, quarter, "DRUG")
    required = ["caseid", "drugname", "prod_ai", "role_cod"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"DRUG missing columns: {missing}")

    df["caseid"] = clean_caseid(df["caseid"])
    df["drugname"] = normalize_text(df["drugname"]).map(normalize_dictionary_term)
    df["prod_ai"] = normalize_text(df["prod_ai"]).map(normalize_dictionary_term)
    df["role_cod"] = normalize_text(df["role_cod"])
    df = df[df["caseid"] != ""].copy()
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))].copy()

    text = df["drugname"] + " " + df["prod_ai"]
    zolpidem_pattern = boundary_pattern(ZOLPIDEM_TERMS)
    other_zdrug_pattern = boundary_pattern(OTHER_ZDRUG_TERMS)
    df["is_zolpidem_hit"] = text.str.contains(zolpidem_pattern, regex=True, na=False)
    df["is_other_zdrug_hit"] = text.str.contains(other_zdrug_pattern, regex=True, na=False)
    df["is_ps"] = df["role_cod"].eq("PS")
    df["is_ss"] = df["role_cod"].eq("SS")
    df["is_c"] = df["role_cod"].eq("C")
    df["is_i"] = df["role_cod"].eq("I")
    df["is_suspect"] = df["role_cod"].isin(["PS", "SS"])
    df["zolpidem_as_ps_row"] = df["is_zolpidem_hit"] & df["is_ps"]
    df["zolpidem_as_suspect_row"] = df["is_zolpidem_hit"] & df["is_suspect"]
    df["other_zdrug_as_suspect_row"] = df["is_other_zdrug_hit"] & df["is_suspect"]

    grouped = df.groupby("caseid", as_index=False).agg(
        ps_drug_n=("is_ps", "sum"),
        ss_drug_n=("is_ss", "sum"),
        concomitant_drug_n=("is_c", "sum"),
        interacting_drug_n=("is_i", "sum"),
        has_ps_drug=("is_ps", "max"),
        has_ss_drug=("is_ss", "max"),
        zolpidem_as_ps=("zolpidem_as_ps_row", "max"),
        zolpidem_as_suspect=("zolpidem_as_suspect_row", "max"),
        other_zdrug_as_suspect=("other_zdrug_as_suspect_row", "max"),
    )

    count_cols = ["ps_drug_n", "ss_drug_n", "concomitant_drug_n", "interacting_drug_n"]
    for column in count_cols:
        grouped[column] = grouped[column].fillna(0).astype(int)
    for column in [col for col in grouped.columns if col.startswith("has_") or col.endswith("_suspect") or col.endswith("_ps")]:
        if column != "caseid":
            grouped[column] = grouped[column].fillna(False).astype(bool)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DRUG role ML-v2 quarterly features.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    qc_frames = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        token = quarter_token(year, quarter)
        print(f"[drug-role-v2] {token}", flush=True)
        features = build_drug_role_features(year, quarter)
        write_parquet(features, QUARTERLY_DIR / f"drug_role_v2_{token}.parquet")
        qc_frames.append(summarize_frame(features, f"drug_role_v2_{token}"))
        qc_frames.append(missingness_table(features, f"drug_role_v2_{token}"))

    if qc_frames:
        write_csv(pd.concat(qc_frames, ignore_index=True), QC_DIR / "drug_role_v2_qc.csv")


if __name__ == "__main__":
    main()
