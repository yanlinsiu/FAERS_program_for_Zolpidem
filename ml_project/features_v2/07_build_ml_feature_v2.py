from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_v2_common import (
    DATASET_DIR,
    GLOBAL_DATASET_DIR,
    QC_DIR,
    QUARTERLY_DIR,
    clean_caseid,
    concat_existing,
    ensure_feature_v2_dirs,
    iter_quarters,
    missingness_table,
    period_token,
    quarter_token,
    summarize_frame,
    write_csv,
    write_parquet,
)


LEAKAGE_COLUMNS = {"fall_pt_list", "fall_pt_count", "fall_narrow_pt_count"}
PHENOTYPE_FEATURE_COLUMNS = [
    "pheno_sedation_somnolence",
    "pheno_consciousness_cognition",
    "pheno_dizziness_vertigo_syncope",
    "pheno_gait_balance_motor",
    "pheno_hypotension",
    "pheno_visual_disturbance",
]


def _feature_paths(prefix: str, start_year: int, end_year: int) -> list[Path]:
    return [
        QUARTERLY_DIR / f"{prefix}_{quarter_token(year, quarter)}.parquet"
        for year, quarter in iter_quarters(start_year, end_year)
    ]


def _load_feature(prefix: str, start_year: int, end_year: int) -> pd.DataFrame:
    df = concat_existing(_feature_paths(prefix, start_year, end_year))
    if df.empty:
        raise FileNotFoundError(f"No quarterly {prefix} files found in {QUARTERLY_DIR}")
    df["caseid"] = clean_caseid(df["caseid"])
    return df.drop_duplicates(subset="caseid", keep="last")


def _load_phenotype_features(token: str) -> pd.DataFrame | None:
    phenotype_file = (
        GLOBAL_DATASET_DIR.parent
        / "phenotypes"
        / f"phenotype_features_{token}_case.parquet"
    )
    if not phenotype_file.exists():
        return None

    phenotype_df = pd.read_parquet(phenotype_file)
    phenotype_df["caseid"] = clean_caseid(phenotype_df["caseid"])
    available_columns = [
        column for column in PHENOTYPE_FEATURE_COLUMNS if column in phenotype_df.columns
    ]
    if not available_columns:
        return None

    phenotype_df = phenotype_df[["caseid", *available_columns]].copy()
    phenotype_df = phenotype_df[phenotype_df["caseid"] != ""]
    phenotype_df = phenotype_df.drop_duplicates(subset="caseid", keep="last")
    for column in available_columns:
        phenotype_df[column] = phenotype_df[column].fillna(False).astype(bool)
    return phenotype_df


def build_ml_feature_v2(start_year: int, end_year: int) -> pd.DataFrame:
    token = period_token(start_year, end_year)
    signal_file = GLOBAL_DATASET_DIR / f"signal_dataset_{token}.parquet"
    drug_feature_file = GLOBAL_DATASET_DIR / f"drug_feature_{token}_case.parquet"
    if not signal_file.exists():
        raise FileNotFoundError(f"Signal dataset not found: {signal_file}")
    if not drug_feature_file.exists():
        raise FileNotFoundError(f"Drug feature dataset not found: {drug_feature_file}")

    signal_df = pd.read_parquet(signal_file)
    signal_df["caseid"] = clean_caseid(signal_df["caseid"])
    if "is_fall" not in signal_df.columns and "is_fall_narrow" in signal_df.columns:
        signal_df["is_fall"] = signal_df["is_fall_narrow"]
    stale_fall_cols = {
        col
        for col in signal_df.columns
        if col.startswith("is_fall_") or (col.startswith("fall_") and col not in LEAKAGE_COLUMNS)
    }
    signal_df = signal_df.drop(
        columns=[col for col in LEAKAGE_COLUMNS | stale_fall_cols if col in signal_df.columns]
    )
    signal_df = signal_df.drop_duplicates(subset="caseid", keep="last")

    drug_feature_df = pd.read_parquet(drug_feature_file)
    drug_feature_df["caseid"] = clean_caseid(drug_feature_df["caseid"])
    drug_feature_df = drug_feature_df.drop_duplicates(subset="caseid", keep="last")
    drug_overlap = sorted((set(signal_df.columns) & set(drug_feature_df.columns)) - {"caseid"})
    if drug_overlap:
        drug_feature_df = drug_feature_df.drop(columns=drug_overlap)

    merged = signal_df.merge(drug_feature_df, on="caseid", how="left")
    for prefix in ["demo_v2", "drug_role_v2", "indi_v2", "rpsr_v2", "ther_v2"]:
        feature_df = _load_feature(prefix, start_year, end_year)
        overlap = sorted((set(merged.columns) & set(feature_df.columns)) - {"caseid"})
        if overlap:
            feature_df = feature_df.drop(columns=overlap)
        merged = merged.merge(feature_df, on="caseid", how="left")

    phenotype_df = _load_phenotype_features(token)
    if phenotype_df is not None:
        overlap = sorted((set(merged.columns) & set(phenotype_df.columns)) - {"caseid"})
        if overlap:
            phenotype_df = phenotype_df.drop(columns=overlap)
        merged = merged.merge(phenotype_df, on="caseid", how="left")

    fill_false_cols = [
        col
        for col in merged.columns
        if col.startswith(("has_", "indi_", "pheno_", "zolpidem_", "other_zdrug_"))
    ]
    fill_false_cols.extend(
        col for col in ["event_date_known", "duration_known"] if col in merged.columns
    )
    for column in fill_false_cols:
        if merged[column].dtype == "bool" or merged[column].dropna().isin([True, False]).all():
            merged[column] = merged[column].astype("boolean").fillna(False).astype(bool)

    numeric_fill_zero = [
        "ps_drug_n",
        "ss_drug_n",
        "concomitant_drug_n",
        "interacting_drug_n",
        "indi_n",
        "distinct_indi_n",
        "indi_mapped_n",
        "indi_unmapped_n",
        "therapy_record_n",
    ]
    for column in numeric_fill_zero:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    for column in ["rept_cod", "e_sub", "reporter_country", "occr_country", "rpsr_cod"]:
        if column in merged.columns:
            merged[column] = merged[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ML-v2 quarterly features into one modeling table.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    token = period_token(args.start_year, args.end_year)
    features = build_ml_feature_v2(args.start_year, args.end_year)
    output_path = DATASET_DIR / f"ml_feature_v2_{token}.parquet"
    qc_path = QC_DIR / f"ml_feature_v2_qc_{token}.csv"
    write_parquet(features, output_path)
    qc_df = pd.concat(
        [
            summarize_frame(features, f"ml_feature_v2_{token}"),
            missingness_table(features, f"ml_feature_v2_{token}"),
        ],
        ignore_index=True,
    )
    write_csv(qc_df, qc_path)
    print(f"Saved ML-v2 feature table to: {output_path}")
    print(f"Saved ML-v2 QC to: {qc_path}")


if __name__ == "__main__":
    main()

