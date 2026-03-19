from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_common import (
    OUTCOME_SPECS,
    describe_signal,
    ensure_output_dir,
    feature_mask,
    merge_signal_and_feature,
    ror_prr_from_counts,
    save_tables,
    two_by_two_counts,
)


EXPOSURE_COL = "is_zolpidem_suspect"
EXPOSURE_COL_PS = "is_zolpidem_suspect_ps"

FEATURE_SPECS = [
    ("age_group", "65-74", "demographic"),
    ("age_group", "75-84", "demographic"),
    ("age_group", ">=85", "demographic"),
    ("sex_clean", "F", "demographic"),
    ("sex_clean", "M", "demographic"),
    ("serious", True, "severity"),
    ("polypharmacy_5", True, "medication_burden"),
    ("is_benzo", True, "co_medication"),
    ("is_antidepressant", True, "co_medication"),
    ("is_antipsychotic", True, "co_medication"),
    ("is_opioid", True, "co_medication"),
    ("is_antiepileptic", True, "co_medication"),
]


def _build_feature_rows(
    df: pd.DataFrame,
    analysis_name: str,
    outcome_name: str,
    outcome_col: str,
    outcome_label: str,
) -> pd.DataFrame:
    rows = []
    for column, value, domain in FEATURE_SPECS:
        if column not in df.columns:
            continue
        mask = feature_mask(df, column, value)
        counts = two_by_two_counts(mask, df[outcome_col])
        metrics = ror_prr_from_counts(**counts)
        exposed_n = int(mask.sum())
        outcome_n = int((mask & df[outcome_col].fillna(False).astype(bool)).sum())
        rows.append(
            {
                "analysis": analysis_name,
                "outcome_name": outcome_name,
                "outcome_definition": outcome_label,
                "feature_domain": domain,
                "feature_name": f"{column}={value}",
                "n_feature_positive": exposed_n,
                "n_feature_positive_outcome": outcome_n,
                "outcome_reporting_rate": (outcome_n / exposed_n) if exposed_n else None,
                "conclusion": describe_signal(metrics),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_feature_analysis(signal_root: str | Path | None = None, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_df = merge_signal_and_feature(signal_root)

    result_frames = []
    qc_rows = []
    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in merged_df.columns:
            continue

        for analysis_name, exposure_col in [
            ("primary_ps_ss", EXPOSURE_COL),
            ("sensitivity_ps_only", EXPOSURE_COL_PS),
        ]:
            subset = merged_df[merged_df[exposure_col].fillna(False).astype(bool)].copy()
            result_frames.append(
                _build_feature_rows(
                    subset,
                    analysis_name=analysis_name,
                    outcome_name=outcome_name,
                    outcome_col=outcome_col,
                    outcome_label=outcome_label,
                )
            )
            qc_rows.append(
                {
                    "analysis": analysis_name,
                    "outcome_name": outcome_name,
                    "n_zolpidem_exposed": int(len(subset)),
                    "n_outcome": int(subset[outcome_col].fillna(False).astype(bool).sum()),
                    "missing_age_group": int(subset["age_group"].isna().sum()) if "age_group" in subset.columns else None,
                    "missing_sex_clean": int(subset["sex_clean"].isna().sum()) if "sex_clean" in subset.columns else None,
                    "missing_serious": int(subset["serious"].isna().sum()) if "serious" in subset.columns else None,
                    "n_polypharmacy_5": int(subset["polypharmacy_5"].fillna(False).astype(bool).sum()) if "polypharmacy_5" in subset.columns else None,
                    "n_serious": int(subset["serious"].fillna(False).astype(bool).sum()) if "serious" in subset.columns else None,
                    "n_benzo": int(subset["is_benzo"].fillna(False).astype(bool).sum()) if "is_benzo" in subset.columns else None,
                    "n_antidepressant": int(subset["is_antidepressant"].fillna(False).astype(bool).sum()) if "is_antidepressant" in subset.columns else None,
                    "n_antipsychotic": int(subset["is_antipsychotic"].fillna(False).astype(bool).sum()) if "is_antipsychotic" in subset.columns else None,
                    "n_opioid": int(subset["is_opioid"].fillna(False).astype(bool).sum()) if "is_opioid" in subset.columns else None,
                    "n_antiepileptic": int(subset["is_antiepileptic"].fillna(False).astype(bool).sum()) if "is_antiepileptic" in subset.columns else None,
                }
            )

    result_df = pd.concat(result_frames, ignore_index=True)
    result_df = result_df.sort_values(
        ["analysis", "outcome_name", "ror"],
        ascending=[True, True, False],
        na_position="last",
    )
    qc_df = pd.DataFrame(qc_rows)
    output_root = ensure_output_dir(output_dir)
    save_tables(
        result_df,
        qc_df,
        output_root / "03_feature_analysis_results.csv",
        output_root / "03_feature_analysis_qc.csv",
    )
    return result_df, qc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature analysis: who is more likely to have fall reports among zolpidem reports")
    parser.add_argument("--signal-root", default=r"D:\program_FAERS\OUTPUT", help="Directory containing signal and drug feature parquet files")
    parser.add_argument("--output-dir", default=r"D:\program_FAERS\OUTPUT\analysis", help="Directory to save result and QC tables")
    args = parser.parse_args()
    results, qc = build_feature_analysis(args.signal_root, args.output_dir)
    print(results.to_string(index=False))
    print("saved QC rows:", len(qc))
