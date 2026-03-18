from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_common import ensure_output_dir, load_signal_dataset, ror_prr_from_counts, save_tables, two_by_two_counts


OUTCOME_COL = "is_fall"


def _build_comparator_subset(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    allowed = {"zolpidem_only", "other_zdrug_only"}
    subset = df[df[group_col].isin(allowed)].copy()
    return subset


def build_comparative_analysis(signal_root: str | Path | None = None, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = load_signal_dataset(signal_root)

    result_rows = []
    qc_rows = []
    configs = [
        ("primary_ps_ss", "target_drug_group", "zolpidem_only"),
        ("sensitivity_ps_only", "target_drug_group_ps", "zolpidem_only"),
    ]

    for analysis_name, group_col, zolpidem_value in configs:
        subset = _build_comparator_subset(signal_df, group_col)
        exposed = subset[group_col].eq(zolpidem_value)
        counts = two_by_two_counts(exposed, subset[OUTCOME_COL])
        metrics = ror_prr_from_counts(**counts)
        result_rows.append(
            {
                "analysis": analysis_name,
                "comparison": "zolpidem_only_vs_other_zdrug_only",
                "outcome_definition": OUTCOME_COL,
                **metrics,
            }
        )

        for drug_group, frame in subset.groupby(group_col, dropna=False):
            qc_rows.append(
                {
                    "analysis": analysis_name,
                    "drug_group": drug_group,
                    "n_cases": int(len(frame)),
                    "n_fall": int(frame[OUTCOME_COL].fillna(False).astype(bool).sum()),
                    "fall_reporting_rate": float(frame[OUTCOME_COL].fillna(False).astype(bool).mean()) if len(frame) else None,
                    "n_female": int(frame["sex_clean"].eq("F").sum()) if "sex_clean" in frame.columns else None,
                    "n_age_75_plus": int(frame["age_group"].isin(["75-84", ">=85"]).sum()) if "age_group" in frame.columns else None,
                }
            )

    result_df = pd.DataFrame(result_rows)
    qc_df = pd.DataFrame(qc_rows)
    output_root = ensure_output_dir(output_dir)
    save_tables(
        result_df,
        qc_df,
        output_root / "02_comparative_analysis_results.csv",
        output_root / "02_comparative_analysis_qc.csv",
    )
    return result_df, qc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative analysis: zolpidem vs other Z-drugs")
    parser.add_argument("--signal-root", default=r"D:\program_FAERS\OUTPUT", help="Directory containing signal_dataset_*.parquet")
    parser.add_argument("--output-dir", default=r"D:\program_FAERS\OUTPUT\analysis", help="Directory to save result and QC tables")
    args = parser.parse_args()
    results, qc = build_comparative_analysis(args.signal_root, args.output_dir)
    print(results.to_string(index=False))
    print("saved QC rows:", len(qc))
