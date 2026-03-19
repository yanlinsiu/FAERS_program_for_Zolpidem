from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_common import (
    OUTCOME_SPECS,
    build_stratified_rows,
    describe_signal,
    ensure_output_dir,
    load_signal_dataset,
    make_overall_qc,
    ror_prr_from_counts,
    save_tables,
    two_by_two_counts,
)


PRIMARY_EXPOSURE = "is_zolpidem_suspect"
SENSITIVITY_EXPOSURE = "is_zolpidem_suspect_ps"


def build_signal_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)

    result_rows = []
    qc_frames = []
    stratified_frames = []

    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in signal_df.columns:
            continue

        for analysis_name, exposure_col, suspect_col, group_col in [
            ("primary_ps_ss", PRIMARY_EXPOSURE, "suspect_role_any", "target_drug_group"),
            ("sensitivity_ps_only", SENSITIVITY_EXPOSURE, "suspect_role_any_ps", "target_drug_group_ps"),
        ]:
            subset = signal_df[signal_df[suspect_col].fillna(False).astype(bool)].copy()
            subset = subset[subset[group_col] != "both_zolpidem_and_other_zdrug"].copy()

            counts = two_by_two_counts(subset[exposure_col], subset[outcome_col])
            metrics = ror_prr_from_counts(**counts)
            result_rows.append(
                {
                    "analysis": analysis_name,
                    "exposure_definition": exposure_col,
                    "outcome_name": outcome_name,
                    "outcome_definition": outcome_label,
                    "comparison_group": "all_other_suspect_drugs_excluding_mixed_zdrug_cases",
                    "conclusion": describe_signal(metrics),
                    **metrics,
                }
            )

            qc_df = make_overall_qc(subset, analysis_name, exposure_col, outcome_col)
            qc_df["outcome_name"] = outcome_name
            qc_df["outcome_definition"] = outcome_label
            group_counts = (
                subset.groupby(group_col, dropna=False)
                .agg(
                    n_cases=("caseid", "count"),
                    n_outcome=(outcome_col, "sum"),
                )
                .reset_index()
                .rename(columns={group_col: "drug_group"})
            )
            group_counts["analysis"] = analysis_name
            group_counts["outcome_name"] = outcome_name
            group_counts["outcome_definition"] = outcome_label
            qc_frames.append(qc_df)
            qc_frames.append(group_counts)

            stratified_df = build_stratified_rows(
                subset,
                analysis_name=analysis_name,
                exposure_col=exposure_col,
                outcome_col=outcome_col,
                outcome_name=outcome_name,
                outcome_label=outcome_label,
            )
            if not stratified_df.empty:
                stratified_frames.append(stratified_df)

    result_df = pd.DataFrame(result_rows)
    qc_result_df = pd.concat(qc_frames, ignore_index=True, sort=False)
    stratified_result_df = (
        pd.concat(stratified_frames, ignore_index=True, sort=False)
        if stratified_frames
        else pd.DataFrame()
    )

    output_root = ensure_output_dir(output_dir)
    save_tables(
        result_df,
        qc_result_df,
        output_root / "01_signal_analysis_results.csv",
        output_root / "01_signal_analysis_qc.csv",
    )
    stratified_result_df.to_csv(output_root / "01_signal_analysis_stratified.csv", index=False, encoding="utf-8-sig")
    return result_df, qc_result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zolpidem vs fall signal analysis on FAERS case-level data")
    parser.add_argument("--signal-root", default=r"D:\program_FAERS\OUTPUT", help="Directory containing signal_dataset_*.parquet")
    parser.add_argument("--output-dir", default=r"D:\program_FAERS\OUTPUT\analysis", help="Directory to save result and QC tables")
    args = parser.parse_args()
    results, qc = build_signal_analysis(args.signal_root, args.output_dir)
    print(results.to_string(index=False))
    print("saved QC rows:", len(qc))
