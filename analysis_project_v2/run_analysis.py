from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .adjusted_models import build_adjusted_analysis
    from .config import (
        EXPLORATORY_SIGNAL_SPECS,
        FEATURE_SPECS,
        GLOBAL_DATASET_DIR,
        GLOBAL_OUTPUT_DIR,
        GROUP_COMPARISON_SPECS,
        OUTCOMES_BY_NAME,
        SIGNAL_SPECS,
    )
    from .data import load_analysis_frame, resolve_dataset_bundle
    from .report_tables import build_exploratory_summary, build_primary_summary
    from .sensitivity_adjusted import build_sensitivity_adjusted_analysis
    from .signal_metrics import (
        add_signal_classification,
        apply_bh_fdr,
        feature_mask,
        signal_metrics,
        two_by_two_counts,
    )
except ImportError:
    from adjusted_models import build_adjusted_analysis
    from config import (
        EXPLORATORY_SIGNAL_SPECS,
        FEATURE_SPECS,
        GLOBAL_DATASET_DIR,
        GLOBAL_OUTPUT_DIR,
        GROUP_COMPARISON_SPECS,
        OUTCOMES_BY_NAME,
        SIGNAL_SPECS,
    )
    from data import load_analysis_frame, resolve_dataset_bundle
    from report_tables import build_exploratory_summary, build_primary_summary
    from sensitivity_adjusted import build_sensitivity_adjusted_analysis
    from signal_metrics import (
        add_signal_classification,
        apply_bh_fdr,
        feature_mask,
        signal_metrics,
        two_by_two_counts,
    )


def _metrics_row(base: dict[str, Any], exposed: pd.Series, outcome: pd.Series) -> dict[str, Any]:
    counts = two_by_two_counts(exposed, outcome)
    return {**base, **signal_metrics(**counts)}


def build_signal_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for spec in SIGNAL_SPECS:
        subset = df[df[spec.suspect_column].fillna(False).astype(bool)].copy()
        if spec.exclude_group:
            subset = subset[subset[spec.group_column] != spec.exclude_group].copy()
        for outcome_name in spec.outcome_names:
            outcome = OUTCOMES_BY_NAME[outcome_name]
            rows.append(
                _metrics_row(
                    {
                        "analysis_tier": spec.tier,
                        "analysis": spec.analysis,
                        "comparison": spec.comparison,
                        "exposure_definition": spec.exposure_column,
                        "outcome_name": outcome.name,
                        "outcome_definition": outcome.label,
                    },
                    subset[spec.exposure_column],
                    subset[outcome.column],
                )
            )
            qc_rows.append(
                {
                    "section": "signal",
                    "analysis_tier": spec.tier,
                    "analysis": spec.analysis,
                    "outcome_name": outcome.name,
                    "n_cases": int(len(subset)),
                    "n_outcome": int(subset[outcome.column].sum()),
                    "n_exposed": int(subset[spec.exposure_column].sum()),
                }
            )

    for spec in GROUP_COMPARISON_SPECS:
        subset = df[df[spec.group_column].isin([spec.exposed_value, spec.reference_value])].copy()
        exposed = subset[spec.group_column].eq(spec.exposed_value)
        for outcome_name in spec.outcome_names:
            outcome = OUTCOMES_BY_NAME[outcome_name]
            rows.append(
                _metrics_row(
                    {
                        "analysis_tier": spec.tier,
                        "analysis": spec.analysis,
                        "comparison": spec.comparison,
                        "exposure_definition": f"{spec.group_column}={spec.exposed_value}",
                        "outcome_name": outcome.name,
                        "outcome_definition": outcome.label,
                    },
                    exposed,
                    subset[outcome.column],
                )
            )
            qc_rows.append(
                {
                    "section": "signal",
                    "analysis_tier": spec.tier,
                    "analysis": spec.analysis,
                    "outcome_name": outcome.name,
                    "n_cases": int(len(subset)),
                    "n_outcome": int(subset[outcome.column].sum()),
                    "n_exposed": int(exposed.sum()),
                }
            )

    signal_df = add_signal_classification(pd.DataFrame(rows), apply_stability_gate=False)
    primary_df = signal_df[signal_df["analysis_tier"].eq("primary")].copy()
    sensitivity_df = signal_df[signal_df["analysis_tier"].eq("sensitivity")].copy()
    return primary_df, sensitivity_df, signal_df, pd.DataFrame(qc_rows)


def build_exploratory_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    for spec in EXPLORATORY_SIGNAL_SPECS:
        subset = df[df[spec.exposure_column].fillna(False).astype(bool)].copy()
        for outcome_name in spec.outcome_names:
            outcome = OUTCOMES_BY_NAME[outcome_name]
            for feature in FEATURE_SPECS:
                if feature.column not in subset.columns:
                    continue
                mask = feature_mask(subset, feature.column, feature.value)
                rows.append(
                    _metrics_row(
                        {
                            "analysis_tier": "exploratory",
                            "analysis": spec.analysis,
                            "comparison": spec.comparison,
                            "outcome_name": outcome.name,
                            "outcome_definition": outcome.label,
                            "feature_domain": feature.domain,
                            "feature_name": f"{feature.column}={feature.value}",
                            "feature_label": feature.label,
                        },
                        mask,
                        subset[outcome.column],
                    )
                )
            qc_rows.append(
                {
                    "section": "exploratory",
                    "analysis_tier": "exploratory",
                    "analysis": spec.analysis,
                    "outcome_name": outcome.name,
                    "n_zolpidem_exposed_cases": int(len(subset)),
                    "n_outcome": int(subset[outcome.column].sum()),
                }
            )

    exploratory_df = pd.DataFrame(rows)
    exploratory_df = add_signal_classification(exploratory_df, apply_stability_gate=True)
    exploratory_df = apply_bh_fdr(exploratory_df)
    exploratory_df["is_stable_signal"] = (
        exploratory_df["is_stable_signal"].astype(bool)
        & exploratory_df["fdr_significant"].fillna(False).astype(bool)
    )
    exploratory_df["conclusion"] = exploratory_df["is_stable_signal"].map(
        {True: "signal_detected", False: "no_stable_signal"}
    )
    return exploratory_df, pd.DataFrame(qc_rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(period_token: str | None, dataset_dir: Path, output_dir: Path) -> dict[str, Path]:
    bundle = resolve_dataset_bundle(dataset_dir=dataset_dir, period_token=period_token)
    df = load_analysis_frame(bundle)

    primary_df, sensitivity_df, all_signal_df, signal_qc = build_signal_tables(df)
    exploratory_df, exploratory_qc = build_exploratory_table(df)
    adjusted_df, adjusted_qc = build_adjusted_analysis(df)
    sensitivity_tables, sensitivity_all_summary, sensitivity_qc = build_sensitivity_adjusted_analysis(df)

    adjusted_primary = adjusted_df[adjusted_df["analysis_tier"].eq("primary")].copy()
    adjusted_sensitivity = adjusted_df[adjusted_df["analysis_tier"].eq("sensitivity")].copy()
    summary_primary = build_primary_summary(pd.concat([primary_df, sensitivity_df], ignore_index=True), adjusted_df)
    summary_exploratory = build_exploratory_summary(exploratory_df)

    qc_df = pd.concat(
        [
            signal_qc,
            exploratory_qc,
            adjusted_qc,
            sensitivity_qc,
            pd.DataFrame(
                [
                    {
                        "section": "dataset",
                        "period_token": bundle.period_token,
                        "signal_file": str(bundle.signal_file),
                        "feature_file": str(bundle.feature_file),
                        "n_rows": int(len(df)),
                        "n_columns": int(len(df.columns)),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    outputs = {
        "signal_primary": output_dir / "signal_primary.csv",
        "signal_sensitivity": output_dir / "signal_sensitivity.csv",
        "signal_exploratory": output_dir / "signal_exploratory.csv",
        "adjusted_primary": output_dir / "adjusted_primary.csv",
        "adjusted_sensitivity": output_dir / "adjusted_sensitivity.csv",
        "summary_primary": output_dir / "summary_primary.csv",
        "summary_exploratory": output_dir / "summary_exploratory.csv",
        "sensitivity_exposure": output_dir / "sensitivity_exposure.csv",
        "sensitivity_indication": output_dir / "sensitivity_indication.csv",
        "sensitivity_reporting_country_time": output_dir / "sensitivity_reporting_country_time.csv",
        "sensitivity_age_comedication": output_dir / "sensitivity_age_comedication.csv",
        "sensitivity_all_summary": output_dir / "sensitivity_all_summary.csv",
        "qc": output_dir / "qc.csv",
    }
    _write_csv(primary_df, outputs["signal_primary"])
    _write_csv(sensitivity_df, outputs["signal_sensitivity"])
    _write_csv(exploratory_df, outputs["signal_exploratory"])
    _write_csv(adjusted_primary, outputs["adjusted_primary"])
    _write_csv(adjusted_sensitivity, outputs["adjusted_sensitivity"])
    _write_csv(summary_primary, outputs["summary_primary"])
    _write_csv(summary_exploratory, outputs["summary_exploratory"])
    for name, table in sensitivity_tables.items():
        _write_csv(table, outputs[name])
    _write_csv(qc_df, outputs["qc"])

    assert not primary_df[
        primary_df["analysis"].eq("primary_ps_ss") & primary_df["outcome_name"].eq("strict_fall")
    ].empty
    for col in ["fdr_q_value", "stability_status", "is_stable_signal"]:
        assert col in exploratory_df.columns
    unstable_signal = exploratory_df[
        ((exploratory_df["a"] < 5) | (exploratory_df["exposed_n"] < 50))
        & exploratory_df["is_stable_signal"].fillna(False).astype(bool)
    ]
    assert unstable_signal.empty
    exposure_terms = adjusted_df[
        adjusted_df["is_exposure_term"].fillna(False)
        & adjusted_df["analysis"].eq("primary_ps_ss")
        & adjusted_df["outcome_name"].eq("strict_fall")
    ]
    assert {"core_clinical_adjusted", "extended_report_indication_adjusted"}.issubset(
        set(exposure_terms["model"])
    )
    assert not sensitivity_all_summary.empty
    assert not sensitivity_all_summary.astype(str).apply(
        lambda col: col.str.contains("is_fall_broad", na=False)
    ).any().any()
    assert "sensitivity_exposure_outcome" not in outputs
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FAERS analysis project v2.")
    parser.add_argument("--period-token", default=None, help="Dataset token such as 2004_2025.")
    parser.add_argument("--dataset-dir", default=GLOBAL_DATASET_DIR, type=Path)
    parser.add_argument("--output-dir", default=GLOBAL_OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    outputs = run(
        period_token=args.period_token,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    print("analysis_project_v2 completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
