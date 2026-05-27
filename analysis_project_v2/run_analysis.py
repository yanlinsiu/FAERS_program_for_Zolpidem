from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.schema_checks import audit_core_analysis_tables, write_audit_report

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


def _outputs(output_dir: Path, period_token: str) -> dict[str, Path]:
    return {
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
        "qc_signal": output_dir / "qc_signal.csv",
        "qc_exploratory": output_dir / "qc_exploratory.csv",
        "qc_adjusted": output_dir / "qc_adjusted.csv",
        "qc_sensitivity": output_dir / "qc_sensitivity.csv",
        "qc": output_dir / "qc.csv",
        "field_audit": output_dir.parent / "qc" / f"field_audit_{period_token}.csv",
    }


@contextmanager
def _timed_stage(name: str):
    start = time.time()
    print(f"[{name}] started")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"[{name}] finished in {elapsed:.1f}s")


def _check_csv(path: Path, required_cols: tuple[str, ...] = ()) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected output was not written: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Expected output is empty: {path}")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Output {path} missing required columns: {missing}")


def _read_csv_checked(path: Path, required_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    _check_csv(path, required_cols)
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_frame(bundle) -> pd.DataFrame:
    print(f"Loading signal dataset: {bundle.signal_file}")
    print(f"Loading feature dataset: {bundle.feature_file}")
    df = load_analysis_frame(bundle)
    print(f"Loaded analysis frame: rows={len(df)}, columns={len(df.columns)}")
    return df


def _run_field_audit(bundle, dataset_dir: Path, output_dir: Path) -> Path:
    case_index_file = Path(dataset_dir) / f"global_case_index_{bundle.period_token}.parquet"
    if not case_index_file.exists():
        raise FileNotFoundError(f"Case index dataset not found: {case_index_file}")

    print(f"Auditing case index: {case_index_file}")
    case_index = pd.read_parquet(case_index_file)
    signal = pd.read_parquet(bundle.signal_file)
    feature = pd.read_parquet(bundle.feature_file)
    report = audit_core_analysis_tables(case_index=case_index, signal=signal, feature=feature)
    output_path = write_audit_report(report, _outputs(output_dir, bundle.period_token)["field_audit"])
    failed = report[report["status"].eq("fail")]
    print(f"Field audit report: {output_path}; checks={len(report)}, failed={len(failed)}")
    if not failed.empty:
        raise ValueError(f"Field audit failed. See: {output_path}")
    return output_path


def _run_signal_stage(df: pd.DataFrame, output_dir: Path, outputs: dict[str, Path]) -> dict[str, Path]:
    primary_df, sensitivity_df, _all_signal_df, signal_qc = build_signal_tables(df)
    exploratory_df, exploratory_qc = build_exploratory_table(df)

    _write_csv(primary_df, outputs["signal_primary"])
    _write_csv(sensitivity_df, outputs["signal_sensitivity"])
    _write_csv(exploratory_df, outputs["signal_exploratory"])
    _write_csv(signal_qc, outputs["qc_signal"])
    _write_csv(exploratory_qc, outputs["qc_exploratory"])

    _check_csv(outputs["signal_primary"], ("analysis", "outcome_name", "ror", "conclusion"))
    _check_csv(outputs["signal_sensitivity"], ("analysis", "outcome_name", "ror", "conclusion"))
    _check_csv(outputs["signal_exploratory"], ("analysis", "outcome_name", "fdr_q_value", "is_stable_signal"))
    for col in ["fdr_q_value", "stability_status", "is_stable_signal"]:
        if col not in exploratory_df.columns:
            raise ValueError(f"signal_exploratory missing required column: {col}")
    unstable_signal = exploratory_df[
        ((exploratory_df["a"] < 5) | (exploratory_df["exposed_n"] < 50))
        & exploratory_df["is_stable_signal"].fillna(False).astype(bool)
    ]
    if not unstable_signal.empty:
        raise ValueError("Exploratory output contains unstable rows marked as stable signals.")

    return {
        "signal_primary": outputs["signal_primary"],
        "signal_sensitivity": outputs["signal_sensitivity"],
        "signal_exploratory": outputs["signal_exploratory"],
        "qc_signal": outputs["qc_signal"],
        "qc_exploratory": outputs["qc_exploratory"],
    }


def _run_adjusted_stage(df: pd.DataFrame, outputs: dict[str, Path]) -> dict[str, Path]:
    adjusted_df, adjusted_qc = build_adjusted_analysis(df)

    adjusted_primary = adjusted_df[adjusted_df["analysis_tier"].eq("primary")].copy()
    adjusted_sensitivity = adjusted_df[adjusted_df["analysis_tier"].eq("sensitivity")].copy()
    _write_csv(adjusted_primary, outputs["adjusted_primary"])
    _write_csv(adjusted_sensitivity, outputs["adjusted_sensitivity"])
    _write_csv(adjusted_qc, outputs["qc_adjusted"])

    _check_csv(outputs["adjusted_primary"], ("analysis", "outcome_name", "model", "term", "is_exposure_term"))
    _check_csv(outputs["adjusted_sensitivity"], ("analysis", "outcome_name", "model", "term", "is_exposure_term"))
    exposure_terms = adjusted_df[
        adjusted_df["is_exposure_term"].fillna(False)
        & adjusted_df["analysis"].eq("primary_ps_ss")
        & adjusted_df["outcome_name"].eq("strict_fall")
    ]
    if not {"core_clinical_adjusted", "extended_report_indication_adjusted"}.issubset(
        set(exposure_terms["model"])
    ):
        raise ValueError("Adjusted primary output is missing required exposure model rows.")
    ps_exposure_terms = adjusted_df[
        adjusted_df["is_exposure_term"].fillna(False)
        & adjusted_df["analysis"].eq("sensitivity_ps_only")
        & adjusted_df["outcome_name"].eq("strict_fall")
    ]
    if ps_exposure_terms.empty:
        raise ValueError("Adjusted sensitivity output is missing PS-only exposure rows.")

    return {
        "adjusted_primary": outputs["adjusted_primary"],
        "adjusted_sensitivity": outputs["adjusted_sensitivity"],
        "qc_adjusted": outputs["qc_adjusted"],
    }


def _run_sensitivity_stage(df: pd.DataFrame, outputs: dict[str, Path]) -> dict[str, Path]:
    sensitivity_tables, sensitivity_all_summary, sensitivity_qc = build_sensitivity_adjusted_analysis(df)

    for name, table in sensitivity_tables.items():
        _write_csv(table, outputs[name])
    _write_csv(sensitivity_qc, outputs["qc_sensitivity"])

    _check_csv(outputs["sensitivity_all_summary"], ("section", "analysis", "status", "skip_reason"))
    skipped = sensitivity_all_summary[
        sensitivity_all_summary["status"].astype(str).eq("skipped")
    ].copy()
    missing_skip_reason = skipped["skip_reason"].isna() | skipped["skip_reason"].astype(str).str.strip().eq("")
    if bool(missing_skip_reason.any()):
        raise ValueError("Sensitivity output has skipped rows without skip_reason.")

    return {
        "sensitivity_exposure": outputs["sensitivity_exposure"],
        "sensitivity_indication": outputs["sensitivity_indication"],
        "sensitivity_reporting_country_time": outputs["sensitivity_reporting_country_time"],
        "sensitivity_age_comedication": outputs["sensitivity_age_comedication"],
        "sensitivity_all_summary": outputs["sensitivity_all_summary"],
        "qc_sensitivity": outputs["qc_sensitivity"],
    }


def _dataset_metadata(bundle) -> dict[str, object]:
    parquet_file = pq.ParquetFile(bundle.signal_file)
    return {
        "section": "dataset",
        "period_token": bundle.period_token,
        "signal_file": str(bundle.signal_file),
        "feature_file": str(bundle.feature_file),
        "n_rows": int(parquet_file.metadata.num_rows),
        "n_columns": int(len(parquet_file.schema_arrow.names)),
    }


def _run_summaries_stage(bundle, outputs: dict[str, Path]) -> dict[str, Path]:
    primary_df = _read_csv_checked(outputs["signal_primary"], ("analysis", "outcome_name"))
    sensitivity_df = _read_csv_checked(outputs["signal_sensitivity"], ("analysis", "outcome_name"))
    exploratory_df = _read_csv_checked(outputs["signal_exploratory"], ("analysis", "outcome_name"))
    adjusted_primary = _read_csv_checked(outputs["adjusted_primary"], ("analysis", "outcome_name"))
    adjusted_sensitivity = _read_csv_checked(outputs["adjusted_sensitivity"], ("analysis", "outcome_name"))
    adjusted_df = pd.concat([adjusted_primary, adjusted_sensitivity], ignore_index=True, sort=False)

    summary_primary = build_primary_summary(pd.concat([primary_df, sensitivity_df], ignore_index=True), adjusted_df)
    summary_exploratory = build_exploratory_summary(exploratory_df)
    _write_csv(summary_primary, outputs["summary_primary"])
    _write_csv(summary_exploratory, outputs["summary_exploratory"])

    qc_frames = [
        _read_csv_checked(outputs["qc_signal"]),
        _read_csv_checked(outputs["qc_exploratory"]),
        _read_csv_checked(outputs["qc_adjusted"]),
        _read_csv_checked(outputs["qc_sensitivity"]),
        pd.DataFrame([_dataset_metadata(bundle)]),
    ]
    qc_df = pd.concat(
        qc_frames,
        ignore_index=True,
        sort=False,
    )
    _write_csv(qc_df, outputs["qc"])

    _check_csv(outputs["summary_primary"], ("analysis", "outcome_name"))
    _check_csv(outputs["summary_exploratory"], ("analysis", "outcome_name"))
    _check_csv(outputs["qc"], ("section",))
    if qc_df.astype(str).apply(lambda col: col.str.contains("is_fall_", na=False)).any().any():
        raise ValueError("Final QC contains obsolete fall variant column naming.")

    return {
        "summary_primary": outputs["summary_primary"],
        "summary_exploratory": outputs["summary_exploratory"],
        "qc": outputs["qc"],
    }


def run(
    period_token: str | None,
    dataset_dir: Path,
    output_dir: Path,
    stage: str = "all",
) -> dict[str, Path]:
    bundle = resolve_dataset_bundle(dataset_dir=dataset_dir, period_token=period_token)
    outputs = _outputs(output_dir, bundle.period_token)
    completed: dict[str, Path] = {}

    if stage == "all":
        with _timed_stage("field_audit"):
            completed["field_audit"] = _run_field_audit(bundle, dataset_dir, output_dir)
        df = _load_frame(bundle)
        with _timed_stage("signal"):
            completed.update(_run_signal_stage(df, output_dir, outputs))
        with _timed_stage("adjusted"):
            completed.update(_run_adjusted_stage(df, outputs))
        with _timed_stage("sensitivity"):
            completed.update(_run_sensitivity_stage(df, outputs))
        with _timed_stage("summaries"):
            completed.update(_run_summaries_stage(bundle, outputs))
        return completed

    if stage in {"signal", "adjusted", "sensitivity"}:
        df = _load_frame(bundle)
        with _timed_stage(stage):
            if stage == "signal":
                completed.update(_run_signal_stage(df, output_dir, outputs))
            elif stage == "adjusted":
                completed.update(_run_adjusted_stage(df, outputs))
            else:
                completed.update(_run_sensitivity_stage(df, outputs))
        return completed

    if stage == "summaries":
        with _timed_stage("summaries"):
            completed.update(_run_summaries_stage(bundle, outputs))
        return completed

    raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FAERS analysis project v2.")
    parser.add_argument("--period-token", default=None, help="Dataset token such as 2004_2025.")
    parser.add_argument("--dataset-dir", default=GLOBAL_DATASET_DIR, type=Path)
    parser.add_argument("--output-dir", default=GLOBAL_OUTPUT_DIR, type=Path)
    parser.add_argument(
        "--stage",
        default="all",
        choices=("signal", "adjusted", "sensitivity", "summaries", "all"),
        help="Run only one analysis stage, or all stages in order.",
    )
    args = parser.parse_args()

    outputs = run(
        period_token=args.period_token,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        stage=args.stage,
    )
    print(f"analysis_project_v2 completed: stage={args.stage}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
