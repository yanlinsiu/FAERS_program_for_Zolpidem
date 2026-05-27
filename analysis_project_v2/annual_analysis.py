from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis_project_v2.config import FEATURE_SPECS, OUTCOME_SPECS
from analysis_project_v2.signal_metrics import feature_mask, signal_metrics, two_by_two_counts
from common.schema_checks import validate_feature_schema, validate_signal_schema


DEFAULT_SIGNAL_ROOT = Path(r"D:\program_FAERS\OUTPUT")
DEFAULT_ANALYSIS_ROOT = Path(r"D:\program_FAERS\OUTPUT\analysis")

STRATUM_SPECS = (
    ("age_group", "65-74", "Age 65-74"),
    ("age_group", "75-84", "Age 75-84"),
    ("age_group", ">=85", "Age >=85"),
    ("sex_clean", "F", "Female"),
    ("sex_clean", "M", "Male"),
    ("serious", True, "Serious outcome"),
    ("polypharmacy_5", True, "Polypharmacy >=5"),
)


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    path = Path(output_dir) if output_dir else DEFAULT_ANALYSIS_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_input_path(path_value: str | Path | None, default_path: Path) -> Path:
    return Path(path_value) if path_value else default_path


def _extract_period_from_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("signal_dataset_"):
        return stem.replace("signal_dataset_", "")
    if stem.startswith("drug_feature_") and stem.endswith("_case"):
        return stem.replace("drug_feature_", "").replace("_case", "")
    return stem


def _list_signal_files(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> list[Path]:
    if signal_file:
        return [Path(signal_file)]

    root = _normalize_input_path(signal_root, DEFAULT_SIGNAL_ROOT)
    if root.is_file():
        return [root]
    return sorted(root.glob("signal_dataset_*.parquet"))


def _list_feature_files(
    signal_root: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> list[Path]:
    if feature_file:
        return [Path(feature_file)]

    root = _normalize_input_path(signal_root, DEFAULT_SIGNAL_ROOT)
    if root.is_file():
        return [root]
    return sorted(root.glob("drug_feature_*_case.parquet"))


def load_signal_dataset(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> pd.DataFrame:
    files = _list_signal_files(signal_root=signal_root, signal_file=signal_file)
    if not files:
        raise FileNotFoundError("No signal_dataset_*.parquet files found.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_parquet(file_path).copy()
        df["dataset_period"] = _extract_period_from_name(file_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["caseid"] = combined["caseid"].astype(str).str.strip()
    combined = combined[combined["caseid"] != ""].copy()
    validate_signal_schema(combined)
    if "serious" in combined.columns:
        combined["serious"] = combined["serious"].fillna(False).astype(bool)
    return combined


def load_feature_dataset(
    signal_root: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> pd.DataFrame:
    files = _list_feature_files(signal_root=signal_root, feature_file=feature_file)
    if not files:
        raise FileNotFoundError("No drug_feature_*_case.parquet files found.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_parquet(file_path).copy()
        df["dataset_period"] = _extract_period_from_name(file_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["caseid"] = combined["caseid"].astype(str).str.strip()
    combined = combined[combined["caseid"] != ""].copy()
    validate_feature_schema(combined)
    return combined


def merge_signal_and_feature(
    signal_root: str | Path | None = None,
    signal_file: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> pd.DataFrame:
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)
    feature_df = load_feature_dataset(signal_root=signal_root, feature_file=feature_file)
    merged = signal_df.merge(feature_df, on=["caseid", "dataset_period"], how="left")

    bool_cols = [
        "is_zolpidem",
        "is_zolpidem_any",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "polypharmacy_5",
        "polypharmacy",
        "serious",
        "is_fall",
    ]
    for col in bool_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    for col in ["drug_n", "distinct_drug_n"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

    return merged


def summarize_missing(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            summary[f"missing_{col}"] = -1
            continue
        missing_mask = df[col].isna()
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            missing_mask = missing_mask | (df[col].astype(str).str.strip() == "")
        summary[f"missing_{col}"] = int(missing_mask.sum())
    return summary


def describe_signal(metrics: dict[str, object]) -> str:
    if (
        metrics.get("signal_flag_ror")
        or metrics.get("signal_flag_mhra")
        or metrics.get("signal_flag_ic")
        or metrics.get("signal_flag_ebgm")
    ):
        return "signal_detected"
    return "no_clear_signal"


def _outcome_records() -> list[dict[str, str]]:
    return [
        {
            "outcome_name": spec.name,
            "outcome_col": spec.column,
            "outcome_label": spec.label,
        }
        for spec in OUTCOME_SPECS
    ]


def _write_tables(
    result_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    result_path: Path,
    qc_path: Path,
) -> None:
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")


def _build_stratified_rows(
    df: pd.DataFrame,
    analysis_name: str,
    exposure_col: str,
    outcome_col: str,
    outcome_name: str,
    outcome_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, value, label in STRATUM_SPECS:
        if column not in df.columns:
            continue
        mask = feature_mask(df, column, value)
        subset = df[mask].copy()
        if subset.empty:
            continue
        metrics = signal_metrics(**two_by_two_counts(subset[exposure_col], subset[outcome_col]))
        rows.append(
            {
                "analysis": analysis_name,
                "outcome_name": outcome_name,
                "outcome_label": outcome_label,
                "stratum_col": column,
                "stratum_value": value,
                "stratum_label": label,
                "n_in_stratum": int(len(subset)),
                "n_outcome": int(subset[outcome_col].fillna(False).astype(bool).sum()),
                "conclusion": describe_signal(metrics),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _make_overall_qc(
    df: pd.DataFrame,
    label: str,
    exposure_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    exposed = df[exposure_col].fillna(False).astype(bool)
    outcome = df[outcome_col].fillna(False).astype(bool)
    qc = {
        "analysis": label,
        "outcome_col": outcome_col,
        "n_total": int(len(df)),
        "n_exposed": int(exposed.sum()),
        "n_unexposed": int((~exposed).sum()),
        "n_outcome": int(outcome.sum()),
        "n_exposed_outcome": int((exposed & outcome).sum()),
    }
    qc.update(summarize_missing(df, ["age_group", "sex_clean", "serious", "fall_pt_list"]))
    return pd.DataFrame([qc])


def build_signal_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)
    result_rows: list[dict[str, object]] = []
    qc_frames: list[pd.DataFrame] = []
    stratified_frames: list[pd.DataFrame] = []

    configs = (
        ("primary_ps_ss", "is_zolpidem_suspect", "suspect_role_any", "target_drug_group"),
        (
            "sensitivity_ps_only",
            "is_zolpidem_suspect_ps",
            "suspect_role_any_ps",
            "target_drug_group_ps",
        ),
    )
    for outcome_spec in _outcome_records():
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in signal_df.columns:
            continue

        for analysis_name, exposure_col, suspect_col, group_col in configs:
            subset = signal_df[signal_df[suspect_col].fillna(False).astype(bool)].copy()
            subset = subset[subset[group_col] != "both_zolpidem_and_other_zdrug"].copy()
            metrics = signal_metrics(**two_by_two_counts(subset[exposure_col], subset[outcome_col]))
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

            qc_df = _make_overall_qc(subset, analysis_name, exposure_col, outcome_col)
            qc_df["outcome_name"] = outcome_name
            qc_df["outcome_definition"] = outcome_label
            group_counts = (
                subset.groupby(group_col, dropna=False)
                .agg(n_cases=("caseid", "count"), n_outcome=(outcome_col, "sum"))
                .reset_index()
                .rename(columns={group_col: "drug_group"})
            )
            group_counts["analysis"] = analysis_name
            group_counts["outcome_name"] = outcome_name
            group_counts["outcome_definition"] = outcome_label
            qc_frames.extend([qc_df, group_counts])

            stratified_df = _build_stratified_rows(
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
    qc_result_df = pd.concat(qc_frames, ignore_index=True, sort=False) if qc_frames else pd.DataFrame()
    stratified_result_df = (
        pd.concat(stratified_frames, ignore_index=True, sort=False)
        if stratified_frames
        else pd.DataFrame()
    )

    output_root = ensure_output_dir(output_dir)
    _write_tables(
        result_df,
        qc_result_df,
        output_root / "01_signal_analysis_results.csv",
        output_root / "01_signal_analysis_qc.csv",
    )
    stratified_result_df.to_csv(
        output_root / "01_signal_analysis_stratified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result_df, qc_result_df


def _build_comparator_subset(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return df[df[group_col].isin({"zolpidem_only", "other_zdrug_only"})].copy()


def build_comparative_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)
    result_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    stratified_frames: list[pd.DataFrame] = []

    configs = (
        ("primary_ps_ss", "target_drug_group", "zolpidem_only"),
        ("sensitivity_ps_only", "target_drug_group_ps", "zolpidem_only"),
    )
    for outcome_spec in _outcome_records():
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in signal_df.columns:
            continue

        for analysis_name, group_col, zolpidem_value in configs:
            subset = _build_comparator_subset(signal_df, group_col)
            exposed = subset[group_col].eq(zolpidem_value)
            metrics = signal_metrics(**two_by_two_counts(exposed, subset[outcome_col]))
            result_rows.append(
                {
                    "analysis": analysis_name,
                    "comparison": "zolpidem_only_vs_other_zdrug_only",
                    "outcome_name": outcome_name,
                    "outcome_definition": outcome_label,
                    "conclusion": describe_signal(metrics),
                    **metrics,
                }
            )

            for drug_group, frame in subset.groupby(group_col, dropna=False):
                outcome = frame[outcome_col].fillna(False).astype(bool)
                qc_rows.append(
                    {
                        "analysis": analysis_name,
                        "outcome_name": outcome_name,
                        "drug_group": drug_group,
                        "n_cases": int(len(frame)),
                        "n_outcome": int(outcome.sum()),
                        "outcome_reporting_rate": float(outcome.mean()) if len(frame) else None,
                        "n_female": int(frame["sex_clean"].eq("F").sum())
                        if "sex_clean" in frame.columns
                        else None,
                        "n_age_75_plus": int(frame["age_group"].isin(["75-84", ">=85"]).sum())
                        if "age_group" in frame.columns
                        else None,
                    }
                )

            stratified_df = _build_stratified_rows(
                subset.assign(exposed_group=exposed),
                analysis_name=analysis_name,
                exposure_col="exposed_group",
                outcome_col=outcome_col,
                outcome_name=outcome_name,
                outcome_label=outcome_label,
            )
            if not stratified_df.empty:
                stratified_frames.append(stratified_df)

    result_df = pd.DataFrame(result_rows)
    qc_df = pd.DataFrame(qc_rows)
    output_root = ensure_output_dir(output_dir)
    _write_tables(
        result_df,
        qc_df,
        output_root / "02_comparative_analysis_results.csv",
        output_root / "02_comparative_analysis_qc.csv",
    )
    (
        pd.concat(stratified_frames, ignore_index=True, sort=False)
        if stratified_frames
        else pd.DataFrame()
    ).to_csv(
        output_root / "02_comparative_analysis_stratified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result_df, qc_df


def _build_feature_rows(
    df: pd.DataFrame,
    analysis_name: str,
    outcome_name: str,
    outcome_col: str,
    outcome_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURE_SPECS:
        if feature.column not in df.columns:
            continue
        mask = feature_mask(df, feature.column, feature.value)
        outcome = df[outcome_col].fillna(False).astype(bool)
        metrics = signal_metrics(**two_by_two_counts(mask, outcome))
        exposed_n = int(mask.sum())
        outcome_n = int((mask & outcome).sum())
        rows.append(
            {
                "analysis": analysis_name,
                "outcome_name": outcome_name,
                "outcome_definition": outcome_label,
                "feature_domain": feature.domain,
                "feature_name": f"{feature.column}={feature.value}",
                "feature_label": feature.label,
                "n_feature_positive": exposed_n,
                "n_feature_positive_outcome": outcome_n,
                "outcome_reporting_rate": (outcome_n / exposed_n) if exposed_n else None,
                "conclusion": describe_signal(metrics),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_feature_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_df = merge_signal_and_feature(
        signal_root=signal_root,
        signal_file=signal_file,
        feature_file=feature_file,
    )
    result_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    configs = (
        ("primary_ps_ss", "is_zolpidem_suspect"),
        ("sensitivity_ps_only", "is_zolpidem_suspect_ps"),
    )
    for outcome_spec in _outcome_records():
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in merged_df.columns:
            continue

        for analysis_name, exposure_col in configs:
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
                    "missing_age_group": int(subset["age_group"].isna().sum())
                    if "age_group" in subset.columns
                    else None,
                    "missing_sex_clean": int(subset["sex_clean"].isna().sum())
                    if "sex_clean" in subset.columns
                    else None,
                    "missing_serious": int(subset["serious"].isna().sum())
                    if "serious" in subset.columns
                    else None,
                    "n_polypharmacy_5": int(subset["polypharmacy_5"].fillna(False).astype(bool).sum())
                    if "polypharmacy_5" in subset.columns
                    else None,
                    "n_serious": int(subset["serious"].fillna(False).astype(bool).sum())
                    if "serious" in subset.columns
                    else None,
                    "n_benzo": int(subset["is_benzo"].fillna(False).astype(bool).sum())
                    if "is_benzo" in subset.columns
                    else None,
                    "n_antidepressant": int(subset["is_antidepressant"].fillna(False).astype(bool).sum())
                    if "is_antidepressant" in subset.columns
                    else None,
                    "n_antipsychotic": int(subset["is_antipsychotic"].fillna(False).astype(bool).sum())
                    if "is_antipsychotic" in subset.columns
                    else None,
                    "n_opioid": int(subset["is_opioid"].fillna(False).astype(bool).sum())
                    if "is_opioid" in subset.columns
                    else None,
                    "n_antiepileptic": int(subset["is_antiepileptic"].fillna(False).astype(bool).sum())
                    if "is_antiepileptic" in subset.columns
                    else None,
                }
            )

    result_df = (
        pd.concat(result_frames, ignore_index=True, sort=False)
        if result_frames
        else pd.DataFrame()
    )
    if not result_df.empty and "ror" in result_df.columns:
        result_df = result_df.sort_values(
            ["analysis", "outcome_name", "ror"],
            ascending=[True, True, False],
            na_position="last",
        )
    qc_df = pd.DataFrame(qc_rows)

    output_root = ensure_output_dir(output_dir)
    _write_tables(
        result_df,
        qc_df,
        output_root / "03_feature_analysis_results.csv",
        output_root / "03_feature_analysis_qc.csv",
    )
    return result_df, qc_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run legacy-compatible annual FAERS analyses.")
    parser.add_argument("--signal-root", default=DEFAULT_SIGNAL_ROOT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_ANALYSIS_ROOT, type=Path)
    parser.add_argument("--analysis", choices=["signal", "comparative", "feature", "all"], default="all")
    args = parser.parse_args()

    if args.analysis in {"signal", "all"}:
        build_signal_analysis(signal_root=args.signal_root, output_dir=args.output_dir)
    if args.analysis in {"comparative", "all"}:
        build_comparative_analysis(signal_root=args.signal_root, output_dir=args.output_dir)
    if args.analysis in {"feature", "all"}:
        build_feature_analysis(signal_root=args.signal_root, output_dir=args.output_dir)
    print("annual analysis completed.")


if __name__ == "__main__":
    main()

