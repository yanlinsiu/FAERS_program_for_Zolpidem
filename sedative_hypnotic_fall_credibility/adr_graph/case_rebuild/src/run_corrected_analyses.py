from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


REBUILD_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REBUILD_DIR.parents[1]
DATASET = REBUILD_DIR / "outputs" / "14_analysis_ready_dataset.parquet"
OUT = REBUILD_DIR / "outputs" / "analysis"


def load_module(filename: str):
    path = PROJECT_DIR / "src" / filename
    name = "corrected_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_signal_without_stale_count(module, results) -> None:
    present = results[results["column_present"] == True].copy()
    for check in ["check_total_matches", "check_fall_total_matches", "check_exposed_total_matches"]:
        if not present[check].all():
            raise ValueError(f"Corrected signal validation failed: {check}")
    metrics = ["ROR", "ROR_95CI_low", "ROR_95CI_high", "PRR", "PRR_95CI_low", "PRR_95CI_high", "IC", "IC025", "OE", "OE05"]
    if not np.isfinite(present[metrics].to_numpy(dtype=float)).all():
        raise ValueError("Corrected signal table contains non-finite metrics.")
    if present[present["target_key"].eq("zolpidem")].empty:
        raise ValueError("Corrected signal table lacks zolpidem.")


def validate_descriptive_without_stale_count(module, data, baseline, exposure) -> None:
    eligible = module.safe_bool(data["analysis_eligible_main"])
    total_n = int(eligible.sum())
    fall_n = int(module.safe_bool(data.loc[eligible, "strict_fall"]).sum())
    population = baseline[(baseline["variable"] == "Study population") & (baseline["category"] == "Yes")]
    falls = baseline[(baseline["variable"] == "Strict fall report") & (baseline["category"] == "Yes")]
    if population.empty or int(population.iloc[0]["overall_n"]) != total_n:
        raise ValueError("Corrected baseline population does not match eligible rows.")
    if falls.empty or int(falls.iloc[0]["overall_n"]) != fall_n:
        raise ValueError("Corrected baseline falls do not match eligible rows.")
    if exposure[exposure["target_key"].eq("zolpidem")].empty:
        raise ValueError("Corrected exposure table lacks zolpidem.")


def run(filename: str, arguments: list[str], patch: str | None = None) -> None:
    module = load_module(filename)
    if patch == "signal":
        module.validate_results = lambda results: validate_signal_without_stale_count(module, results)
    elif patch == "descriptive":
        module.validate_results = lambda data, baseline, exposure: validate_descriptive_without_stale_count(
            module, data, baseline, exposure
        )
    previous = sys.argv
    try:
        sys.argv = [filename, *arguments]
        print(f"\n=== {filename} ===", flush=True)
        module.main()
    finally:
        sys.argv = previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATASET)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument(
        "--start-at",
        choices=["06", "06a", "06b", "07", "08", "09", "10"],
        default="06",
    )
    args = parser.parse_args()
    stage_order = ["06", "06a", "06b", "07", "08", "09", "10"]
    start_index = stage_order.index(args.start_at)
    should_run = lambda stage: stage_order.index(stage) >= start_index
    output_root = args.output_root.resolve()
    tables = output_root / "tables"
    qc = output_root / "qc"
    figures = output_root / "figures"
    intermediate = output_root / "intermediate"
    for directory in [tables, qc, figures, intermediate]:
        directory.mkdir(parents=True, exist_ok=True)
    dataset = str(args.input.resolve())

    if should_run("06"):
        run(
        "06_signal_landscape.py",
        ["--main-dataset", dataset, "--drug-master", str(PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"),
         "--table-out", str(tables / "table_1_signal_landscape.csv"), "--qc-out", str(qc / "06_signal_landscape_qc.csv"),
         "--figure-out", str(figures / "figure_2_signal_landscape.png")],
        patch="signal",
        )
    if should_run("06a"):
        run(
        "06a_descriptive_analysis.py",
        ["--main-dataset", dataset, "--drug-master", str(PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"),
         "--baseline-out", str(tables / "table_s3_baseline_characteristics.csv"),
         "--exposure-out", str(tables / "table_s4_exposure_characteristics.csv"),
         "--qc-out", str(qc / "06a_descriptive_analysis_qc.csv"),
         "--flow-figure-out", str(figures / "figure_1_study_flow.png")],
        patch="descriptive",
        )
    if should_run("06b"):
        run(
        "06b_bcpnn_signal_sensitivity.py",
        ["--signal-table", str(tables / "table_1_signal_landscape.csv"),
         "--table-out", str(tables / "table_s5_bcpnn_signal_sensitivity.csv"),
         "--qc-out", str(qc / "06b_bcpnn_signal_sensitivity_qc.csv")],
        )
    if should_run("07"):
        run(
        "07_active_comparator.py",
        ["--main-dataset", dataset, "--table-out", str(tables / "table_2_active_comparator_results.csv"),
         "--qc-out", str(qc / "07_active_comparator_qc.csv"),
         "--figure-out", str(figures / "figure_s1_active_comparator_forest.png")],
        )
    if should_run("08"):
        run(
        "08_adjusted_ror.py",
        ["--main-dataset", dataset, "--table-out", str(tables / "table_3_adjusted_ror.csv"),
         "--qc-out", str(qc / "08_adjusted_model_qc.csv"),
         "--figure-out", str(figures / "figure_3_adjusted_ror_forest.png")],
        )
    if should_run("09"):
        run(
        "09_sensitivity_analyses.py",
        ["--main-dataset", dataset, "--ps-only-out", str(tables / "table_s6_ps_only_results.csv"),
         "--excluding-mixed-out", str(tables / "table_s7_excluding_mixed_results.csv"),
         "--reporting-source-out", str(tables / "table_s8_reporting_source_results.csv"),
         "--qc-out", str(qc / "09_sensitivity_analyses_qc.csv")],
        )
    if should_run("10"):
        run(
        "10_phenotype_fingerprint_analysis.py",
        ["--main-dataset", dataset, "--case-labels-out", str(intermediate / "10_phenotype_fingerprint_case_labels.parquet"),
         "--profile-out", str(tables / "table_4_phenotype_profiles.csv"),
         "--primary-out", str(tables / "table_s9_primary_phenotype.csv"),
         "--chisq-out", str(tables / "table_s10_phenotype_chisq.csv"),
         "--logit-out", str(tables / "table_s11_phenotype_logit.csv"),
         "--drug-profile-out", str(tables / "table_s12_drug_phenotype_profiles.csv"),
         "--drug-contrast-out", str(tables / "table_s13_drug_phenotype_contrasts.csv"),
         "--qc-out", str(qc / "10_phenotype_fingerprint_analysis_qc.csv"),
         "--figure-out", str(figures / "figure_4_phenotype_fingerprint.png")],
        )
    print("\nCorrected analyses complete.", flush=True)


if __name__ == "__main__":
    main()
