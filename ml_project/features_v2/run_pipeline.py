from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


PIPELINE_SCRIPTS = [
    "00_field_audit.py",
    "01_meddra_lookup.py",
    "02_build_demo_features.py",
    "03_build_drug_role_features.py",
    "04_build_indi_features.py",
    "05_build_rpsr_features.py",
    "06_build_ther_features.py",
    "07_build_ml_feature_v2.py",
]


def _load_script(script_name: str):
    script_path = Path(__file__).resolve().parent / script_name
    module_name = script_path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ML-v2 script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_feature_v2_pipeline(start_year: int, end_year: int) -> None:
    for script_name in PIPELINE_SCRIPTS:
        module = _load_script(script_name)
        if not hasattr(module, "main"):
            raise AttributeError(f"{script_name} does not expose main()")
        print(f"[feature-v2] running {script_name}", flush=True)
        script_args = [script_name]
        if script_name != "01_meddra_lookup.py":
            script_args.extend(["--start-year", str(start_year), "--end-year", str(end_year)])
        old_argv = sys.argv
        try:
            sys.argv = script_args
            module.main()
        finally:
            sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ML-v2 feature-building pipeline.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--cleaned-output-root", type=Path, default=None)
    parser.add_argument("--global-dataset-dir", type=Path, default=None)
    parser.add_argument("--ml-output-root", type=Path, default=None)
    args = parser.parse_args()

    if args.cleaned_output_root is not None:
        os.environ["FAERS_CLEAN_OUTPUT_ROOT"] = str(args.cleaned_output_root.resolve())
    if args.global_dataset_dir is not None:
        os.environ["FAERS_GLOBAL_DATASET_DIR"] = str(args.global_dataset_dir.resolve())
    if args.ml_output_root is not None:
        os.environ["FAERS_ML_OUTPUT_ROOT"] = str(args.ml_output_root.resolve())

    run_feature_v2_pipeline(start_year=args.start_year, end_year=args.end_year)


if __name__ == "__main__":
    main()
