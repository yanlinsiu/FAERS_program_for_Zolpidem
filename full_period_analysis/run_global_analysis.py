from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from config import GLOBAL_OUTPUT_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_project"

if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))


def _load_function(module_file: str, function_name: str):
    module_path = ANALYSIS_ROOT / module_file
    spec = importlib.util.spec_from_file_location(function_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


build_signal_analysis = _load_function("01_signal_analysis.py", "build_signal_analysis")
build_comparative_analysis = _load_function("02_comparative_analysis.py", "build_comparative_analysis")
build_feature_analysis = _load_function("03_feature_analysis.py", "build_feature_analysis")


def _extract_token(path: Path, prefix: str, suffix: str = "") -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected dataset file name: {path.name}")

    token = stem[len(prefix) :]
    if suffix:
        if not token.endswith(suffix):
            raise ValueError(f"Unexpected dataset file name: {path.name}")
        token = token[: -len(suffix)]
    return token


def _resolve_dataset_bundle(dataset_dir: Path, period_token: str | None = None) -> tuple[Path, Path, str]:
    signal_files = sorted(dataset_dir.glob("signal_dataset_*.parquet"))
    feature_files = sorted(dataset_dir.glob("drug_feature_*_case.parquet"))

    if not signal_files:
        raise FileNotFoundError(f"No signal_dataset_*.parquet files found in {dataset_dir}")
    if not feature_files:
        raise FileNotFoundError(f"No drug_feature_*_case.parquet files found in {dataset_dir}")

    signal_by_token = {
        _extract_token(path, "signal_dataset_"): path
        for path in signal_files
    }
    feature_by_token = {
        _extract_token(path, "drug_feature_", "_case"): path
        for path in feature_files
    }

    shared_tokens = sorted(set(signal_by_token) & set(feature_by_token))
    if not shared_tokens:
        raise RuntimeError(
            "No matching global signal/feature dataset pair found in "
            f"{dataset_dir}"
        )

    if period_token:
        if period_token not in signal_by_token or period_token not in feature_by_token:
            raise FileNotFoundError(
                f"Requested period token not found in {dataset_dir}: {period_token}"
            )
        selected_token = period_token
    elif len(shared_tokens) == 1:
        selected_token = shared_tokens[0]
    else:
        joined_tokens = ", ".join(shared_tokens)
        raise RuntimeError(
            "Multiple global dataset versions found. "
            "Please rerun with --period-token to select one: "
            f"{joined_tokens}"
        )

    return signal_by_token[selected_token], feature_by_token[selected_token], selected_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Run analysis on one explicit global dataset bundle.")
    parser.add_argument(
        "--period-token",
        default=None,
        help="Dataset token such as 2024_2024 or 2004_2025. Required if multiple bundles exist.",
    )
    args = parser.parse_args()

    dataset_dir = GLOBAL_OUTPUT_ROOT / "datasets"
    output_dir = GLOBAL_OUTPUT_ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_file, feature_file, selected_token = _resolve_dataset_bundle(
        dataset_dir, period_token=args.period_token
    )

    build_signal_analysis(output_dir=output_dir, signal_file=signal_file)
    build_comparative_analysis(output_dir=output_dir, signal_file=signal_file)
    build_feature_analysis(output_dir=output_dir, signal_file=signal_file, feature_file=feature_file)

    print(f"Global analysis completed for period: {selected_token}")
    print(f"Signal dataset: {signal_file}")
    print(f"Feature dataset: {feature_file}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
