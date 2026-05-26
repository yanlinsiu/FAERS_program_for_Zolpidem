from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import GLOBAL_DATASET_DIR, GLOBAL_OUTPUT_DIR
    from .regulatory_trend_analysis import DEFAULT_OUTPUT_DIR as DEFAULT_TREND_OUTPUT_DIR
    from .regulatory_trend_analysis import run as run_regulatory_trend
    from .run_analysis import run as run_main_analysis
except ImportError:
    from config import GLOBAL_DATASET_DIR, GLOBAL_OUTPUT_DIR
    from regulatory_trend_analysis import DEFAULT_OUTPUT_DIR as DEFAULT_TREND_OUTPUT_DIR
    from regulatory_trend_analysis import run as run_regulatory_trend
    from run_analysis import run as run_main_analysis


def run(
    period_token: str | None,
    dataset_dir: Path,
    output_dir: Path,
    trend_output_dir: Path,
    skip_main: bool,
    skip_trend: bool,
) -> dict[str, dict[str, Path]]:
    outputs: dict[str, dict[str, Path]] = {}
    if not skip_main:
        outputs["main_analysis"] = run_main_analysis(
            period_token=period_token,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
    if not skip_trend:
        outputs["regulatory_trend"] = run_regulatory_trend(
            period_token=period_token,
            dataset_dir=dataset_dir,
            output_dir=trend_output_dir,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the main FAERS analysis and the linked regulatory trend analysis."
    )
    parser.add_argument("--period-token", default="2004_2025")
    parser.add_argument("--dataset-dir", default=GLOBAL_DATASET_DIR, type=Path)
    parser.add_argument("--output-dir", default=GLOBAL_OUTPUT_DIR, type=Path)
    parser.add_argument("--trend-output-dir", default=DEFAULT_TREND_OUTPUT_DIR, type=Path)
    parser.add_argument("--skip-main", action="store_true")
    parser.add_argument("--skip-trend", action="store_true")
    args = parser.parse_args()

    outputs = run(
        period_token=args.period_token,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        trend_output_dir=args.trend_output_dir,
        skip_main=args.skip_main,
        skip_trend=args.skip_trend,
    )
    print("analysis_project_v2 run_all completed.")
    for section, section_outputs in outputs.items():
        print(f"[{section}]")
        for name, path in section_outputs.items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
