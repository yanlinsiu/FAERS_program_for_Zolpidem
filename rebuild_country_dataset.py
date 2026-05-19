from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT_COUNTRY"
DEFAULT_GLOBAL_OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT_GLOBAL_COUNTRY"


def _project_python() -> str:
    windows_venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if windows_venv_python.exists():
        return str(windows_venv_python)

    posix_venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if posix_venv_python.exists():
        return str(posix_venv_python)

    return sys.executable


def _run_command(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _write_readme(
    output_root: Path,
    global_output_root: Path,
    start_year: int,
    end_year: int,
    force: bool,
) -> Path:
    readme_path = output_root / "README_country_rebuild.md"
    output_root.mkdir(parents=True, exist_ok=True)
    global_output_root.mkdir(parents=True, exist_ok=True)

    lines = [
        "# FAERS country-field rebuild",
        "",
        f"- Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Year range: {start_year}-{end_year}",
        f"- Cleaned output root: `{output_root}`",
        f"- Global output root: `{global_output_root}`",
        f"- Force rebuild: `{force}`",
        "",
        "## Purpose",
        "",
        "This rebuild keeps DEMO country fields in the cleaned case-level datasets.",
        "",
        "Newly retained fields:",
        "",
        "- `reporter_country`: country of report source",
        "- `occr_country`: country where the adverse event occurred",
        "",
        "The original `OUTPUT` and `OUTPUT_GLOBAL` directories are not used as write targets by this script.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild FAERS cleaned datasets into separate country-preserving output roots."
    )
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Separate cleaned output root. Defaults to OUTPUT_COUNTRY.",
    )
    parser.add_argument(
        "--global-output-root",
        type=Path,
        default=DEFAULT_GLOBAL_OUTPUT_ROOT,
        help="Separate global output root. Defaults to OUTPUT_GLOBAL_COUNTRY.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild yearly outputs even if this separate output root already has results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-step processor output.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Only rebuild yearly cleaned outputs; do not build global 2004-2025 datasets.",
    )
    args = parser.parse_args()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    output_root = args.output_root.resolve()
    global_output_root = args.global_output_root.resolve()

    readme_path = _write_readme(
        output_root=output_root,
        global_output_root=global_output_root,
        start_year=start_year,
        end_year=end_year,
        force=bool(args.force),
    )

    python_executable = _project_python()

    yearly_command = [
        python_executable,
        str(PROJECT_ROOT / "faers_project" / "year_batch_runner.py"),
        "--start-year",
        str(start_year),
        "--end-year",
        str(end_year),
        "--output-root",
        str(output_root),
    ]
    if args.force:
        yearly_command.append("--force")
    if args.verbose:
        yearly_command.append("--verbose")
    _run_command(yearly_command, cwd=PROJECT_ROOT)

    if not args.skip_global:
        global_command = [
            python_executable,
            str(PROJECT_ROOT / "full_period_analysis" / "build_global_datasets.py"),
            "--start-year",
            str(start_year),
            "--end-year",
            str(end_year),
            "--input-output-root",
            str(output_root),
            "--global-output-root",
            str(global_output_root),
        ]
        _run_command(global_command, cwd=PROJECT_ROOT)

    print("Country-field rebuild finished.")
    print(f"Version note: {readme_path}")
    print(f"Cleaned output root: {output_root}")
    print(f"Global output root: {global_output_root}")


if __name__ == "__main__":
    main()
