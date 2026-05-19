from __future__ import annotations

import argparse

from config import DEFAULT_OUTPUT_ROOT
from pipeline import TABLE_CHOICES, normalize_quarter, normalize_table, run_quarter_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one FAERS quarterly processing step, or the full quarterly pipeline."
    )
    parser.add_argument("--year", required=True, type=int, help="Year, for example 2024")
    parser.add_argument(
        "--quarter",
        required=True,
        choices=("Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"),
        help="Quarter, for example Q1",
    )
    parser.add_argument(
        "--table",
        required=True,
        choices=TABLE_CHOICES,
        help="Step to run. Use 'all' for the whole quarterly pipeline.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        help="Output directory. Defaults to the project OUTPUT directory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    quarter = normalize_quarter(args.quarter)
    table = normalize_table(args.table)

    completed = run_quarter_step(
        year=args.year,
        quarter=quarter,
        table=table,
        output_root=args.output,
    )

    print(f"Completed {args.year} {quarter}: {', '.join(completed)}")
    print(f"Output root: {args.output}")


if __name__ == "__main__":
    main()
