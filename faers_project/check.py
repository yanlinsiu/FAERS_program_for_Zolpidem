from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_OUTPUT_ROOT_PATH


def _default_case_file() -> Path:
    candidates = sorted(
        DEFAULT_OUTPUT_ROOT_PATH.glob("**/case_dataset_*.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No case_dataset_*.parquet file found under {DEFAULT_OUTPUT_ROOT_PATH}"
        )
    return candidates[0]


def _bool_count(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int(df[column].fillna(False).astype(bool).sum())


def inspect_case_dataset(case_file: Path, head: int = 5) -> dict[str, object]:
    df = pd.read_parquet(case_file)
    metrics = {
        "file": case_file,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "head": df.head(int(head)),
        "n_fall": _bool_count(df, "is_fall"),
        "n_zolpidem_any": _bool_count(df, "is_zolpidem"),
        "n_zolpidem_suspect": _bool_count(df, "is_zolpidem_suspect"),
        "n_polypharmacy": _bool_count(df, "polypharmacy"),
    }
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quickly inspect a FAERS case dataset parquet file.")
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
        help="Path to case_dataset parquet. Defaults to the newest one under OUTPUT.",
    )
    parser.add_argument("--head", type=int, default=5, help="Number of preview rows to print.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    case_file = args.case_file or _default_case_file()
    metrics = inspect_case_dataset(case_file=case_file, head=args.head)

    print(f"File: {metrics['file']}")
    print(f"Rows: {metrics['rows']}")
    print("Columns:")
    print(metrics["columns"])
    print("")
    print("Preview:")
    print(metrics["head"])
    print("")
    print("Key counts:")
    for key in [
        "n_fall",
        "n_zolpidem_any",
        "n_zolpidem_suspect",
        "n_polypharmacy",
    ]:
        print(f"- {key}: {metrics[key]}")


if __name__ == "__main__":
    main()

