from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import resolve_signal_feature_bundle
from common.schema_checks import audit_core_analysis_tables, write_audit_report


DEFAULT_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
DEFAULT_QC_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "qc"


def run_audit(
    period_token: str | None = "2004_2025",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_dir: Path = DEFAULT_QC_DIR,
) -> tuple[pd.DataFrame, Path]:
    bundle = resolve_signal_feature_bundle(dataset_dir=dataset_dir, period_token=period_token)
    case_index_file = Path(dataset_dir) / f"global_case_index_{bundle.period_token}.parquet"
    if not case_index_file.exists():
        raise FileNotFoundError(f"Case index dataset not found: {case_index_file}")

    case_index = pd.read_parquet(case_index_file)
    signal = pd.read_parquet(bundle.signal_file)
    feature = pd.read_parquet(bundle.feature_file)

    report = audit_core_analysis_tables(case_index=case_index, signal=signal, feature=feature)
    output_file = output_dir / f"field_audit_{bundle.period_token}.csv"
    output_path = write_audit_report(report, output_file)
    return report, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official FAERS analysis fields.")
    parser.add_argument("--period-token", default="2004_2025", help="Dataset token, for example 2004_2025.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_QC_DIR, type=Path)
    args = parser.parse_args()

    report, output_path = run_audit(
        period_token=args.period_token,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    failed = report[report["status"].eq("fail")]
    print(f"Field audit report: {output_path}")
    print(f"Checks: {len(report)}, failed: {len(failed)}")
    if not failed.empty:
        print(failed.to_string(index=False))
        raise SystemExit(1)
    print("Field audit passed.")


if __name__ == "__main__":
    main()
