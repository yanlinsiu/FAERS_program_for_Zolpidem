from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output20260601"
DEFAULT_RESULT_DIR = DEFAULT_OUTPUT_ROOT / "20260603结果" / "mcc_final"


@dataclass(frozen=True)
class SourceSpec:
    model: str
    source_group: str
    pattern: str


SOURCE_SPECS = [
    SourceSpec(
        model="logistic_regression",
        source_group="output20260601_machine_learning_existing",
        pattern=(
            "machine_learning/logistic_regression_all_post20260525_runs/"
            "is_fall_all_2004_2025_v2_*"
        ),
    ),
    SourceSpec(
        model="random_forest",
        source_group="output20260601_machine_learning_existing",
        pattern=(
            "machine_learning/random_forest_all_post20260525_runs/"
            "is_fall_all_2004_2025_v2_*"
        ),
    ),
    SourceSpec(
        model="xgboost",
        source_group="xgboost_20260603_corrected",
        pattern="20260603结果/xgboost_full_run_remote_outputs/is_fall_all_2004_2025_v2_*20260603_*",
    ),
]


def calc_mcc(tn: int, fp: int, fn: int, tp: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom <= 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / math.sqrt(denom)


def infer_feature_set(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "_enhanced_" in name:
        return "enhanced"
    if "_core_" in name:
        return "core"
    return ""


def read_metrics_json(run_dir: Path) -> dict[str, Any]:
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return {}
    return json.loads(metrics_file.read_text(encoding="utf-8"))


def count_prediction_file(prediction_file: Path) -> dict[str, int | float]:
    tn = fp = fn = tp = n_rows = 0
    with prediction_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"target", "predicted_label_optimal"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{prediction_file} is missing columns: {sorted(missing)}")

        for row in reader:
            y_true = int(float(row["target"]))
            y_pred = int(float(row["predicted_label_optimal"]))
            n_rows += 1
            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            elif y_true == 1 and y_pred == 0:
                fn += 1

    return {
        "n_rows": n_rows,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mcc": calc_mcc(tn, fp, fn, tp),
    }


def collect_rows(output_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_run_dirs: set[Path] = set()

    for spec in SOURCE_SPECS:
        for run_dir in sorted(output_root.glob(spec.pattern)):
            if not run_dir.is_dir() or run_dir in seen_run_dirs:
                continue
            seen_run_dirs.add(run_dir)
            metrics = read_metrics_json(run_dir)

            for split in ("validation", "test"):
                prediction_file = run_dir / f"{split}_predictions.csv"
                if not prediction_file.exists():
                    continue
                metric_block = metrics.get(f"{split}_metrics", {})
                rows.append(
                    {
                        "model": metrics.get("model", spec.model),
                        "display_name": metrics.get("display_name", spec.model),
                        "feature_version": metrics.get("feature_version", "v2"),
                        "feature_set": metrics.get("feature_set") or infer_feature_set(run_dir),
                        "target_col": metrics.get("target_col", "is_fall"),
                        "cohort": metrics.get("cohort", "all"),
                        "period_token": metrics.get("period_token", "2004_2025"),
                        "source_group": spec.source_group,
                        "split": split,
                        "threshold": metric_block.get("threshold"),
                        **count_prediction_file(prediction_file),
                        "prediction_file": str(prediction_file),
                        "run_dir": str(run_dir),
                    }
                )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "model",
        "display_name",
        "feature_version",
        "feature_set",
        "target_col",
        "cohort",
        "period_token",
        "source_group",
        "split",
        "threshold",
        "n_rows",
        "tn",
        "fp",
        "fn",
        "tp",
        "mcc",
        "prediction_file",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary_file: Path, rows: list[dict[str, object]]) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    test_rows = sorted(test_rows, key=lambda row: float(row["mcc"]), reverse=True)
    lines = [
        "# Final MCC summary",
        "",
        "This folder contains the final MCC post-processing result.",
        "",
        "Source rule:",
        "- Logistic Regression: existing runs under `output20260601/machine_learning`.",
        "- Random Forest: existing runs under `output20260601/machine_learning`.",
        "- XGBoost: only corrected 2026-06-03 runs under `output20260601/20260603结果/xgboost_full_run_remote_outputs`.",
        "",
        "MCC inputs:",
        "- `target` from each `validation_predictions.csv` / `test_predictions.csv`.",
        "- `predicted_label_optimal` from each `validation_predictions.csv` / `test_predictions.csv`.",
        "",
        f"Result file: `{summary_file.name}`",
        "",
        "Test-set MCC ranking:",
    ]
    for row in test_rows:
        lines.append(
            "- "
            f"{row['model']} {row['feature_set']}: "
            f"MCC={float(row['mcc']):.6f}, "
            f"TP={row['tp']}, TN={row['tn']}, FP={row['fp']}, FN={row['fn']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate final MCC summary from existing ML prediction files."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args.output_root)
    if not rows:
        raise RuntimeError(f"No prediction files found under {args.output_root}")

    summary_file = result_dir / "final_mcc_summary.csv"
    readme_file = result_dir / "README.md"
    write_csv(summary_file, rows)
    write_readme(readme_file, summary_file, rows)
    print(f"Wrote {len(rows)} rows to {summary_file}")
    print(f"Wrote notes to {readme_file}")


if __name__ == "__main__":
    main()
