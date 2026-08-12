from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ADR_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ADR_DIR.parent
DEFAULT_CONFIG = ADR_DIR / "configs" / "pilot.json"
DEFAULT_OUTPUT_DIR = ADR_DIR / "outputs"


@dataclass(frozen=True)
class RorResult:
    ror: float
    lower: float
    upper: float


def ror_stats(a: int, b: int, c: int, d: int) -> RorResult:
    """Calculate ROR and Wald 95% CI, using 0.5 correction only when needed."""
    cells = np.asarray([a, b, c, d], dtype=float)
    if np.any(cells == 0):
        cells += 0.5
    aa, bb, cc, dd = cells
    ror = (aa * dd) / (bb * cc)
    se = math.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    log_ror = math.log(ror)
    return RorResult(
        ror=float(ror),
        lower=float(math.exp(log_ror - 1.96 * se)),
        upper=float(math.exp(log_ror + 1.96 * se)),
    )


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ADR_DIR / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def exposure_column(drug_key: str, suffix: str) -> str:
    return f"exposure_{drug_key}_{suffix}"


def count_raw_quarters(root: Path, first_year: int, last_year: int) -> tuple[int, list[str]]:
    found: list[str] = []
    for year in range(first_year, last_year + 1):
        year_dir = root / str(year)
        if not year_dir.exists():
            continue
        for quarter in range(1, 5):
            quarter_dir = year_dir / f"Q{quarter}"
            if not quarter_dir.exists():
                continue
            ascii_dirs = [quarter_dir / "ASCII", quarter_dir / "ascii"]
            if any(directory.exists() for directory in ascii_dirs):
                found.append(f"{year}Q{quarter}")
    expected = (last_year - first_year + 1) * 4
    missing = sorted(
        {f"{year}Q{quarter}" for year in range(first_year, last_year + 1) for quarter in range(1, 5)}
        - set(found)
    )
    return expected, missing


def contingency_counts(exposure: np.ndarray, event: np.ndarray) -> tuple[int, int, int, int]:
    exposure = exposure.astype(bool, copy=False)
    event = event.astype(bool, copy=False)
    a = int(np.count_nonzero(exposure & event))
    b = int(np.count_nonzero(exposure & ~event))
    c = int(np.count_nonzero(~exposure & event))
    d = int(len(exposure) - a - b - c)
    return a, b, c, d


def build_pair_table(
    data: pd.DataFrame,
    master: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    train_mask = data["year"].le(int(config["train_end_year"])).to_numpy()
    test_mask = data["year"].between(
        int(config["test_start_year"]), int(config["test_end_year"]), inclusive="both"
    ).to_numpy()
    rows: list[dict[str, Any]] = []

    for drug in master.itertuples(index=False):
        column = exposure_column(drug.drug_key, config["exposure_suffix"])
        exposure = data[column].fillna(False).to_numpy(dtype=bool)
        for event_name in config["events"]:
            event = data[event_name].fillna(False).to_numpy(dtype=bool)
            train_cells = contingency_counts(exposure[train_mask], event[train_mask])
            test_cells = contingency_counts(exposure[test_mask], event[test_mask])
            train_ror = ror_stats(*train_cells)
            test_ror = ror_stats(*test_cells)
            test_positive = (
                test_cells[0] >= int(config["positive_min_cases"])
                and test_ror.lower > float(config["positive_min_ror_lower"])
            )
            rows.append(
                {
                    "drug_key": drug.drug_key,
                    "drug_group": drug.drug_group,
                    "event": event_name,
                    "train_exposed_cases": train_cells[0] + train_cells[1],
                    "train_pair_cases": train_cells[0],
                    "train_event_cases": train_cells[0] + train_cells[2],
                    "train_ror": train_ror.ror,
                    "train_ror_lower": train_ror.lower,
                    "train_ror_upper": train_ror.upper,
                    "test_exposed_cases": test_cells[0] + test_cells[1],
                    "test_pair_cases": test_cells[0],
                    "test_event_cases": test_cells[0] + test_cells[2],
                    "test_ror": test_ror.ror,
                    "test_ror_lower": test_ror.lower,
                    "test_ror_upper": test_ror.upper,
                    "test_positive": int(test_positive),
                }
            )
    return pd.DataFrame(rows)


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2_000, random_state=20260812)),
        ]
    )


def grouped_predictions(
    table: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> np.ndarray:
    y = table["test_positive"].to_numpy(dtype=int)
    groups = table["drug_key"].to_numpy()
    predictions = np.full(len(table), np.nan, dtype=float)
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(table, y, groups):
        y_train = y[train_index]
        if np.unique(y_train).size < 2:
            continue
        model = make_model(numeric, categorical)
        model.fit(table.iloc[train_index], y_train)
        predictions[test_index] = model.predict_proba(table.iloc[test_index])[:, 1]
    return predictions


def evaluate_model(name: str, y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(probability)
    y_valid = y[valid]
    p_valid = probability[valid]
    result: dict[str, Any] = {
        "model": name,
        "n_pairs": int(valid.sum()),
        "n_positive": int(y_valid.sum()),
        "n_negative": int(len(y_valid) - y_valid.sum()),
    }
    if len(np.unique(y_valid)) < 2:
        result.update({"auprc": np.nan, "auroc": np.nan, "brier": np.nan})
        return result
    result.update(
        {
            "auprc": float(average_precision_score(y_valid, p_valid)),
            "auroc": float(roc_auc_score(y_valid, p_valid)),
            "brier": float(brier_score_loss(y_valid, p_valid)),
        }
    )
    return result


def baseline_comparison(table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    y = table["test_positive"].to_numpy(dtype=int)
    working = table.copy()
    for source in ["train_exposed_cases", "train_pair_cases", "train_event_cases"]:
        working[f"log1p_{source}"] = np.log1p(working[source].astype(float))
    working["log_train_ror"] = np.log(working["train_ror"].clip(lower=1e-6, upper=1e6))

    volume_probability = grouped_predictions(
        working,
        numeric=["log1p_train_exposed_cases"],
        categorical=["event"],
    )
    signal_probability = grouped_predictions(
        working,
        numeric=[
            "log1p_train_exposed_cases",
            "log1p_train_pair_cases",
            "log1p_train_event_cases",
            "log_train_ror",
        ],
        categorical=["event", "drug_group"],
    )
    return pd.DataFrame(
        [
            evaluate_model("volume_only", y, volume_probability),
            evaluate_model("historical_signal", y, signal_probability),
        ]
    )


def target_drug_summary(pair_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    strict = pair_table[pair_table["event"].eq("strict_fall")].copy()
    strict["test_data_ready"] = strict["test_exposed_cases"].ge(
        int(config["target_drug_min_test_exposures"])
    )
    strict["cold_start_candidate"] = (
        strict["train_exposed_cases"].le(int(config["cold_start_max_train_exposures"]))
        & strict["test_exposed_cases"].ge(int(config["cold_start_min_test_exposures"]))
    )
    return strict[
        [
            "drug_key",
            "drug_group",
            "train_exposed_cases",
            "train_pair_cases",
            "test_exposed_cases",
            "test_pair_cases",
            "test_positive",
            "test_data_ready",
            "cold_start_candidate",
        ]
    ].reset_index(drop=True)


def build_report(
    pair_table: pd.DataFrame,
    target_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    expected_quarters: int,
    missing_quarters: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    positive_pairs = int(pair_table["test_positive"].sum())
    negative_pairs = int(len(pair_table) - positive_pairs)
    ready_drugs = int(target_summary["test_data_ready"].sum())
    cold_start_drugs = target_summary.loc[target_summary["cold_start_candidate"], "drug_key"].tolist()
    metric_by_name = metrics.set_index("model")
    volume_auprc = float(metric_by_name.loc["volume_only", "auprc"])
    signal_auprc = float(metric_by_name.loc["historical_signal", "auprc"])
    auprc_gain = signal_auprc - volume_auprc

    gates = {
        "raw_faers_complete": len(missing_quarters) == 0,
        "target_time_data_ready": ready_drugs >= int(config["target_drug_min_count"]),
        "future_labels_sufficient": (
            positive_pairs >= int(config["minimum_future_positive_pairs"])
            and negative_pairs >= int(config["minimum_future_negative_pairs"])
        ),
        "historical_signal_beats_volume": auprc_gain
        >= float(config["minimum_auprc_gain_over_volume"]),
    }
    required_for_next_phase = [
        gates["raw_faers_complete"],
        gates["target_time_data_ready"],
        gates["future_labels_sufficient"],
    ]
    decision = "PROCEED_TO_GENERAL_LABEL_BUILD" if all(required_for_next_phase) else "PAUSE_AND_REVISE_SCOPE"
    return {
        "decision": decision,
        "important_note": (
            "The baseline gate is preliminary and uses only the prespecified sedative-hypnotic/event panel. "
            "It does not validate a final knowledge-graph model."
        ),
        "time_split": {
            "train_end_year": int(config["train_end_year"]),
            "test_start_year": int(config["test_start_year"]),
            "test_end_year": int(config["test_end_year"]),
        },
        "raw_faers": {
            "expected_quarters": expected_quarters,
            "available_quarters": expected_quarters - len(missing_quarters),
            "missing_quarters": missing_quarters,
        },
        "target_panel": {
            "drug_count": int(target_summary["drug_key"].nunique()),
            "drugs_with_test_data": ready_drugs,
            "cold_start_candidates": cold_start_drugs,
            "drug_event_pairs": int(len(pair_table)),
            "future_positive_pairs": positive_pairs,
            "future_negative_pairs": negative_pairs,
        },
        "baseline": {
            "volume_only_auprc": volume_auprc,
            "historical_signal_auprc": signal_auprc,
            "auprc_gain": auprc_gain,
        },
        "gates": gates,
        "next_step": (
            "Build a versioned all-drug FAERS drug-event label table, then repeat temporal and drug-cold-start "
            "validation before any GNN implementation."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    gates = report["gates"]
    panel = report["target_panel"]
    baseline = report["baseline"]
    lines = [
        "# 可行性审计结果",
        "",
        f"**决策：{report['decision']}**",
        "",
        "## 已实测",
        "",
        f"- 原始 FAERS 季度：{report['raw_faers']['available_quarters']}/{report['raw_faers']['expected_quarters']}。",
        f"- 当前目标药物：{panel['drug_count']} 种，其中 {panel['drugs_with_test_data']} 种达到未来期最低暴露量。",
        f"- 药物—事件组合：{panel['drug_event_pairs']} 对；未来期阳性 {panel['future_positive_pairs']} 对，阴性 {panel['future_negative_pairs']} 对。",
        f"- 冷启动候选药物：{', '.join(panel['cold_start_candidates']) if panel['cold_start_candidates'] else '无'}。",
        f"- 报告量基线 AUPRC：{baseline['volume_only_auprc']:.3f}。",
        f"- 历史信号模型 AUPRC：{baseline['historical_signal_auprc']:.3f}，差值 {baseline['auprc_gain']:.3f}。",
        "",
        "## 闸门",
        "",
    ]
    for name, passed in gates.items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "这一步只检验现有镇静催眠药与预设事件面板的数据基础。它不能代替全药物知识图谱的正式验证，也不能证明因果。",
            "",
            f"下一步：{report['next_step']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit temporal feasibility before building an ADR graph model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_path = resolve_path(config["main_dataset"])
    master_path = resolve_path(config["drug_master"])
    raw_root = resolve_path(config["raw_faers_root"])
    master = pd.read_csv(master_path)
    drug_keys = master["drug_key"].tolist()
    exposure_columns = [exposure_column(key, config["exposure_suffix"]) for key in drug_keys]
    columns = ["year", *exposure_columns, *config["events"]]
    data = pd.read_parquet(dataset_path, columns=columns)
    data["year"] = pd.to_numeric(data["year"], errors="coerce")

    pair_table = build_pair_table(data, master, config)
    target_summary = target_drug_summary(pair_table, config)
    metrics = baseline_comparison(pair_table, config)
    expected_quarters, missing_quarters = count_raw_quarters(
        raw_root, 2004, int(config["test_end_year"])
    )
    report = build_report(
        pair_table,
        target_summary,
        metrics,
        expected_quarters,
        missing_quarters,
        config,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_table.to_csv(args.output_dir / "drug_event_time_counts.csv", index=False, encoding="utf-8-sig")
    target_summary.to_csv(args.output_dir / "target_drug_time_summary.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(args.output_dir / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "feasibility_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, args.output_dir / "feasibility_report.md")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
