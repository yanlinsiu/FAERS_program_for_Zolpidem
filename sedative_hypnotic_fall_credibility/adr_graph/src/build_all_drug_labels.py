from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ADR_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ADR_DIR / "configs" / "all_drug_labels.json"
DEFAULT_OUTPUT_DIR = ADR_DIR / "outputs" / "all_drug_labels"
ANNUAL_DRUG_PATTERN = re.compile(r"drug_(\d{4})\.parquet$", re.I)
INVALID_DRUG_TERMS = {
    "",
    "UNKNOWN",
    "UNK",
    "UNSPECIFIED",
    "NOT SPECIFIED",
    "NOT REPORTED",
    "MULTIPLE",
    "OTHER",
    "NONE",
    "NA",
    "N A",
}


@dataclass(frozen=True)
class RorResult:
    ror: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ADR_DIR / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def normalize_drug_series(series: pd.Series) -> pd.Series:
    normalized = (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return normalized.mask(normalized.isin(INVALID_DRUG_TERMS), "")


def stable_drug_id(canonical_key: str) -> str:
    digest = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16]
    return f"drug_{digest}"


def find_annual_drug_files(root: Path, first_year: int, last_year: int) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in root.glob("*/drug_*.parquet"):
        match = ANNUAL_DRUG_PATTERN.fullmatch(path.name)
        if not match:
            continue
        year = int(match.group(1))
        if first_year <= year <= last_year:
            if year in files:
                raise ValueError(f"Duplicate annual drug file for {year}: {path} and {files[year]}")
            files[year] = path
    missing = sorted(set(range(first_year, last_year + 1)) - set(files))
    if missing:
        raise FileNotFoundError(f"Missing annual drug parquet files: {missing}")
    return dict(sorted(files.items()))


def load_case_data(path: Path, events: list[str], first_year: int, last_year: int) -> pd.DataFrame:
    columns = ["caseid", "primaryid_example", "year", *events]
    data = pd.read_parquet(path, columns=columns).rename(columns={"primaryid_example": "primaryid"})
    data["caseid"] = normalize_id(data["caseid"])
    data["primaryid"] = normalize_id(data["primaryid"])
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data = data[data["year"].between(first_year, last_year, inclusive="both")].copy()
    for event in events:
        data[event] = data[event].fillna(False).astype(bool)
    if data["caseid"].eq("").any() or data["primaryid"].eq("").any():
        raise ValueError("Case data contains blank caseid or primaryid values.")
    if data["caseid"].duplicated().any():
        raise ValueError("Case data contains duplicated caseid values.")
    if data.duplicated(["caseid", "primaryid"]).any():
        raise ValueError("Case data contains duplicated caseid-primaryid pairs.")
    return data


def read_suspect_drug_rows(path: Path, case_keys: pd.DataFrame, roles: set[str]) -> tuple[pd.DataFrame, int]:
    columns = ["primaryid", "caseid", "role_cod", "drugname", "prod_ai"]
    raw = pd.read_parquet(path, columns=columns)
    raw_rows = len(raw)
    raw["primaryid"] = normalize_id(raw["primaryid"])
    raw["caseid"] = normalize_id(raw["caseid"])
    raw["role_cod"] = raw["role_cod"].fillna("").astype(str).str.upper().str.strip()
    selected = raw[raw["role_cod"].isin(roles)].merge(
        case_keys[["caseid", "primaryid"]].drop_duplicates(),
        on=["caseid", "primaryid"],
        how="inner",
        validate="many_to_one",
    )
    selected["drugname_norm"] = normalize_drug_series(selected["drugname"])
    selected["prod_ai_norm"] = normalize_drug_series(selected["prod_ai"])
    return selected, raw_rows


def build_alias_map(
    files: dict[int, Path],
    cases_by_year: dict[int, pd.DataFrame],
    roles: set[str],
    min_rows: int,
    min_top_share: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    grouped_chunks: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    for index, (year, path) in enumerate(files.items(), start=1):
        year_cases = cases_by_year[year]
        selected, raw_rows = read_suspect_drug_rows(path, year_cases, roles)
        usable = selected[selected["drugname_norm"].ne("") & selected["prod_ai_norm"].ne("")]
        if not usable.empty:
            counts = (
                usable.groupby(["drugname_norm", "prod_ai_norm"], observed=True)
                .size()
                .rename("row_count")
                .reset_index()
            )
            grouped_chunks.append(counts)
        inventory.append(
            {
                "year": year,
                "raw_drug_rows": raw_rows,
                "matched_suspect_rows": len(selected),
                "matched_suspect_cases": selected["primaryid"].nunique(),
                "reported_prod_ai_rows": int(selected["prod_ai_norm"].ne("").sum()),
            }
        )
        print(
            f"[alias {index:02d}/{len(files)}] {year}: "
            f"suspect_rows={len(selected):,}, active_ingredient_rows={len(usable):,}",
            flush=True,
        )

    if not grouped_chunks:
        return pd.DataFrame(columns=["drugname_norm", "prod_ai_norm", "row_count", "total_rows", "top_share"]), inventory
    pair_counts = (
        pd.concat(grouped_chunks, ignore_index=True)
        .groupby(["drugname_norm", "prod_ai_norm"], observed=True)["row_count"]
        .sum()
        .reset_index()
    )
    totals = pair_counts.groupby("drugname_norm", observed=True)["row_count"].sum().rename("total_rows")
    pair_counts = pair_counts.merge(totals, on="drugname_norm", how="left")
    pair_counts["top_share"] = pair_counts["row_count"] / pair_counts["total_rows"]
    pair_counts = pair_counts.sort_values(
        ["drugname_norm", "row_count", "prod_ai_norm"], ascending=[True, False, True], kind="mergesort"
    )
    best = pair_counts.drop_duplicates("drugname_norm", keep="first")
    accepted = best[
        best["total_rows"].ge(min_rows) & best["top_share"].ge(min_top_share)
    ].reset_index(drop=True)
    return accepted, inventory


def assign_canonical_drug(rows: pd.DataFrame, alias_lookup: dict[str, str]) -> pd.DataFrame:
    work = rows.copy()
    learned_ai = work["drugname_norm"].map(alias_lookup).fillna("")
    direct_ai = work["prod_ai_norm"]
    resolved_ai = direct_ai.where(direct_ai.ne(""), learned_ai)
    has_ai = resolved_ai.ne("")
    work["canonical_name"] = resolved_ai.where(has_ai, work["drugname_norm"])
    work["canonical_basis"] = np.where(has_ai, "active_ingredient_resolved", "name_only")
    work["resolution_source"] = np.select(
        [direct_ai.ne(""), direct_ai.eq("") & learned_ai.ne("")],
        ["reported_prod_ai", "learned_name_to_ai"],
        default="normalized_drugname",
    )
    work = work[work["canonical_name"].ne("")].copy()
    work["canonical_key"] = np.where(
        work["canonical_basis"].eq("active_ingredient_resolved"),
        "AI|" + work["canonical_name"],
        "NAME|" + work["canonical_name"],
    )
    unique_keys = work["canonical_key"].drop_duplicates()
    id_lookup = {key: stable_drug_id(key) for key in unique_keys}
    if len(set(id_lookup.values())) != len(id_lookup):
        raise RuntimeError("Stable drug identifier collision detected.")
    work["drug_id"] = work["canonical_key"].map(id_lookup)
    return work


def aggregate_period(
    files: dict[int, Path],
    cases_by_year: dict[int, pd.DataFrame],
    roles: set[str],
    events: list[str],
    alias_lookup: dict[str, str],
    train_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    period_chunks: dict[str, list[pd.DataFrame]] = {"train": [], "test": []}
    matched_caseids: set[str] = set()
    for index, (year, path) in enumerate(files.items(), start=1):
        year_cases = cases_by_year[year]
        selected, _ = read_suspect_drug_rows(path, year_cases, roles)
        selected = assign_canonical_drug(selected, alias_lookup)
        selected = selected.merge(
            year_cases[["caseid", "primaryid", *events]],
            on=["caseid", "primaryid"],
            how="inner",
            validate="many_to_one",
        )
        selected = selected.drop_duplicates(["caseid", "drug_id"], keep="first")
        matched_caseids.update(selected["caseid"])
        aggregated = (
            selected.groupby(["drug_id", "canonical_name", "canonical_basis"], observed=True)
            .agg(exposed_cases=("caseid", "size"), **{event: (event, "sum") for event in events})
            .reset_index()
        )
        period = "train" if year <= train_end_year else "test"
        period_chunks[period].append(aggregated)
        print(
            f"[count {index:02d}/{len(files)}] {year}: "
            f"case-drug links={len(selected):,}, canonical_drugs={selected['drug_id'].nunique():,}",
            flush=True,
        )

    outputs: list[pd.DataFrame] = []
    for period in ["train", "test"]:
        combined = pd.concat(period_chunks[period], ignore_index=True)
        combined = (
            combined.groupby(["drug_id", "canonical_name", "canonical_basis"], observed=True)[
                ["exposed_cases", *events]
            ]
            .sum()
            .reset_index()
        )
        outputs.append(combined)
    return outputs[0], outputs[1], matched_caseids


def ror_arrays(a: Iterable[int], b: Iterable[int], c: Iterable[int], d: Iterable[int]) -> RorResult:
    cells = np.column_stack([a, b, c, d]).astype(float)
    if np.any(cells < 0):
        raise ValueError("A 2x2 contingency cell is negative.")
    needs_correction = np.any(cells == 0, axis=1)
    cells[needs_correction] += 0.5
    aa, bb, cc, dd = cells.T
    ror = (aa * dd) / (bb * cc)
    se = np.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    log_ror = np.log(ror)
    return RorResult(ror=ror, lower=np.exp(log_ror - 1.96 * se), upper=np.exp(log_ror + 1.96 * se))


def period_pair_rows(
    aggregate: pd.DataFrame,
    events: list[str],
    total_cases: int,
    event_totals: dict[str, int],
    prefix: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for event in events:
        frame = aggregate[["drug_id", "canonical_name", "canonical_basis", "exposed_cases", event]].copy()
        frame = frame.rename(columns={"exposed_cases": f"{prefix}_exposed_cases", event: f"{prefix}_pair_cases"})
        frame["event"] = event
        frame[f"{prefix}_total_cases"] = total_cases
        frame[f"{prefix}_event_cases"] = int(event_totals[event])
        a = frame[f"{prefix}_pair_cases"].to_numpy(dtype=int)
        b = frame[f"{prefix}_exposed_cases"].to_numpy(dtype=int) - a
        c = frame[f"{prefix}_event_cases"].to_numpy(dtype=int) - a
        d = total_cases - a - b - c
        stats = ror_arrays(a, b, c, d)
        frame[f"{prefix}_ror"] = stats.ror
        frame[f"{prefix}_ror_lower"] = stats.lower
        frame[f"{prefix}_ror_upper"] = stats.upper
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_label_table(
    train_aggregate: pd.DataFrame,
    test_aggregate: pd.DataFrame,
    case_data: pd.DataFrame,
    events: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    train_cases = case_data[case_data["year"].le(int(config["train_end_year"]))]
    test_cases = case_data[
        case_data["year"].between(int(config["test_start_year"]), int(config["test_end_year"]), inclusive="both")
    ]
    train_rows = period_pair_rows(
        train_aggregate,
        events,
        len(train_cases),
        {event: int(train_cases[event].sum()) for event in events},
        "train",
    )
    test_rows = period_pair_rows(
        test_aggregate,
        events,
        len(test_cases),
        {event: int(test_cases[event].sum()) for event in events},
        "test",
    )
    labels = train_rows.merge(
        test_rows,
        on=["drug_id", "canonical_name", "canonical_basis", "event"],
        how="outer",
        validate="one_to_one",
    )
    integer_columns = [column for column in labels if column.endswith("_cases")]
    labels[integer_columns] = labels[integer_columns].fillna(0).astype("int64")
    for prefix, total, period_cases in [
        ("train", len(train_cases), train_cases),
        ("test", len(test_cases), test_cases),
    ]:
        for event in events:
            mask = labels["event"].eq(event)
            labels.loc[mask, f"{prefix}_total_cases"] = total
            labels.loc[mask, f"{prefix}_event_cases"] = int(period_cases[event].sum())
        missing_stats = labels[f"{prefix}_ror"].isna()
        if missing_stats.any():
            a = labels.loc[missing_stats, f"{prefix}_pair_cases"].to_numpy(dtype=int)
            b = labels.loc[missing_stats, f"{prefix}_exposed_cases"].to_numpy(dtype=int) - a
            c = labels.loc[missing_stats, f"{prefix}_event_cases"].to_numpy(dtype=int) - a
            d = labels.loc[missing_stats, f"{prefix}_total_cases"].to_numpy(dtype=int) - a - b - c
            stats = ror_arrays(a, b, c, d)
            labels.loc[missing_stats, f"{prefix}_ror"] = stats.ror
            labels.loc[missing_stats, f"{prefix}_ror_lower"] = stats.lower
            labels.loc[missing_stats, f"{prefix}_ror_upper"] = stats.upper

    labels["train_signal_positive"] = (
        labels["train_pair_cases"].ge(int(config["positive_min_cases"]))
        & labels["train_ror_lower"].gt(float(config["positive_min_ror_lower"]))
    )
    labels["test_expected_pair_cases"] = (
        labels["test_exposed_cases"] * labels["test_event_cases"] / labels["test_total_cases"].clip(lower=1)
    )
    positive = (
        labels["test_pair_cases"].ge(int(config["positive_min_cases"]))
        & labels["test_ror_lower"].gt(float(config["positive_min_ror_lower"]))
    )
    evaluable = (
        labels["test_exposed_cases"].ge(int(config["evaluable_min_exposures"]))
        & labels["test_expected_pair_cases"].ge(float(config["evaluable_min_expected_cases"]))
    )
    labels["future_label_status"] = np.select(
        [positive, evaluable & ~positive],
        ["positive", "operational_negative"],
        default="insufficient_information",
    )
    labels["future_label"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels.loc[positive, "future_label"] = 1
    labels.loc[evaluable & ~positive, "future_label"] = 0
    labels["cold_start_candidate"] = (
        labels["train_exposed_cases"].le(int(config["cold_start_max_train_exposures"]))
        & labels["test_exposed_cases"].ge(int(config["cold_start_min_test_exposures"]))
    )
    labels["label_definition"] = (
        "future FAERS signal; positive if pair cases and ROR lower-bound thresholds pass; "
        "0 is an operational non-signal, not proof of no biological risk"
    )
    return labels.sort_values(["event", "canonical_name", "drug_id"], kind="mergesort").reset_index(drop=True)


def build_vocabulary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    vocab = train[["drug_id", "canonical_name", "canonical_basis", "exposed_cases"]].rename(
        columns={"exposed_cases": "train_exposed_cases"}
    ).merge(
        test[["drug_id", "canonical_name", "canonical_basis", "exposed_cases"]].rename(
            columns={"exposed_cases": "test_exposed_cases"}
        ),
        on=["drug_id", "canonical_name", "canonical_basis"],
        how="outer",
        validate="one_to_one",
    )
    vocab[["train_exposed_cases", "test_exposed_cases"]] = vocab[
        ["train_exposed_cases", "test_exposed_cases"]
    ].fillna(0).astype("int64")
    vocab["observed_both_periods"] = vocab["train_exposed_cases"].gt(0) & vocab["test_exposed_cases"].gt(0)
    return vocab.sort_values(["canonical_basis", "canonical_name"], kind="mergesort").reset_index(drop=True)


def build_qc(
    case_data: pd.DataFrame,
    matched_caseids: set[str],
    vocabulary: pd.DataFrame,
    labels: pd.DataFrame,
    alias_map: pd.DataFrame,
    inventory: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    status_counts = (
        labels.groupby(["event", "future_label_status"], observed=True)
        .size()
        .rename("pair_count")
        .reset_index()
    )
    total_exposures = int(vocabulary[["train_exposed_cases", "test_exposed_cases"]].sum().sum())
    name_only_exposures = int(
        vocabulary.loc[vocabulary["canonical_basis"].eq("name_only"), ["train_exposed_cases", "test_exposed_cases"]]
        .sum()
        .sum()
    )
    summary = {
        "case_count": int(len(case_data)),
        "case_drug_coverage": float(len(matched_caseids) / max(len(case_data), 1)),
        "drug_count": int(vocabulary["drug_id"].nunique()),
        "drugs_observed_both_periods": int(vocabulary["observed_both_periods"].sum()),
        "accepted_name_to_ingredient_aliases": int(len(alias_map)),
        "name_only_exposure_fraction": float(name_only_exposures / max(total_exposures, 1)),
        "drug_event_pair_count": int(len(labels)),
        "evaluable_pair_count": int(labels["future_label"].notna().sum()),
        "positive_pair_count": int(labels["future_label"].eq(1).sum()),
        "operational_negative_pair_count": int(labels["future_label"].eq(0).sum()),
        "cold_start_pair_count": int(labels["cold_start_candidate"].sum()),
    }
    thresholds = config["quality_gates"]
    gates = {
        "case_drug_coverage": summary["case_drug_coverage"] >= float(thresholds["minimum_case_drug_coverage"]),
        "name_resolution": summary["name_only_exposure_fraction"]
        <= float(thresholds["maximum_name_only_exposure_fraction"]),
        "drugs_observed_both_periods": summary["drugs_observed_both_periods"]
        >= int(thresholds["minimum_drugs_observed_both_periods"]),
        "evaluable_pairs": summary["evaluable_pair_count"] >= int(thresholds["minimum_evaluable_pairs"]),
        "positive_pairs": summary["positive_pair_count"] >= int(thresholds["minimum_positive_pairs"]),
        "operational_negative_pairs": summary["operational_negative_pair_count"]
        >= int(thresholds["minimum_operational_negative_pairs"]),
    }
    report = {
        "decision": "PROCEED_TO_PHARMACOLOGY_MAPPING" if all(gates.values()) else "PAUSE_AND_REPAIR_LABEL_DATA",
        "time_split": {
            "train_end_year": int(config["train_end_year"]),
            "test_start_year": int(config["test_start_year"]),
            "test_end_year": int(config["test_end_year"]),
        },
        "summary": summary,
        "quality_gates": gates,
        "label_warning": (
            "future_label=0 means sufficiently observed without meeting the prespecified FAERS signal rule. "
            "It is not a verified absence of biological adverse-event risk."
        ),
        "source_warning": (
            "This build reuses the existing elderly case cohort and its retained primaryid/outcome definitions. "
            "A latest-version/deleted-case reconstruction remains required before a publication-grade final model."
        ),
        "next_step": (
            "Map the resolved ingredient vocabulary to PubChem/ChEMBL/Open Targets identifiers, freeze a mappable "
            "modeling cohort, and rerun temporal/cold-start baselines before any GNN."
        ),
    }
    inventory_summary = inventory.assign(
        reported_prod_ai_fraction=lambda frame: frame["reported_prod_ai_rows"]
        / frame["matched_suspect_rows"].clip(lower=1)
    )
    qc = pd.concat(
        [
            status_counts.assign(qc_table="label_status_by_event"),
            inventory_summary.assign(qc_table="source_inventory"),
        ],
        ignore_index=True,
        sort=False,
    )
    return qc, report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# 全药物标签集审计",
        "",
        f"**决策：{report['decision']}**",
        "",
        "## 核心结果",
        "",
        f"- 老年病例：{summary['case_count']:,}。",
        f"- 至少匹配一个 PS/SS 药物的病例比例：{summary['case_drug_coverage']:.1%}。",
        f"- 归一化药物实体：{summary['drug_count']:,}；训练期和测试期均出现：{summary['drugs_observed_both_periods']:,}。",
        f"- 药物—事件组合：{summary['drug_event_pair_count']:,}；可评价：{summary['evaluable_pair_count']:,}。",
        f"- 未来阳性：{summary['positive_pair_count']:,}；操作性阴性：{summary['operational_negative_pair_count']:,}。",
        f"- 冷启动组合：{summary['cold_start_pair_count']:,}。",
        f"- 仍只能按药名识别的暴露比例：{summary['name_only_exposure_fraction']:.1%}。",
        "",
        "## 质量闸门",
        "",
    ]
    for name, passed in report["quality_gates"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines.extend(
        [
            "",
            "## 必须保留的解释边界",
            "",
            f"- {report['label_warning']}",
            f"- {report['source_warning']}",
            "",
            f"下一步：{report['next_step']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal all-drug FAERS event labels for the ADR graph project.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = load_config(args.config)
    events = list(config["events"])
    first_year = 2004
    last_year = int(config["test_end_year"])
    case_data = load_case_data(resolve_path(config["case_dataset"]), events, first_year, last_year)
    cases_by_year = {year: frame.copy() for year, frame in case_data.groupby("year", observed=True)}
    files = find_annual_drug_files(resolve_path(config["drug_root"]), first_year, last_year)
    roles = {str(role).upper() for role in config["suspect_roles"]}

    alias_map, inventory_rows = build_alias_map(
        files,
        cases_by_year,
        roles,
        int(config["alias_min_rows"]),
        float(config["alias_min_top_share"]),
    )
    alias_lookup = dict(zip(alias_map["drugname_norm"], alias_map["prod_ai_norm"]))
    train_aggregate, test_aggregate, matched_caseids = aggregate_period(
        files,
        cases_by_year,
        roles,
        events,
        alias_lookup,
        int(config["train_end_year"]),
    )
    vocabulary = build_vocabulary(train_aggregate, test_aggregate)
    labels = build_label_table(train_aggregate, test_aggregate, case_data, events, config)
    inventory = pd.DataFrame(inventory_rows)
    qc, report = build_qc(case_data, matched_caseids, vocabulary, labels, alias_map, inventory, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.output_dir / "all_drug_event_labels.parquet", index=False)
    vocabulary.to_parquet(args.output_dir / "drug_vocabulary.parquet", index=False)
    vocabulary.to_csv(args.output_dir / "drug_vocabulary.csv", index=False, encoding="utf-8-sig")
    alias_map.to_csv(args.output_dir / "name_to_ingredient_map.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(args.output_dir / "label_build_qc.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "label_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, args.output_dir / "label_build_report.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
