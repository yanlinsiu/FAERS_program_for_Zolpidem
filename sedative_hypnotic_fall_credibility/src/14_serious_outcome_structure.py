from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_DRUG_MASTER = PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"
DEFAULT_TABLE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_6_serious_outcomes_among_fall_reports.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "14_serious_outcome_structure_qc.csv"

GROUP_TARGETS = [
    ("z_drug", "Z-drugs"),
    ("other_z_drug", "Other Z-drugs"),
    ("benzodiazepine", "Benzodiazepines"),
    ("orexin_antagonist", "Orexin receptor antagonists"),
    ("other_insomnia_related", "Other insomnia-related drugs"),
]

SERIOUS_OUTCOME_COLUMNS = [
    ("serious_any", "Any serious outcome"),
    ("serious_hospitalization", "Hospitalization"),
    ("serious_death", "Death"),
    ("serious_disability", "Disability"),
    ("serious_life_threatening", "Life-threatening"),
    ("serious_required_intervention", "Required intervention"),
    ("serious_congenital_anomaly", "Congenital anomaly"),
    ("serious_other", "Other serious outcome"),
]


@dataclass(frozen=True)
class Target:
    analysis_level: str
    target_key: str
    target_label: str
    drug_group: str
    exposure_column: str


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator * 100


def read_available_columns(path: Path, requested_columns: list[str]) -> pd.DataFrame:
    available = pq.ParquetFile(path).schema.names
    missing = sorted(set(requested_columns) - set(available))
    if missing:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing}")
    return pd.read_parquet(path, columns=requested_columns)


def build_targets(drug_master: pd.DataFrame) -> list[Target]:
    targets: list[Target] = []
    candidates = drug_master["main_analysis_candidate"].astype(str).str.lower().isin(["yes", "count_dependent"])
    for _, row in drug_master.loc[candidates].iterrows():
        drug_key = str(row["drug_key"])
        targets.append(
            Target(
                analysis_level="drug",
                target_key=drug_key,
                target_label=f"{row['generic_name']} ({row['generic_name_cn']})",
                drug_group=str(row["drug_group"]),
                exposure_column=f"exposure_{drug_key}_ps_ss",
            )
        )

    for group_key, group_label in GROUP_TARGETS:
        targets.append(
            Target(
                analysis_level="group",
                target_key=group_key,
                target_label=group_label,
                drug_group=group_key,
                exposure_column=f"exposure_{group_key}_ps_ss",
            )
        )
    return targets


def summarize_target(df: pd.DataFrame, target: Target) -> dict[str, object]:
    eligible = safe_bool(df["analysis_eligible_main"])
    exposed = safe_bool(df[target.exposure_column])
    strict_fall = safe_bool(df["strict_fall"])
    target_falls = eligible & exposed & strict_fall
    fall_n = int(target_falls.sum())

    row: dict[str, object] = {
        "analysis_level": target.analysis_level,
        "target_key": target.target_key,
        "target_label": target.target_label,
        "drug_group": target.drug_group,
        "exposure_column": target.exposure_column,
        "fall_report_n": fall_n,
    }

    for column, label in SERIOUS_OUTCOME_COLUMNS:
        outcome_n = int((target_falls & safe_bool(df[column])).sum())
        row[f"{column}_n"] = outcome_n
        row[f"{column}_percent_among_fall_reports"] = percent(outcome_n, fall_n)

    serious_any_n = int(row["serious_any_n"])
    row["no_serious_outcome_n"] = fall_n - serious_any_n
    row["no_serious_outcome_percent_among_fall_reports"] = percent(fall_n - serious_any_n, fall_n)

    row["note"] = (
        "Denominator is analysis-eligible strict fall reports with the target PS+SS exposure. "
        "These are FAERS reporting proportions, not clinical incidence rates."
    )
    return row


def build_table(df: pd.DataFrame, targets: list[Target]) -> pd.DataFrame:
    rows = [summarize_target(df, target) for target in targets]
    table = pd.DataFrame(rows)
    return table.sort_values(["analysis_level", "drug_group", "fall_report_n"], ascending=[True, True, False])


def build_qc(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    strict_fall = safe_bool(df["strict_fall"])
    rows = [
        {"qc_domain": "serious_outcome_structure", "metric": "input_rows", "value": len(df), "note": ""},
        {
            "qc_domain": "serious_outcome_structure",
            "metric": "analysis_eligible_rows",
            "value": int(eligible.sum()),
            "note": "",
        },
        {
            "qc_domain": "serious_outcome_structure",
            "metric": "strict_fall_rows",
            "value": int((eligible & strict_fall).sum()),
            "note": "",
        },
        {"qc_domain": "serious_outcome_structure", "metric": "summary_rows", "value": len(table), "note": ""},
        {
            "qc_domain": "serious_outcome_structure",
            "metric": "min_fall_report_n",
            "value": int(table["fall_report_n"].min()),
            "note": "Small denominators should be interpreted cautiously.",
        },
        {
            "qc_domain": "serious_outcome_structure",
            "metric": "max_fall_report_n",
            "value": int(table["fall_report_n"].max()),
            "note": "",
        },
    ]
    for column, _ in SERIOUS_OUTCOME_COLUMNS:
        rows.append(
            {
                "qc_domain": "serious_outcome_structure",
                "metric": f"available__{column}",
                "value": int(column in df.columns),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def validate(table: pd.DataFrame) -> None:
    if table.empty:
        raise ValueError("Serious outcome structure table is empty.")
    zolpidem = table[table["target_key"].eq("zolpidem")]
    if zolpidem.empty:
        raise ValueError("Serious outcome structure table is missing zolpidem.")
    if int(zolpidem.iloc[0]["fall_report_n"]) != 986:
        raise ValueError("Zolpidem fall report count does not match the established main-analysis count.")
    percent_columns = [column for column in table.columns if column.endswith("_percent_among_fall_reports")]
    invalid = table[percent_columns].stack().dropna().lt(0).any() or table[percent_columns].stack().dropna().gt(100).any()
    if invalid:
        raise ValueError("Serious outcome percentages must be between 0 and 100.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--drug-master", type=Path, default=DEFAULT_DRUG_MASTER)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    if not args.main_dataset.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {args.main_dataset}")
    if not args.drug_master.exists():
        raise FileNotFoundError(f"Drug master not found: {args.drug_master}")

    drug_master = pd.read_csv(args.drug_master)
    targets = build_targets(drug_master)
    requested_columns = [
        "analysis_eligible_main",
        "strict_fall",
        *[column for column, _ in SERIOUS_OUTCOME_COLUMNS],
        *sorted({target.exposure_column for target in targets}),
    ]
    df = read_available_columns(args.main_dataset, requested_columns)
    table = build_table(df, targets)
    qc = build_qc(df, table)
    validate(table)

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.qc_out}")
    print(
        table[
            [
                "analysis_level",
                "target_key",
                "fall_report_n",
                "serious_any_percent_among_fall_reports",
                "serious_hospitalization_percent_among_fall_reports",
                "serious_death_percent_among_fall_reports",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
