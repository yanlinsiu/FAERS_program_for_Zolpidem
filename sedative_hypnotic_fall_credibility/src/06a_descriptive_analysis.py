from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_DRUG_MASTER = PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"
DEFAULT_BASELINE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_1_baseline_description.csv"
DEFAULT_EXPOSURE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_1b_drug_exposure_description.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "06a_descriptive_analysis_qc.csv"
DEFAULT_FLOW_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_1_study_flow.png"

GROUP_TARGETS = [
    ("z_drug", "Z-drugs"),
    ("other_z_drug", "Other Z-drugs"),
    ("benzodiazepine", "Benzodiazepines"),
    ("orexin_antagonist", "Orexin receptor antagonists"),
    ("other_insomnia_related", "Other insomnia-related drugs"),
]

BASELINE_COLUMNS = [
    "analysis_eligible_main",
    "strict_fall",
    "age_group_3",
    "sex_clean",
    "regulatory_period",
    "country_group",
    "rept_cod",
    "e_sub",
    "polypharmacy",
    "polypharmacy_5",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
]

BASELINE_VARIABLES = [
    ("Study population", "analysis_eligible_main", "true_only"),
    ("Strict fall report", "strict_fall", "true_only"),
    ("Age group", "age_group_3", "categorical"),
    ("Sex", "sex_clean", "categorical"),
    ("Regulatory period", "regulatory_period", "categorical"),
    ("Country group", "country_group", "categorical"),
    ("Reporter type", "rept_cod", "categorical"),
    ("Electronic submission", "e_sub", "categorical"),
    ("Polypharmacy", "polypharmacy", "true_only"),
    ("Polypharmacy >=5 drugs", "polypharmacy_5", "true_only"),
    ("Concomitant antidepressant", "is_antidepressant", "true_only"),
    ("Concomitant antipsychotic", "is_antipsychotic", "true_only"),
    ("Concomitant opioid", "is_opioid", "true_only"),
    ("Concomitant antiepileptic", "is_antiepileptic", "true_only"),
    ("Insomnia indication", "indi_insomnia", "true_only"),
    ("Anxiety indication", "indi_anxiety", "true_only"),
    ("Depression indication", "indi_depression", "true_only"),
    ("Pain indication", "indi_pain", "true_only"),
    ("Epilepsy indication", "indi_epilepsy", "true_only"),
]

FLOW_EXPOSURES = [
    ("Zolpidem PS+SS", "exposure_zolpidem_ps_ss"),
    ("Other Z-drugs PS+SS", "exposure_other_z_drug_ps_ss"),
    ("Benzodiazepines PS+SS", "exposure_benzodiazepine_ps_ss"),
    ("Orexin antagonists PS+SS", "exposure_orexin_antagonist_ps_ss"),
    ("Other insomnia-related PS+SS", "exposure_other_insomnia_related_ps_ss"),
]


@dataclass(frozen=True)
class DrugTarget:
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


def format_count(n: int) -> str:
    return f"{n:,}"


def read_available_columns(path: Path, requested_columns: list[str]) -> pd.DataFrame:
    available_columns = pq.ParquetFile(path).schema.names
    read_columns = [column for column in requested_columns if column in available_columns]
    missing = sorted(set(requested_columns) - set(read_columns))
    if missing:
        raise ValueError(f"Main analysis dataset is missing required descriptive columns: {missing}")
    return pd.read_parquet(path, columns=read_columns)


def build_drug_targets(drug_master: pd.DataFrame) -> list[DrugTarget]:
    targets: list[DrugTarget] = []
    candidates = drug_master["main_analysis_candidate"].astype(str).str.lower().isin(["yes", "count_dependent"])
    for _, row in drug_master.loc[candidates].iterrows():
        drug_key = str(row["drug_key"])
        label = f"{row['generic_name']} ({row['generic_name_cn']})"
        targets.append(
            DrugTarget(
                analysis_level="drug",
                target_key=drug_key,
                target_label=label,
                drug_group=str(row["drug_group"]),
                exposure_column=f"exposure_{drug_key}_ps_ss",
            )
        )

    for group_key, group_label in GROUP_TARGETS:
        targets.append(
            DrugTarget(
                analysis_level="group",
                target_key=group_key,
                target_label=group_label,
                drug_group=group_key,
                exposure_column=f"exposure_{group_key}_ps_ss",
            )
        )
    return targets


def build_baseline_table(df: pd.DataFrame) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible].copy()
    strict_fall = safe_bool(analysis_df["strict_fall"])
    total_n = len(analysis_df)
    fall_total_n = int(strict_fall.sum())

    rows = []
    for variable_label, column, mode in BASELINE_VARIABLES:
        if mode == "true_only":
            mask = safe_bool(analysis_df[column])
            categories = [("Yes", mask)]
        else:
            counts = analysis_df[column].fillna("Missing").astype(str).value_counts(dropna=False)
            categories = [(str(category), analysis_df[column].fillna("Missing").astype(str) == str(category)) for category in counts.index]

        for category, mask in categories:
            category_n = int(mask.sum())
            category_fall_n = int((mask & strict_fall).sum())
            rows.append(
                {
                    "variable": variable_label,
                    "category": category,
                    "overall_n": category_n,
                    "overall_percent": percent(category_n, total_n),
                    "strict_fall_n": category_fall_n,
                    "strict_fall_percent_within_category": percent(category_fall_n, category_n),
                    "strict_fall_percent_of_all_falls": percent(category_fall_n, fall_total_n),
                }
            )

    return pd.DataFrame(rows)


def build_exposure_table(df: pd.DataFrame, targets: list[DrugTarget]) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible].copy()
    strict_fall = safe_bool(analysis_df["strict_fall"])
    total_n = len(analysis_df)

    rows = []
    for target in targets:
        if target.exposure_column not in analysis_df.columns:
            rows.append(
                {
                    "analysis_level": target.analysis_level,
                    "target_key": target.target_key,
                    "target_label": target.target_label,
                    "drug_group": target.drug_group,
                    "exposure_column": target.exposure_column,
                    "column_present": False,
                    "exposed_n": pd.NA,
                    "exposed_percent": pd.NA,
                    "strict_fall_n": pd.NA,
                    "strict_fall_percent_within_exposed": pd.NA,
                    "serious_fall_n": pd.NA,
                    "serious_fall_percent_within_fall": pd.NA,
                }
            )
            continue

        exposure = safe_bool(analysis_df[target.exposure_column])
        exposed_n = int(exposure.sum())
        fall_n = int((exposure & strict_fall).sum())
        serious_fall_n = int((exposure & strict_fall & safe_bool(analysis_df["serious_any"])).sum())
        rows.append(
            {
                "analysis_level": target.analysis_level,
                "target_key": target.target_key,
                "target_label": target.target_label,
                "drug_group": target.drug_group,
                "exposure_column": target.exposure_column,
                "column_present": True,
                "exposed_n": exposed_n,
                "exposed_percent": percent(exposed_n, total_n),
                "strict_fall_n": fall_n,
                "strict_fall_percent_within_exposed": percent(fall_n, exposed_n),
                "serious_fall_n": serious_fall_n,
                "serious_fall_percent_within_fall": percent(serious_fall_n, fall_n),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values(["analysis_level", "drug_group", "exposed_n"], ascending=[True, True, False])


def build_qc(df: pd.DataFrame, baseline: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible]
    rows = [
        {"qc_domain": "descriptive_analysis", "metric": "input_rows", "value": len(df), "note": ""},
        {"qc_domain": "descriptive_analysis", "metric": "analysis_eligible_rows", "value": len(analysis_df), "note": ""},
        {"qc_domain": "descriptive_analysis", "metric": "strict_fall_rows", "value": int(safe_bool(analysis_df["strict_fall"]).sum()), "note": ""},
        {"qc_domain": "descriptive_analysis", "metric": "baseline_rows", "value": len(baseline), "note": ""},
        {"qc_domain": "descriptive_analysis", "metric": "exposure_rows", "value": len(exposure), "note": ""},
        {
            "qc_domain": "descriptive_analysis",
            "metric": "missing_exposure_columns",
            "value": int((~safe_bool(exposure["column_present"])).sum()),
            "note": "",
        },
    ]
    return pd.DataFrame(rows)


def validate_results(df: pd.DataFrame, baseline: pd.DataFrame, exposure: pd.DataFrame) -> None:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible]
    total_n = len(analysis_df)
    fall_n = int(safe_bool(analysis_df["strict_fall"]).sum())

    population_row = baseline[(baseline["variable"] == "Study population") & (baseline["category"] == "Yes")]
    if population_row.empty or int(population_row.iloc[0]["overall_n"]) != total_n:
        raise ValueError("Baseline population row does not match analysis eligible rows.")

    fall_row = baseline[(baseline["variable"] == "Strict fall report") & (baseline["category"] == "Yes")]
    if fall_row.empty or int(fall_row.iloc[0]["overall_n"]) != fall_n:
        raise ValueError("Baseline strict fall row does not match strict fall rows.")

    zolpidem = exposure[exposure["target_key"] == "zolpidem"]
    if zolpidem.empty:
        raise ValueError("Exposure description is missing zolpidem.")
    if int(zolpidem.iloc[0]["strict_fall_n"]) != 986:
        raise ValueError("Zolpidem strict fall count does not match existing QC value.")


def plot_study_flow(df: pd.DataFrame, figure_out: Path) -> None:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible].copy()
    strict_fall = safe_bool(analysis_df["strict_fall"])
    total_n = len(analysis_df)
    fall_n = int(strict_fall.sum())

    exposure_lines = []
    for label, column in FLOW_EXPOSURES:
        exposure = safe_bool(analysis_df[column])
        exposed_n = int(exposure.sum())
        exposed_fall_n = int((exposure & strict_fall).sum())
        exposure_lines.append(f"{label}: {format_count(exposed_n)} reports; {format_count(exposed_fall_n)} strict fall")

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.axis("off")

    def add_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        facecolor: str = "#ffffff",
        edgecolor: str = "#333333",
        title_size: float = 10.5,
        body_size: float = 8.5,
        title_y: float = 0.72,
        body_y: float = 0.38,
    ) -> None:
        rect = plt.Rectangle((x, y), w, h, fill=True, linewidth=1.1, facecolor=facecolor, edgecolor=edgecolor)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, y + h * body_y, body, ha="center", va="center", fontsize=body_size, linespacing=1.32)

    def add_arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        arrow_style = dict(arrowstyle="-|>", linewidth=1.05, color="#333333", mutation_scale=12)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)

    source_color = "#eef3f8"
    definition_color = "#f8f3e8"
    output_color = "#eef7ef"
    interpretation_color = "#f4eef8"

    ax.text(0.04, 0.95, "Study design and analytic workflow", fontsize=16, fontweight="bold", va="top")
    ax.text(
        0.04,
        0.905,
        "Signal-stability-phenotype framework for sedative-hypnotic-related strict-fall reports in older adults",
        fontsize=10.5,
        va="top",
    )

    main_boxes = [
        (
            "FAERS source data",
            "Quarterly reports\n2004Q1-2025Q4",
            source_color,
        ),
        (
            "Older-adult case base",
            f"Deduplicated case-level reports\nAged >=65 years\nn = {format_count(total_n)}",
            source_color,
        ),
        (
            "Exposure and endpoint\ndefinitions",
            "Target sedative-hypnotics\nPS+SS exposure roles\nStrict fall endpoint",
            definition_color,
        ),
        (
            "Analytic outputs",
            "Four-layer signal credibility\nassessment",
            output_color,
        ),
    ]
    main_y = 0.69
    main_h = 0.16
    main_w = 0.20
    main_xs = [0.045, 0.295, 0.545, 0.795]
    for x, (title, body, color) in zip(main_xs, main_boxes):
        add_box(x, main_y, main_w, main_h, title, body, color, title_size=10.6, body_size=8.4, title_y=0.74, body_y=0.39)
    for x1, x2 in zip(main_xs[:-1], main_xs[1:]):
        add_arrow(x1 + main_w, main_y + main_h / 2, x2, main_y + main_h / 2)

    ax.text(0.545, 0.62, "Exposure groups", fontsize=9.4, fontweight="bold", ha="left", va="top")
    ax.text(
        0.545,
        0.588,
        "Zolpidem; other Z-drugs; benzodiazepines;\n"
        "orexin receptor antagonists; other insomnia-related drugs",
        fontsize=8.2,
        ha="left",
        va="top",
        linespacing=1.25,
    )
    ax.text(
        0.545,
        0.555,
        f"Strict fall reports: n = {format_count(fall_n)} ({percent(fall_n, total_n):.2f}% of older-adult case reports)",
        fontsize=8.2,
        ha="left",
        va="top",
    )

    ax.text(0.04, 0.49, "Four-layer analytic output", fontsize=13.2, fontweight="bold", va="top")
    analysis_boxes = [
        (
            "1. Full-database\nsignal landscape",
            "Multiple disproportionality measures\nacross target drugs and groups",
        ),
        (
            "2. Active-comparator\nand adjusted analyses",
            "Clinically relevant comparator contrasts\nwith covariate-adjusted models",
        ),
        (
            "3. Sensitivity and\nreporting-structure analyses",
            "Primary-suspect-only, clean-exposure,\nand reporting-region checks",
        ),
        (
            "4. Phenotype and\nintegrated credibility",
            "Phenotype fingerprint, seriousness profile,\nand integrated signal interpretation",
        ),
    ]
    box_w = 0.215
    xs = [0.045, 0.285, 0.525, 0.765]
    for x, (title, body) in zip(xs, analysis_boxes):
        add_box(x, 0.28, box_w, 0.145, title, body, output_color, title_size=8.6, body_size=7.5, title_y=0.74, body_y=0.34)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        add_arrow(x1 + box_w, 0.352, x2, 0.352)
    add_arrow(0.895, main_y, 0.895, 0.425)

    add_box(
        0.18,
        0.085,
        0.64,
        0.105,
        "Interpretation boundary",
        "FAERS reporting patterns and signal prioritization, not incidence, absolute risk, or causal effect estimation",
        interpretation_color,
        title_size=10.0,
        body_size=8.4,
        title_y=0.66,
        body_y=0.34,
    )
    add_arrow(0.875, 0.28, 0.82, 0.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(figure_out, dpi=300, bbox_inches="tight")
    for suffix in [".svg", ".pdf", ".tiff"]:
        fig.savefig(figure_out.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--drug-master", type=Path, default=DEFAULT_DRUG_MASTER)
    parser.add_argument("--baseline-out", type=Path, default=DEFAULT_BASELINE_OUT)
    parser.add_argument("--exposure-out", type=Path, default=DEFAULT_EXPOSURE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--flow-figure-out", type=Path, default=DEFAULT_FLOW_FIGURE_OUT)
    args = parser.parse_args()

    if not args.main_dataset.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {args.main_dataset}")
    if not args.drug_master.exists():
        raise FileNotFoundError(f"Drug master not found: {args.drug_master}")

    drug_master = pd.read_csv(args.drug_master)
    targets = build_drug_targets(drug_master)
    exposure_columns = sorted({target.exposure_column for target in targets} | {column for _, column in FLOW_EXPOSURES})
    read_columns = BASELINE_COLUMNS + exposure_columns + ["serious_any"]
    df = read_available_columns(args.main_dataset, read_columns)

    baseline = build_baseline_table(df)
    exposure = build_exposure_table(df, targets)
    qc = build_qc(df, baseline, exposure)
    validate_results(df, baseline, exposure)

    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    args.exposure_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)

    baseline.to_csv(args.baseline_out, index=False, encoding="utf-8-sig")
    exposure.to_csv(args.exposure_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    plot_study_flow(df, args.flow_figure_out)

    print(f"Wrote {args.baseline_out}")
    print(f"Wrote {args.exposure_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.flow_figure_out}")
    print(f"Analysis eligible rows: {int(safe_bool(df['analysis_eligible_main']).sum()):,}")
    print(f"Strict fall rows: {int((safe_bool(df['analysis_eligible_main']) & safe_bool(df['strict_fall'])).sum()):,}")


if __name__ == "__main__":
    main()
