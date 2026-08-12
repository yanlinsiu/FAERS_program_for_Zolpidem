from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_DRUG_MASTER = PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"
DEFAULT_TABLE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_1_signal_landscape.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "06_signal_landscape_qc.csv"
DEFAULT_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_2_signal_landscape_forest.png"

GROUP_KEYS = [
    "z_drug",
    "other_z_drug",
    "benzodiazepine",
    "orexin_antagonist",
    "other_insomnia_related",
]

CLASS_PANEL_TARGETS = [
    ("z_drug", "Z-drugs"),
    ("benzodiazepine", "Benzodiazepines"),
    ("other_insomnia_related", "Other insomnia-related\nmedications"),
    ("orexin_antagonist", "Orexin receptor\nantagonists"),
]
DRUG_PANEL_TARGETS = [
    ("zopiclone", "Zopiclone"),
    ("zolpidem", "Zolpidem"),
    ("lorazepam", "Lorazepam"),
    ("mirtazapine", "Mirtazapine"),
    ("diazepam", "Diazepam"),
    ("alprazolam", "Alprazolam"),
    ("clonazepam", "Clonazepam"),
    ("trazodone", "Trazodone"),
    ("suvorexant", "Suvorexant"),
    ("eszopiclone", "Eszopiclone"),
    ("zaleplon", "Zaleplon"),
]
GROUP_COLOR_MAP = {
    "z_drug": "#2f6f9f",
    "benzodiazepine": "#2f7d5f",
    "orexin_antagonist": "#c27a2c",
    "other_insomnia_related": "#7a688f",
}
REQUIRED_BASE_COLUMNS = [
    "analysis_eligible_main",
    "strict_fall",
    "serious_any",
    "serious_death",
    "serious_hospitalization",
    "serious_disability",
    "serious_life_threatening",
]


@dataclass(frozen=True)
class AnalysisTarget:
    analysis_level: str
    target_key: str
    target_label: str
    drug_group: str
    exposure_column: str


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def build_targets(drug_master: pd.DataFrame) -> list[AnalysisTarget]:
    targets = []
    candidates = drug_master["main_analysis_candidate"].astype(str).str.lower().isin(["yes", "count_dependent"])
    for _, row in drug_master.loc[candidates].iterrows():
        drug_key = str(row["drug_key"])
        targets.append(
            AnalysisTarget(
                analysis_level="drug",
                target_key=drug_key,
                target_label=str(row["generic_name"]),
                drug_group=str(row["drug_group"]),
                exposure_column=f"exposure_{drug_key}_ps_ss",
            )
        )

    for group_key in GROUP_KEYS:
        targets.append(
            AnalysisTarget(
                analysis_level="group",
                target_key=group_key,
                target_label=group_key,
                drug_group=group_key,
                exposure_column=f"exposure_{group_key}_ps_ss",
            )
        )
    return targets


def read_inputs(main_dataset_path: Path, drug_master_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[AnalysisTarget]]:
    if not main_dataset_path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {main_dataset_path}")
    if not drug_master_path.exists():
        raise FileNotFoundError(f"Drug master not found: {drug_master_path}")

    drug_master = pd.read_csv(drug_master_path)
    targets = build_targets(drug_master)
    exposure_columns = sorted({target.exposure_column for target in targets})
    requested_columns = REQUIRED_BASE_COLUMNS + exposure_columns
    available_columns = pq.ParquetFile(main_dataset_path).schema.names
    read_columns = [column for column in requested_columns if column in available_columns]
    missing_base = [column for column in REQUIRED_BASE_COLUMNS if column not in available_columns]
    if missing_base:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing_base}")

    df = pd.read_parquet(main_dataset_path, columns=read_columns)
    return df, drug_master, targets


def corrected_cells(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float, bool]:
    has_zero_cell = min(a, b, c, d) == 0
    if has_zero_cell:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5, True
    return float(a), float(b), float(c), float(d), False


def calculate_metrics(a: int, b: int, c: int, d: int) -> dict[str, float | bool]:
    ac, bc, cc, dc, continuity_correction = corrected_cells(a, b, c, d)
    total = ac + bc + cc + dc
    exposed_total = ac + bc
    unexposed_total = cc + dc
    fall_total = ac + cc

    ror = (ac * dc) / (bc * cc)
    ror_se = math.sqrt((1 / ac) + (1 / bc) + (1 / cc) + (1 / dc))
    ror_low = math.exp(math.log(ror) - 1.96 * ror_se)
    ror_high = math.exp(math.log(ror) + 1.96 * ror_se)

    prr = (ac / exposed_total) / (cc / unexposed_total)
    prr_se = math.sqrt((1 / ac) - (1 / exposed_total) + (1 / cc) - (1 / unexposed_total))
    prr_low = math.exp(math.log(prr) - 1.96 * prr_se)
    prr_high = math.exp(math.log(prr) + 1.96 * prr_se)

    expected = exposed_total * fall_total / total
    ic = math.log2(ac / expected)
    ic_se = 1 / (math.log(2) * math.sqrt(ac))
    ic025 = ic - 1.96 * ic_se

    oe = ac / expected
    eb_se = 1 / math.sqrt(ac)
    oe05 = math.exp(math.log(oe) - 1.645 * eb_se)

    return {
        "ROR": ror,
        "ROR_95CI_low": ror_low,
        "ROR_95CI_high": ror_high,
        "PRR": prr,
        "PRR_95CI_low": prr_low,
        "PRR_95CI_high": prr_high,
        "IC": ic,
        "IC025": ic025,
        "OE": oe,
        "OE05": oe05,
        "continuity_correction": continuity_correction,
    }


def count_fall_serious_outcomes(exposure: pd.Series, outcome: pd.Series, df: pd.DataFrame, fall_n: int) -> dict[str, float | int]:
    fall_exposed = exposure & outcome
    serious_columns = {
        "serious_any": "serious_fall",
        "serious_death": "death_fall",
        "serious_hospitalization": "hospitalization_fall",
        "serious_disability": "disability_fall",
        "serious_life_threatening": "life_threatening_fall",
    }

    results: dict[str, float | int] = {}
    for source_column, output_prefix in serious_columns.items():
        serious = safe_bool(df[source_column])
        count = int((fall_exposed & serious).sum())
        percent = (count / fall_n * 100) if fall_n else np.nan
        results[f"{output_prefix}_n"] = count
        results[f"{output_prefix}_percent"] = percent
    return results


def analyze_target(df: pd.DataFrame, target: AnalysisTarget, analysis_n: int, strict_fall_n: int) -> dict[str, object]:
    if target.exposure_column not in df.columns:
        return {
            "analysis_level": target.analysis_level,
            "target_key": target.target_key,
            "target_label": target.target_label,
            "drug_group": target.drug_group,
            "exposure_column": target.exposure_column,
            "column_present": False,
            "note": "Exposure column not found in main analysis dataset.",
        }

    exposure = safe_bool(df[target.exposure_column])
    outcome = safe_bool(df["strict_fall"])
    a = int((exposure & outcome).sum())
    b = int((exposure & ~outcome).sum())
    c = int((~exposure & outcome).sum())
    d = int((~exposure & ~outcome).sum())
    exposed_n = a + b
    fall_percent = (a / exposed_n * 100) if exposed_n else np.nan
    metrics = calculate_metrics(a, b, c, d)
    serious_outcomes = count_fall_serious_outcomes(exposure, outcome, df, a)

    row = {
        "analysis_level": target.analysis_level,
        "target_key": target.target_key,
        "target_label": target.target_label,
        "drug_group": target.drug_group,
        "exposure_column": target.exposure_column,
        "column_present": True,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "exposed_n": exposed_n,
        "fall_n": a,
        "fall_percent": fall_percent,
        "signal_positive_ror": metrics["ROR_95CI_low"] > 1,
        "enough_cases_main": exposed_n >= 50 and a >= 5,
        "preferred_for_main_text": exposed_n >= 50 and a >= 10,
        "check_total_matches": (a + b + c + d) == analysis_n,
        "check_fall_total_matches": (a + c) == strict_fall_n,
        "check_exposed_total_matches": (a + b) == exposed_n,
        "note": "",
    }
    row.update(metrics)
    row.update(serious_outcomes)
    return row


def build_signal_landscape(df: pd.DataFrame, targets: list[AnalysisTarget]) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_df = df.loc[eligible].copy()
    analysis_df["strict_fall"] = safe_bool(analysis_df["strict_fall"])
    analysis_n = len(analysis_df)
    strict_fall_n = int(analysis_df["strict_fall"].sum())
    rows = [analyze_target(analysis_df, target, analysis_n, strict_fall_n) for target in targets]
    results = pd.DataFrame(rows)

    metric_columns = ["ROR", "ROR_95CI_low", "ROR_95CI_high", "PRR", "PRR_95CI_low", "PRR_95CI_high", "IC", "IC025", "OE", "OE05"]
    for column in metric_columns:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")

    return results.sort_values(["analysis_level", "drug_group", "ROR"], ascending=[True, True, False])


def build_qc(input_rows: int, analysis_rows: int, strict_fall_n: int, results: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "qc_domain": "signal_landscape",
            "analysis_level": "overall",
            "target_key": "overall",
            "metric": "input_rows",
            "value": input_rows,
            "note": "",
        },
        {
            "qc_domain": "signal_landscape",
            "analysis_level": "overall",
            "target_key": "overall",
            "metric": "analysis_eligible_rows",
            "value": analysis_rows,
            "note": "",
        },
        {
            "qc_domain": "signal_landscape",
            "analysis_level": "overall",
            "target_key": "overall",
            "metric": "strict_fall_rows",
            "value": strict_fall_n,
            "note": "",
        },
    ]

    for _, row in results.iterrows():
        missing_note = "" if bool(row["column_present"]) else "Exposure column missing."
        for metric in [
            "column_present",
            "a",
            "b",
            "c",
            "d",
            "exposed_n",
            "fall_n",
            "serious_fall_n",
            "serious_fall_percent",
            "death_fall_n",
            "death_fall_percent",
            "hospitalization_fall_n",
            "hospitalization_fall_percent",
            "disability_fall_n",
            "disability_fall_percent",
            "life_threatening_fall_n",
            "life_threatening_fall_percent",
            "enough_cases_main",
            "preferred_for_main_text",
            "continuity_correction",
        ]:
            rows.append(
                {
                    "qc_domain": "signal_landscape",
                    "analysis_level": row["analysis_level"],
                    "target_key": row["target_key"],
                    "metric": metric,
                    "value": row.get(metric, pd.NA),
                    "note": missing_note,
                }
            )
    return pd.DataFrame(rows)


def validate_results(results: pd.DataFrame) -> None:
    present = results[results["column_present"] == True].copy()
    required_checks = ["check_total_matches", "check_fall_total_matches", "check_exposed_total_matches"]
    for check in required_checks:
        if not present[check].all():
            failed = present.loc[~present[check], ["analysis_level", "target_key"]].to_dict("records")
            raise ValueError(f"Signal landscape validation failed for {check}: {failed}")

    metric_columns = ["ROR", "ROR_95CI_low", "ROR_95CI_high", "PRR", "PRR_95CI_low", "PRR_95CI_high", "IC", "IC025", "OE", "OE05"]
    values = present[metric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Signal landscape contains non-finite values in core metrics.")

    zolpidem = present[present["target_key"] == "zolpidem"]
    if zolpidem.empty:
        raise ValueError("Zolpidem row is missing from signal landscape output.")
    zolpidem_a = int(zolpidem.iloc[0]["a"])
    if abs(zolpidem_a - 986) > 2:
        raise ValueError(f"Zolpidem strict fall count differs from expected QC value: {zolpidem_a}")


def format_count(value: int | float) -> str:
    return f"{int(value):,}"


def format_ci(row: pd.Series) -> str:
    return f"{row['ROR']:.2f} ({row['ROR_95CI_low']:.2f}-{row['ROR_95CI_high']:.2f})"


def figure_table_text(row: pd.Series) -> tuple[str, str]:
    return (
        f"{format_count(row['fall_n'])} / {format_count(row['exposed_n'])}",
        f"{row['fall_percent']:.2f}%",
    )


def get_panel_df(results: pd.DataFrame, targets: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    present = results[results["column_present"] == True].copy()
    for order, (target_key, display_label) in enumerate(targets):
        match = present[present["target_key"] == target_key]
        if match.empty:
            raise ValueError(f"Missing signal-landscape row for figure target: {target_key}")
        row = match.iloc[0].copy()
        row["display_label"] = display_label
        row["display_order"] = order
        rows.append(row)
    return pd.DataFrame(rows).sort_values("display_order", ascending=True).reset_index(drop=True)


def add_forest_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    panel_title: str,
    x_min: float,
    x_max: float,
    show_xlabel: bool,
) -> None:
    y = np.arange(len(panel_df))[::-1]
    ror = panel_df["ROR"].to_numpy(dtype=float)
    low = panel_df["ROR_95CI_low"].to_numpy(dtype=float)
    high = panel_df["ROR_95CI_high"].to_numpy(dtype=float)

    clipped_low = np.maximum(low, x_min)
    clipped_high = np.minimum(high, x_max)
    xerr = np.vstack([ror - clipped_low, clipped_high - ror])
    colors = [GROUP_COLOR_MAP.get(str(group), "#555555") for group in panel_df["drug_group"]]

    for idx, (x_value, yy, err_low, err_high, color) in enumerate(zip(ror, y, xerr[0], xerr[1], colors)):
        ax.errorbar(
            x_value,
            yy,
            xerr=np.array([[err_low], [err_high]]),
            fmt="o",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor=color,
            ecolor="#444444",
            elinewidth=1.0,
            capsize=2.8,
            zorder=3,
        )
        if low[idx] < x_min:
            ax.annotate("", xy=(x_min, yy), xytext=(x_min * 1.11, yy), arrowprops=dict(arrowstyle="<|-", color="#444444", lw=0.9))
        if high[idx] > x_max:
            ax.annotate("", xy=(x_max, yy), xytext=(x_max / 1.11, yy), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.9))

    ax.axvline(1, color="#8a8a8a", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.75, len(panel_df) - 0.25)
    ax.set_yticks(y)
    ax.set_yticklabels(panel_df["display_label"])
    ax.set_title(panel_title, loc="left", fontsize=8.4, fontweight="bold", pad=6)
    ax.grid(axis="x", linestyle=":", linewidth=0.55, alpha=0.35)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("Reporting odds ratio (ROR, log scale)", fontsize=7.4)
    else:
        ax.set_xlabel("")

    x_text_ror = 1.04
    x_text_count = 1.43
    x_text_prop = 1.76
    ax.text(x_text_ror, 1.025, "ROR (95% CI)", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, fontweight="bold")
    ax.text(x_text_count, 1.025, "Strict fall /\nexposed", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, fontweight="bold", linespacing=1.05)
    ax.text(x_text_prop, 1.025, "Strict-fall\nreporting %", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, fontweight="bold", linespacing=1.05)

    for row, yy in zip(panel_df.itertuples(), y):
        row_series = panel_df.loc[row.Index]
        count_text, prop_text = figure_table_text(row_series)
        ax.text(x_text_ror, yy, format_ci(row_series), transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8)
        ax.text(x_text_count, yy, count_text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8)
        ax.text(x_text_prop, yy, prop_text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8)


def plot_forest(results: pd.DataFrame, figure_out: Path) -> None:
    class_df = get_panel_df(results, CLASS_PANEL_TARGETS)
    drug_df = get_panel_df(results, DRUG_PANEL_TARGETS)

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.7, 6.7),
        gridspec_kw={"height_ratios": [1.0, 2.15], "hspace": 0.42},
    )
    x_min, x_max = 0.28, 7.4
    add_forest_panel(axes[0], class_df, "A  Class-level strict-fall reporting signals", x_min, x_max, show_xlabel=False)
    add_forest_panel(axes[1], drug_df, "B  Selected drug-level strict-fall reporting signals", x_min, x_max, show_xlabel=True)
    fig.suptitle("Full-database strict-fall signal landscape", fontsize=9.5, fontweight="bold", x=0.49, y=0.995)
    fig.text(
        0.015,
        0.01,
        "Point estimates are RORs; horizontal bars are 95% CIs. Values are reporting proportions in FAERS, not incidence or causal risk.",
        fontsize=6.7,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.22, right=0.58, top=0.93, bottom=0.09)
    fig.savefig(figure_out, dpi=300, bbox_inches="tight")
    for suffix in [".svg", ".pdf", ".tiff"]:
        fig.savefig(figure_out.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--drug-master", type=Path, default=DEFAULT_DRUG_MASTER)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    args = parser.parse_args()

    df, _, targets = read_inputs(args.main_dataset, args.drug_master)
    results = build_signal_landscape(df, targets)
    eligible = safe_bool(df["analysis_eligible_main"])
    analysis_rows = int(eligible.sum())
    strict_fall_n = int(safe_bool(df.loc[eligible, "strict_fall"]).sum())
    qc = build_qc(len(df), analysis_rows, strict_fall_n, results)

    validate_results(results)

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    plot_forest(results, args.figure_out)

    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.figure_out}")
    print(f"Analysis eligible rows: {analysis_rows:,}")
    print(f"Strict fall rows: {strict_fall_n:,}")
    print(f"Targets analyzed: {int(results['column_present'].sum()):,}")


if __name__ == "__main__":
    main()
