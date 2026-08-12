from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_3_signal_refinement_forest.png"

ACTIVE_COMPARATOR_TABLE = TABLE_DIR / "table_2_active_comparator_results.csv"
ADJUSTED_TABLE = TABLE_DIR / "table_3_adjusted_ror.csv"
PS_ONLY_TABLE = TABLE_DIR / "table_s3_ps_only_sensitivity.csv"
EXCLUDING_MIXED_TABLE = TABLE_DIR / "table_s4_excluding_mixed_exposure_sensitivity.csv"
REPORTING_SOURCE_TABLE = TABLE_DIR / "table_s5_reporting_source_stratified_sensitivity.csv"

MAIN_COMPARISONS = [
    ("zolpidem_vs_other_z_drugs", "Zolpidem vs other Z-drugs"),
    ("zolpidem_vs_benzodiazepines", "Zolpidem vs benzodiazepines"),
    ("zolpidem_vs_orexin_antagonists", "Zolpidem vs orexin antagonists"),
    ("zolpidem_vs_other_insomnia_related", "Zolpidem vs other insomnia-related drugs"),
    ("z_drugs_vs_benzodiazepines", "Z-drugs vs benzodiazepines"),
    ("z_drugs_vs_orexin_antagonists", "Z-drugs vs orexin antagonists"),
    ("benzodiazepines_vs_orexin_antagonists", "Benzodiazepines vs orexin antagonists"),
    ("other_insomnia_related_vs_orexin_antagonists", "Other insomnia-related drugs vs orexin antagonists"),
]

SENSITIVITY_ROWS = [
    ("ps_only", "overall", "overall", "zolpidem_vs_other_z_drugs", "PS-only: Zolpidem vs other Z-drugs"),
    ("ps_only", "overall", "overall", "zolpidem_vs_benzodiazepines", "PS-only: Zolpidem vs benzodiazepines"),
    ("ps_only", "overall", "overall", "zolpidem_vs_orexin_antagonists", "PS-only: Zolpidem vs orexin antagonists"),
    ("ps_only", "overall", "overall", "z_drugs_vs_benzodiazepines", "PS-only: Z-drugs vs benzodiazepines"),
    ("ps_only", "overall", "overall", "z_drugs_vs_orexin_antagonists", "PS-only: Z-drugs vs orexin antagonists"),
    ("reporting_source_stratified", "country_group", "US", "zolpidem_vs_other_z_drugs", "US: Zolpidem vs other Z-drugs"),
    (
        "reporting_source_stratified",
        "country_group",
        "non-US",
        "zolpidem_vs_other_z_drugs",
        "Non-US: Zolpidem vs other Z-drugs",
    ),
]

PANEL_COLORS = {
    "active": "#2f6f9f",
    "adjusted": "#2f7d5f",
    "sensitivity": "#7a688f",
    "context": "#c27a2c",
}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required table not found: {path}")
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: list[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def ensure_unique_row(df: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    rows = df.loc[mask].copy()
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for {description}, found {len(rows)}")
    return rows.iloc[0]


def format_estimate(value: float, low: float, high: float) -> str:
    return f"{value:.2f} ({low:.2f}-{high:.2f})"


def build_active_panel(active: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        active,
        ["comparison_id", "check_analysis_total_matches", "ROR", "ROR_95CI_low", "ROR_95CI_high"],
        ACTIVE_COMPARATOR_TABLE.name,
    )
    if not active["check_analysis_total_matches"].astype(bool).all():
        failed = active.loc[~active["check_analysis_total_matches"].astype(bool), "comparison_id"].tolist()
        raise ValueError(f"Active comparator total checks failed: {failed}")

    rows = []
    for order, (comparison_id, label) in enumerate(MAIN_COMPARISONS):
        row = ensure_unique_row(active, active["comparison_id"].eq(comparison_id), f"active comparator {comparison_id}")
        rows.append(
            {
                "order": order,
                "label": label,
                "estimate": float(row["ROR"]),
                "low": float(row["ROR_95CI_low"]),
                "high": float(row["ROR_95CI_high"]),
                "estimate_text": format_estimate(float(row["ROR"]), float(row["ROR_95CI_low"]), float(row["ROR_95CI_high"])),
                "color": PANEL_COLORS["active"],
            }
        )
    return pd.DataFrame(rows)


def build_adjusted_panel(adjusted: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        adjusted,
        ["comparison_id", "model_id", "fit_status", "OR", "OR_95CI_low", "OR_95CI_high"],
        ADJUSTED_TABLE.name,
    )
    model3 = adjusted[adjusted["model_id"].eq("model_3_full")].copy()
    if model3.empty:
        raise ValueError("No model_3_full rows found in adjusted model table.")
    failed = model3[~model3["fit_status"].astype(str).eq("ok")]
    if not failed.empty:
        failed_rows = failed[["comparison_id", "fit_status"]].to_dict("records")
        raise ValueError(f"Non-ok model_3_full rows found: {failed_rows}")

    rows = []
    for order, (comparison_id, label) in enumerate(MAIN_COMPARISONS):
        row = ensure_unique_row(model3, model3["comparison_id"].eq(comparison_id), f"model_3_full {comparison_id}")
        rows.append(
            {
                "order": order,
                "label": label,
                "estimate": float(row["OR"]),
                "low": float(row["OR_95CI_low"]),
                "high": float(row["OR_95CI_high"]),
                "estimate_text": format_estimate(float(row["OR"]), float(row["OR_95CI_low"]), float(row["OR_95CI_high"])),
                "color": PANEL_COLORS["adjusted"],
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity_panel(ps_only: pd.DataFrame, reporting_source: pd.DataFrame) -> pd.DataFrame:
    for df, name in [(ps_only, PS_ONLY_TABLE.name), (reporting_source, REPORTING_SOURCE_TABLE.name)]:
        require_columns(
            df,
            [
                "analysis_type",
                "stratum_variable",
                "stratum_value",
                "comparison_id",
                "check_analysis_total_matches",
                "ROR",
                "ROR_95CI_low",
                "ROR_95CI_high",
            ],
            name,
        )
        if not df["check_analysis_total_matches"].astype(bool).all():
            failed = df.loc[~df["check_analysis_total_matches"].astype(bool), ["analysis_type", "comparison_id"]].to_dict("records")
            raise ValueError(f"Sensitivity total checks failed in {name}: {failed}")

    rows = []
    for order, (analysis_type, stratum_variable, stratum_value, comparison_id, label) in enumerate(SENSITIVITY_ROWS):
        source = ps_only if analysis_type == "ps_only" else reporting_source
        mask = (
            source["analysis_type"].eq(analysis_type)
            & source["stratum_variable"].eq(stratum_variable)
            & source["stratum_value"].eq(stratum_value)
            & source["comparison_id"].eq(comparison_id)
        )
        row = ensure_unique_row(source, mask, f"{analysis_type}/{stratum_value}/{comparison_id}")
        color = PANEL_COLORS["context"] if analysis_type == "reporting_source_stratified" else PANEL_COLORS["sensitivity"]
        rows.append(
            {
                "order": order,
                "label": label,
                "estimate": float(row["ROR"]),
                "low": float(row["ROR_95CI_low"]),
                "high": float(row["ROR_95CI_high"]),
                "estimate_text": format_estimate(float(row["ROR"]), float(row["ROR_95CI_low"]), float(row["ROR_95CI_high"])),
                "color": color,
            }
        )
    return pd.DataFrame(rows)


def add_forest_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    title: str,
    estimate_header: str,
    x_min: float,
    x_max: float,
    show_xlabel: bool,
) -> None:
    panel_df = panel_df.sort_values("order", ascending=True).reset_index(drop=True)
    y = np.arange(len(panel_df))[::-1]
    estimate = panel_df["estimate"].to_numpy(dtype=float)
    low = panel_df["low"].to_numpy(dtype=float)
    high = panel_df["high"].to_numpy(dtype=float)
    clipped_low = np.maximum(low, x_min)
    clipped_high = np.minimum(high, x_max)
    xerr = np.vstack([estimate - clipped_low, clipped_high - estimate])

    for idx, (x_value, yy, err_low, err_high, color) in enumerate(zip(estimate, y, xerr[0], xerr[1], panel_df["color"])):
        ax.errorbar(
            x_value,
            yy,
            xerr=np.array([[err_low], [err_high]]),
            fmt="o",
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=color,
            ecolor="#444444",
            elinewidth=0.9,
            capsize=2.4,
            zorder=3,
        )
        if low[idx] < x_min:
            ax.annotate("", xy=(x_min, yy), xytext=(x_min * 1.12, yy), arrowprops=dict(arrowstyle="<|-", color="#444444", lw=0.8))
        if high[idx] > x_max:
            ax.annotate("", xy=(x_max, yy), xytext=(x_max / 1.12, yy), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=0.8))

    ax.axvline(1, color="#888888", linestyle="--", linewidth=0.85, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.75, len(panel_df) - 0.25)
    ax.set_yticks(y)
    ax.set_yticklabels(panel_df["label"])
    ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold", pad=4)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.35)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Estimate (log scale)" if show_xlabel else "", fontsize=7.0)

    ax.text(1.03, 1.02, estimate_header, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.7, fontweight="bold")
    for row, yy in zip(panel_df.itertuples(), y):
        ax.text(1.03, yy, row.estimate_text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.6)


def plot_figure(active_panel: pd.DataFrame, adjusted_panel: pd.DataFrame, sensitivity_panel: pd.DataFrame, figure_out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8.6, 8.7),
        gridspec_kw={"height_ratios": [1.25, 1.25, 1.1], "hspace": 0.43},
    )
    x_min, x_max = 0.52, 15.5
    add_forest_panel(axes[0], active_panel, "A  Active-comparator restricted ROR", "ROR (95% CI)", x_min, x_max, False)
    add_forest_panel(axes[1], adjusted_panel, "B  Fully adjusted active-comparator models", "OR (95% CI)", x_min, x_max, False)
    add_forest_panel(axes[2], sensitivity_panel, "C  Sensitivity and reporting-context checks", "ROR (95% CI)", x_min, x_max, True)

    fig.suptitle("Active-comparator, adjusted, and sensitivity signal refinement", fontsize=9.3, fontweight="bold", x=0.48, y=0.995)
    fig.text(
        0.014,
        0.01,
        "Point estimates are report-based RORs or adjusted ORs; horizontal bars are 95% CIs. Estimates describe FAERS reporting signals, not incidence or causal effects.",
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.30, right=0.70, top=0.94, bottom=0.08)
    fig.savefig(figure_out, dpi=300, bbox_inches="tight")
    for suffix in [".svg", ".pdf", ".tiff"]:
        fig.savefig(figure_out.with_suffix(suffix), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    active = read_required_csv(ACTIVE_COMPARATOR_TABLE)
    adjusted = read_required_csv(ADJUSTED_TABLE)
    ps_only = read_required_csv(PS_ONLY_TABLE)
    _ = read_required_csv(EXCLUDING_MIXED_TABLE)
    reporting_source = read_required_csv(REPORTING_SOURCE_TABLE)

    active_panel = build_active_panel(active)
    adjusted_panel = build_adjusted_panel(adjusted)
    sensitivity_panel = build_sensitivity_panel(ps_only, reporting_source)
    plot_figure(active_panel, adjusted_panel, sensitivity_panel, FIGURE_OUT)

    print(f"Wrote {FIGURE_OUT}")
    print(f"Active-comparator rows plotted: {len(active_panel)}")
    print(f"Adjusted model rows plotted: {len(adjusted_panel)}")
    print(f"Sensitivity/context rows plotted: {len(sensitivity_panel)}")


if __name__ == "__main__":
    main()
