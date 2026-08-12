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
DEFAULT_TABLE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_2_active_comparator_results.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "07_active_comparator_qc.csv"
DEFAULT_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_2_active_comparator_forest.png"

BASE_COLUMNS = [
    "analysis_eligible_main",
    "strict_fall",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
]

DRUG_KEYS = [
    "zolpidem",
    "eszopiclone",
    "zaleplon",
    "zopiclone",
    "temazepam",
    "triazolam",
    "lorazepam",
    "diazepam",
    "alprazolam",
    "clonazepam",
    "suvorexant",
    "lemborexant",
    "daridorexant",
    "trazodone",
    "mirtazapine",
    "doxepin",
    "ramelteon",
]
GROUP_KEYS = ["z_drug", "other_z_drug", "benzodiazepine", "orexin_antagonist", "other_insomnia_related"]


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    tier: str
    exposure_label: str
    comparator_label: str
    exposure_mask: str
    comparator_mask: str
    research_question: str


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def exposure_column(key: str) -> str:
    return f"exposure_{key}_ps_ss"


def read_main_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")

    required_columns = BASE_COLUMNS + [exposure_column(key) for key in DRUG_KEYS + GROUP_KEYS]
    available = pq.ParquetFile(path).schema.names
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing}")
    return pd.read_parquet(path, columns=required_columns)


def build_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    eligible = safe_bool(df["analysis_eligible_main"])
    one_group = pd.to_numeric(df["n_sedative_hypnotic_groups_ps_ss"], errors="coerce").fillna(0).eq(1)
    one_drug = pd.to_numeric(df["n_sedative_hypnotic_drugs_ps_ss"], errors="coerce").fillna(0).eq(1)

    masks: dict[str, pd.Series] = {"eligible": eligible}

    for group_key in GROUP_KEYS:
        masks[f"{group_key}_only"] = eligible & safe_bool(df[exposure_column(group_key)]) & one_group

    for drug_key in DRUG_KEYS:
        masks[f"{drug_key}_only"] = eligible & safe_bool(df[exposure_column(drug_key)]) & one_drug

    masks["other_z_drugs_only"] = masks["other_z_drug_only"]
    masks["benzodiazepines_only"] = masks["benzodiazepine_only"]
    masks["orexin_antagonists_only"] = masks["orexin_antagonist_only"]
    masks["other_insomnia_related_only"] = masks["other_insomnia_related_only"]

    masks["other_z_drugs_without_zolpidem_only"] = (
        masks["z_drug_only"]
        & ~safe_bool(df[exposure_column("zolpidem")])
    )
    masks["benzodiazepines_without_lorazepam_only"] = (
        masks["benzodiazepine_only"]
        & ~safe_bool(df[exposure_column("lorazepam")])
    )
    masks["other_z_drugs_without_zopiclone_only"] = (
        masks["z_drug_only"]
        & ~safe_bool(df[exposure_column("zopiclone")])
    )
    return masks


def build_comparisons() -> list[ComparisonSpec]:
    return [
        ComparisonSpec(
            "zolpidem_vs_other_z_drugs",
            "zolpidem_centered",
            "zolpidem-only",
            "other Z-drugs-only",
            "zolpidem_only",
            "other_z_drugs_without_zolpidem_only",
            "Zolpidem is more fall-disproportionate than other Z-drugs.",
        ),
        ComparisonSpec(
            "zolpidem_vs_benzodiazepines",
            "zolpidem_centered",
            "zolpidem-only",
            "benzodiazepines-only",
            "zolpidem_only",
            "benzodiazepines_only",
            "Zolpidem is more fall-disproportionate than benzodiazepines.",
        ),
        ComparisonSpec(
            "zolpidem_vs_orexin_antagonists",
            "zolpidem_centered",
            "zolpidem-only",
            "orexin antagonists-only",
            "zolpidem_only",
            "orexin_antagonists_only",
            "Zolpidem is more fall-disproportionate than orexin receptor antagonists.",
        ),
        ComparisonSpec(
            "zolpidem_vs_other_insomnia_related",
            "zolpidem_centered",
            "zolpidem-only",
            "other insomnia-related drugs-only",
            "zolpidem_only",
            "other_insomnia_related_only",
            "Zolpidem is more fall-disproportionate than other insomnia-related drugs.",
        ),
        ComparisonSpec(
            "z_drugs_vs_benzodiazepines",
            "class_comparison",
            "Z-drugs-only",
            "benzodiazepines-only",
            "z_drug_only",
            "benzodiazepines_only",
            "Z-drugs are more fall-disproportionate than benzodiazepines.",
        ),
        ComparisonSpec(
            "z_drugs_vs_orexin_antagonists",
            "class_comparison",
            "Z-drugs-only",
            "orexin antagonists-only",
            "z_drug_only",
            "orexin_antagonists_only",
            "Z-drugs are more fall-disproportionate than orexin receptor antagonists.",
        ),
        ComparisonSpec(
            "benzodiazepines_vs_orexin_antagonists",
            "class_comparison",
            "benzodiazepines-only",
            "orexin antagonists-only",
            "benzodiazepines_only",
            "orexin_antagonists_only",
            "Benzodiazepines are more fall-disproportionate than orexin receptor antagonists.",
        ),
        ComparisonSpec(
            "other_insomnia_related_vs_orexin_antagonists",
            "class_comparison",
            "other insomnia-related drugs-only",
            "orexin antagonists-only",
            "other_insomnia_related_only",
            "orexin_antagonists_only",
            "Other insomnia-related drugs are more fall-disproportionate than orexin receptor antagonists.",
        ),
        ComparisonSpec(
            "zolpidem_vs_eszopiclone",
            "within_class_supplement",
            "zolpidem-only",
            "eszopiclone-only",
            "zolpidem_only",
            "eszopiclone_only",
            "Zolpidem is more fall-disproportionate than eszopiclone.",
        ),
        ComparisonSpec(
            "zolpidem_vs_zopiclone",
            "within_class_supplement",
            "zolpidem-only",
            "zopiclone-only",
            "zolpidem_only",
            "zopiclone_only",
            "Zolpidem is more fall-disproportionate than zopiclone.",
        ),
        ComparisonSpec(
            "zolpidem_vs_zaleplon",
            "within_class_supplement",
            "zolpidem-only",
            "zaleplon-only",
            "zolpidem_only",
            "zaleplon_only",
            "Zolpidem is more fall-disproportionate than zaleplon.",
        ),
        ComparisonSpec(
            "lorazepam_vs_other_benzodiazepines",
            "within_class_supplement",
            "lorazepam-only",
            "other benzodiazepines-only",
            "lorazepam_only",
            "benzodiazepines_without_lorazepam_only",
            "Lorazepam is more fall-disproportionate than other benzodiazepines.",
        ),
        ComparisonSpec(
            "zopiclone_vs_other_z_drugs",
            "within_class_supplement",
            "zopiclone-only",
            "other Z-drugs-only",
            "zopiclone_only",
            "other_z_drugs_without_zopiclone_only",
            "Zopiclone is more fall-disproportionate than other Z-drugs.",
        ),
    ]


def corrected_cells(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float, bool]:
    if min(a, b, c, d) == 0:
        return a + 0.5, b + 0.5, c + 0.5, d + 0.5, True
    return float(a), float(b), float(c), float(d), False


def calculate_metrics(a: int, b: int, c: int, d: int) -> dict[str, float | bool]:
    ac, bc, cc, dc, continuity_correction = corrected_cells(a, b, c, d)
    ror = (ac * dc) / (bc * cc)
    ror_se = math.sqrt((1 / ac) + (1 / bc) + (1 / cc) + (1 / dc))
    ror_low = math.exp(math.log(ror) - 1.96 * ror_se)
    ror_high = math.exp(math.log(ror) + 1.96 * ror_se)

    exposed_total = ac + bc
    comparator_total = cc + dc
    prr = (ac / exposed_total) / (cc / comparator_total)
    prr_se = math.sqrt((1 / ac) - (1 / exposed_total) + (1 / cc) - (1 / comparator_total))
    prr_low = math.exp(math.log(prr) - 1.96 * prr_se)
    prr_high = math.exp(math.log(prr) + 1.96 * prr_se)

    return {
        "ROR": ror,
        "ROR_95CI_low": ror_low,
        "ROR_95CI_high": ror_high,
        "PRR": prr,
        "PRR_95CI_low": prr_low,
        "PRR_95CI_high": prr_high,
        "continuity_correction": continuity_correction,
    }


def interpretation(ror_low: float, ror_high: float) -> str:
    if ror_low > 1:
        return "exposure_higher"
    if ror_high < 1:
        return "exposure_lower"
    return "not_clearly_different"


def analyze_comparison(df: pd.DataFrame, masks: dict[str, pd.Series], spec: ComparisonSpec) -> dict[str, object]:
    exposure = masks[spec.exposure_mask]
    comparator = masks[spec.comparator_mask]
    overlap_n = int((exposure & comparator).sum())
    if overlap_n:
        raise ValueError(f"Comparison masks overlap for {spec.comparison_id}: {overlap_n}")

    outcome = safe_bool(df["strict_fall"])
    analysis_mask = exposure | comparator
    a = int((exposure & outcome).sum())
    b = int((exposure & ~outcome).sum())
    c = int((comparator & outcome).sum())
    d = int((comparator & ~outcome).sum())
    exposure_n = a + b
    comparator_n = c + d

    metrics = calculate_metrics(a, b, c, d)
    row = {
        "comparison_id": spec.comparison_id,
        "tier": spec.tier,
        "exposure_group": spec.exposure_label,
        "comparator_group": spec.comparator_label,
        "exposure_mask": spec.exposure_mask,
        "comparator_mask": spec.comparator_mask,
        "research_question": spec.research_question,
        "analysis_n": int(analysis_mask.sum()),
        "exposure_n": exposure_n,
        "exposure_fall_n": a,
        "exposure_nonfall_n": b,
        "exposure_fall_percent": (a / exposure_n * 100) if exposure_n else np.nan,
        "comparator_n": comparator_n,
        "comparator_fall_n": c,
        "comparator_nonfall_n": d,
        "comparator_fall_percent": (c / comparator_n * 100) if comparator_n else np.nan,
        "enough_cases": exposure_n >= 50 and comparator_n >= 50 and a >= 5 and c >= 5,
        "preferred_for_main_text": spec.tier != "within_class_supplement" and exposure_n >= 50 and comparator_n >= 50 and a >= 10 and c >= 10,
        "check_analysis_total_matches": int(analysis_mask.sum()) == exposure_n + comparator_n,
    }
    row.update(metrics)
    row["direction"] = interpretation(float(row["ROR_95CI_low"]), float(row["ROR_95CI_high"]))
    return row


def build_active_comparator_results(df: pd.DataFrame) -> pd.DataFrame:
    masks = build_masks(df)
    rows = [analyze_comparison(df, masks, spec) for spec in build_comparisons()]
    return pd.DataFrame(rows)


def build_qc(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "qc_domain": "active_comparator",
            "comparison_id": "overall",
            "metric": "input_rows",
            "value": len(df),
            "note": "",
        },
        {
            "qc_domain": "active_comparator",
            "comparison_id": "overall",
            "metric": "analysis_eligible_rows",
            "value": int(safe_bool(df["analysis_eligible_main"]).sum()),
            "note": "",
        },
        {
            "qc_domain": "active_comparator",
            "comparison_id": "overall",
            "metric": "strict_fall_rows",
            "value": int(safe_bool(df["strict_fall"]).sum()),
            "note": "",
        },
    ]
    for _, row in results.iterrows():
        for metric in [
            "analysis_n",
            "exposure_n",
            "exposure_fall_n",
            "exposure_fall_percent",
            "comparator_n",
            "comparator_fall_n",
            "comparator_fall_percent",
            "enough_cases",
            "preferred_for_main_text",
            "continuity_correction",
            "check_analysis_total_matches",
            "direction",
        ]:
            rows.append(
                {
                    "qc_domain": "active_comparator",
                    "comparison_id": row["comparison_id"],
                    "metric": metric,
                    "value": row[metric],
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def validate_results(results: pd.DataFrame) -> None:
    if not results["check_analysis_total_matches"].all():
        failed = results.loc[~results["check_analysis_total_matches"], "comparison_id"].tolist()
        raise ValueError(f"Active comparator analysis total check failed: {failed}")
    metric_columns = ["ROR", "ROR_95CI_low", "ROR_95CI_high", "PRR", "PRR_95CI_low", "PRR_95CI_high"]
    if not np.isfinite(results[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Active comparator results contain non-finite metrics.")


def plot_forest(results: pd.DataFrame, figure_out: Path) -> None:
    plot_df = results[results["preferred_for_main_text"]].copy()
    plot_df = plot_df.sort_values(["tier", "ROR"], ascending=[True, True])

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No active-comparator result met plotting thresholds.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(figure_out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [f"{row.exposure_group} vs {row.comparator_group}" for row in plot_df.itertuples()]
    y = np.arange(len(plot_df))
    x = plot_df["ROR"].to_numpy(dtype=float)
    low = plot_df["ROR_95CI_low"].to_numpy(dtype=float)
    high = plot_df["ROR_95CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - low, high - x])

    height = max(4, 0.48 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, height))
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#22577a", ecolor="#4d4d4d", elinewidth=1, capsize=3)
    ax.axvline(1, color="#8c8c8c", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Restricted reporting odds ratio (log scale)")
    ax.set_title("Active-comparator strict fall comparisons")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(figure_out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    args = parser.parse_args()

    df = read_main_dataset(args.main_dataset)
    results = build_active_comparator_results(df)
    qc = build_qc(df, results)
    validate_results(results)

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    plot_forest(results, args.figure_out)

    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.figure_out}")
    print(f"Comparisons analyzed: {len(results):,}")
    print(f"Main-text comparisons: {int(results['preferred_for_main_text'].sum()):,}")


if __name__ == "__main__":
    main()
