from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_TABLE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_3_adjusted_ror.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "08_adjusted_model_qc.csv"
DEFAULT_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_3_adjusted_ror_forest.png"

BASE_COLUMNS = [
    "analysis_eligible_main",
    "strict_fall",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
]
EXPOSURE_COLUMNS = [
    "exposure_zolpidem_ps_ss",
    "exposure_eszopiclone_ps_ss",
    "exposure_zaleplon_ps_ss",
    "exposure_zopiclone_ps_ss",
    "exposure_benzodiazepine_ps_ss",
    "exposure_orexin_antagonist_ps_ss",
    "exposure_other_insomnia_related_ps_ss",
    "exposure_z_drug_ps_ss",
    "exposure_other_z_drug_ps_ss",
]
COVARIATE_COLUMNS = [
    "age_group_3",
    "sex_clean",
    "year",
    "quarter",
    "country_group",
    "rept_cod",
    "e_sub",
    "polypharmacy",
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

CATEGORICAL_COVARIATES = [
    "age_group_3",
    "sex_clean",
    "year",
    "quarter",
    "country_group",
    "rept_cod",
    "e_sub",
]
BINARY_COVARIATES = [
    "polypharmacy",
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


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    tier: str
    exposure_label: str
    comparator_label: str
    exposure_mask: str
    comparator_mask: str


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_label: str
    categorical_covariates: tuple[str, ...]
    binary_covariates: tuple[str, ...]


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def read_main_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")
    required = BASE_COLUMNS + EXPOSURE_COLUMNS + COVARIATE_COLUMNS
    available = pq.ParquetFile(path).schema.names
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing}")
    return pd.read_parquet(path, columns=required)


def build_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    eligible = safe_bool(df["analysis_eligible_main"])
    one_group = pd.to_numeric(df["n_sedative_hypnotic_groups_ps_ss"], errors="coerce").fillna(0).eq(1)
    one_drug = pd.to_numeric(df["n_sedative_hypnotic_drugs_ps_ss"], errors="coerce").fillna(0).eq(1)

    masks: dict[str, pd.Series] = {"eligible": eligible}
    masks["zolpidem_only"] = eligible & safe_bool(df["exposure_zolpidem_ps_ss"]) & one_drug
    masks["z_drug_only"] = eligible & safe_bool(df["exposure_z_drug_ps_ss"]) & one_group
    masks["benzodiazepines_only"] = eligible & safe_bool(df["exposure_benzodiazepine_ps_ss"]) & one_group
    masks["orexin_antagonists_only"] = eligible & safe_bool(df["exposure_orexin_antagonist_ps_ss"]) & one_group
    masks["other_insomnia_related_only"] = eligible & safe_bool(df["exposure_other_insomnia_related_ps_ss"]) & one_group
    masks["other_z_drugs_without_zolpidem_only"] = masks["z_drug_only"] & ~safe_bool(df["exposure_zolpidem_ps_ss"])
    return masks


def build_comparisons() -> list[ComparisonSpec]:
    return [
        ComparisonSpec("zolpidem_vs_other_z_drugs", "zolpidem_centered", "zolpidem-only", "other Z-drugs-only", "zolpidem_only", "other_z_drugs_without_zolpidem_only"),
        ComparisonSpec("zolpidem_vs_benzodiazepines", "zolpidem_centered", "zolpidem-only", "benzodiazepines-only", "zolpidem_only", "benzodiazepines_only"),
        ComparisonSpec("zolpidem_vs_orexin_antagonists", "zolpidem_centered", "zolpidem-only", "orexin antagonists-only", "zolpidem_only", "orexin_antagonists_only"),
        ComparisonSpec("zolpidem_vs_other_insomnia_related", "zolpidem_centered", "zolpidem-only", "other insomnia-related drugs-only", "zolpidem_only", "other_insomnia_related_only"),
        ComparisonSpec("z_drugs_vs_benzodiazepines", "class_comparison", "Z-drugs-only", "benzodiazepines-only", "z_drug_only", "benzodiazepines_only"),
        ComparisonSpec("z_drugs_vs_orexin_antagonists", "class_comparison", "Z-drugs-only", "orexin antagonists-only", "z_drug_only", "orexin_antagonists_only"),
        ComparisonSpec("benzodiazepines_vs_orexin_antagonists", "class_comparison", "benzodiazepines-only", "orexin antagonists-only", "benzodiazepines_only", "orexin_antagonists_only"),
        ComparisonSpec("other_insomnia_related_vs_orexin_antagonists", "class_comparison", "other insomnia-related drugs-only", "orexin antagonists-only", "other_insomnia_related_only", "orexin_antagonists_only"),
    ]


def build_models() -> list[ModelSpec]:
    return [
        ModelSpec("crude", "Crude", (), ()),
        ModelSpec("model_1_demographic_time", "Model 1: demographic and time", ("age_group_3", "sex_clean", "year", "quarter"), ()),
        ModelSpec("model_2_reporting", "Model 2: plus reporting source", ("age_group_3", "sex_clean", "year", "quarter", "country_group", "rept_cod", "e_sub"), ()),
        ModelSpec(
            "model_3_full",
            "Model 3: plus medication and indication",
            ("age_group_3", "sex_clean", "year", "quarter", "country_group", "rept_cod", "e_sub"),
            (
                "polypharmacy",
                "is_antidepressant",
                "is_antipsychotic",
                "is_opioid",
                "is_antiepileptic",
                "indi_insomnia",
                "indi_anxiety",
                "indi_depression",
                "indi_pain",
                "indi_epilepsy",
            ),
        ),
    ]


def prepare_subset(df: pd.DataFrame, exposure_mask: pd.Series, comparator_mask: pd.Series) -> pd.DataFrame:
    subset = df.loc[exposure_mask | comparator_mask, ["strict_fall"] + COVARIATE_COLUMNS].copy()
    subset["strict_fall_int"] = safe_bool(subset["strict_fall"]).astype(int)
    subset["exposure_int"] = exposure_mask.loc[subset.index].astype(int).to_numpy()

    for column in CATEGORICAL_COVARIATES:
        subset[column] = subset[column].astype("object").where(subset[column].notna(), "missing").astype(str)
    for column in BINARY_COVARIATES:
        subset[column] = safe_bool(subset[column]).astype(int)
    return subset


def usable_terms(subset: pd.DataFrame, model: ModelSpec) -> tuple[list[str], list[str]]:
    terms: list[str] = []
    skipped: list[str] = []
    for column in model.categorical_covariates:
        if subset[column].nunique(dropna=False) >= 2:
            terms.append(f"C({column})")
        else:
            skipped.append(column)
    for column in model.binary_covariates:
        if subset[column].nunique(dropna=False) >= 2:
            terms.append(column)
        else:
            skipped.append(column)
    return terms, skipped


def direction(or_low: float, or_high: float) -> str:
    if or_low > 1:
        return "exposure_higher"
    if or_high < 1:
        return "exposure_lower"
    return "not_clearly_different"


def fit_model(subset: pd.DataFrame, model: ModelSpec) -> dict[str, object]:
    terms, skipped = usable_terms(subset, model)
    formula = "strict_fall_int ~ exposure_int"
    if terms:
        formula += " + " + " + ".join(terms)

    result: dict[str, object] = {
        "model_id": model.model_id,
        "model_label": model.model_label,
        "formula": formula,
        "covariates_used": ";".join(terms),
        "covariates_skipped": ";".join(skipped),
        "converged": False,
        "fit_status": "not_fit",
        "OR": np.nan,
        "OR_95CI_low": np.nan,
        "OR_95CI_high": np.nan,
        "p_value": np.nan,
        "direction": "not_available",
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", PerfectSeparationWarning)
            fitted = smf.glm(formula=formula, data=subset, family=sm.families.Binomial()).fit(maxiter=200)

        coef = float(fitted.params["exposure_int"])
        se = float(fitted.bse["exposure_int"])
        or_value = math.exp(coef)
        or_low = math.exp(coef - 1.96 * se)
        or_high = math.exp(coef + 1.96 * se)
        warning_names = sorted({warning.category.__name__ for warning in caught})

        result.update(
            {
                "converged": bool(getattr(fitted, "converged", False)),
                "fit_status": "ok" if not warning_names else "ok_with_warning:" + ";".join(warning_names),
                "OR": or_value,
                "OR_95CI_low": or_low,
                "OR_95CI_high": or_high,
                "p_value": float(fitted.pvalues["exposure_int"]),
                "direction": direction(or_low, or_high),
            }
        )
    except Exception as exc:
        result["fit_status"] = f"failed:{type(exc).__name__}:{exc}"
    return result


def build_results(df: pd.DataFrame) -> pd.DataFrame:
    masks = build_masks(df)
    rows: list[dict[str, object]] = []

    for comparison in build_comparisons():
        exposure_mask = masks[comparison.exposure_mask]
        comparator_mask = masks[comparison.comparator_mask]
        if int((exposure_mask & comparator_mask).sum()):
            raise ValueError(f"Comparison masks overlap: {comparison.comparison_id}")
        subset = prepare_subset(df, exposure_mask, comparator_mask)

        exposure_rows = subset[subset["exposure_int"].eq(1)]
        comparator_rows = subset[subset["exposure_int"].eq(0)]
        base = {
            "comparison_id": comparison.comparison_id,
            "tier": comparison.tier,
            "exposure_group": comparison.exposure_label,
            "comparator_group": comparison.comparator_label,
            "analysis_n": len(subset),
            "exposure_n": len(exposure_rows),
            "exposure_fall_n": int(exposure_rows["strict_fall_int"].sum()),
            "exposure_fall_percent": float(exposure_rows["strict_fall_int"].mean() * 100),
            "comparator_n": len(comparator_rows),
            "comparator_fall_n": int(comparator_rows["strict_fall_int"].sum()),
            "comparator_fall_percent": float(comparator_rows["strict_fall_int"].mean() * 100),
        }
        for model in build_models():
            row = dict(base)
            row.update(fit_model(subset, model))
            rows.append(row)
    return pd.DataFrame(rows)


def build_qc(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        for metric in [
            "analysis_n",
            "exposure_n",
            "exposure_fall_n",
            "comparator_n",
            "comparator_fall_n",
            "converged",
            "fit_status",
            "covariates_used",
            "covariates_skipped",
            "direction",
        ]:
            rows.append(
                {
                    "qc_domain": "adjusted_ror",
                    "comparison_id": row["comparison_id"],
                    "model_id": row["model_id"],
                    "metric": metric,
                    "value": row[metric],
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def validate_results(results: pd.DataFrame) -> None:
    failed = results[results["fit_status"].astype(str).str.startswith("failed")]
    if not failed.empty:
        failed_ids = failed[["comparison_id", "model_id", "fit_status"]].to_dict("records")
        raise ValueError(f"Adjusted ROR model failures: {failed_ids}")
    metric_columns = ["OR", "OR_95CI_low", "OR_95CI_high"]
    if not np.isfinite(results[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Adjusted ROR results contain non-finite OR metrics.")


def plot_model3(results: pd.DataFrame, figure_out: Path) -> None:
    plot_df = results[results["model_id"].eq("model_3_full")].copy()
    plot_df = plot_df.sort_values(["tier", "OR"], ascending=[True, True])
    labels = [f"{row.exposure_group} vs {row.comparator_group}" for row in plot_df.itertuples()]

    y = np.arange(len(plot_df))
    x = plot_df["OR"].to_numpy(dtype=float)
    low = plot_df["OR_95CI_low"].to_numpy(dtype=float)
    high = plot_df["OR_95CI_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - low, high - x])

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.5 * len(plot_df) + 1.8)))
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#31572c", ecolor="#4d4d4d", elinewidth=1, capsize=3)
    ax.axvline(1, color="#8c8c8c", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Adjusted odds ratio for strict fall reports (log scale)")
    ax.set_title("Fully adjusted active-comparator models")
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
    results = build_results(df)
    qc = build_qc(results)
    validate_results(results)

    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.table_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    plot_model3(results, args.figure_out)

    print(f"Wrote {args.table_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.figure_out}")
    print(f"Comparisons analyzed: {results['comparison_id'].nunique():,}")
    print(f"Models fit: {len(results):,}")


if __name__ == "__main__":
    main()
