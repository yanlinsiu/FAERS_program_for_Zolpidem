from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_PS_ONLY_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s3_ps_only_sensitivity.csv"
DEFAULT_EXCLUDING_MIXED_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s4_excluding_mixed_exposure_sensitivity.csv"
DEFAULT_REPORTING_SOURCE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s5_reporting_source_stratified_sensitivity.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "09_sensitivity_analyses_qc.csv"

DRUG_KEYS = [
    "zolpidem",
    "eszopiclone",
    "zaleplon",
    "zopiclone",
    "benzodiazepine",
    "orexin_antagonist",
    "z_drug",
    "other_z_drug",
]
BASE_COLUMNS = [
    "analysis_eligible_main",
    "strict_fall",
    "country_group",
    "rept_cod",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
    "mixed_z_drug_ps_ss",
    "mixed_sedative_hypnotic_group_ps_ss",
    "z_drug_plus_benzo_ps_ss",
    "n_sedative_hypnotic_drugs_ps_only",
    "n_sedative_hypnotic_groups_ps_only",
]
CORE_COMPARISON_IDS = [
    "zolpidem_vs_other_z_drugs",
    "zolpidem_vs_benzodiazepines",
    "zolpidem_vs_orexin_antagonists",
    "z_drugs_vs_benzodiazepines",
    "z_drugs_vs_orexin_antagonists",
]


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    tier: str
    exposure_label: str
    comparator_label: str
    exposure_mask: str
    comparator_mask: str
    research_question: str


@dataclass(frozen=True)
class AnalysisContext:
    analysis_type: str
    role_suffix: str
    base_mask: pd.Series
    stratum_variable: str
    stratum_value: str
    note: str = ""


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def exposure_column(key: str, role_suffix: str) -> str:
    return f"exposure_{key}_{role_suffix}"


def read_main_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")

    exposure_columns = [exposure_column(key, suffix) for key in DRUG_KEYS for suffix in ["ps_ss", "ps_only"]]
    required_columns = BASE_COLUMNS + exposure_columns
    available = pq.ParquetFile(path).schema.names
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing}")
    return pd.read_parquet(path, columns=required_columns)


def build_core_comparisons() -> list[ComparisonSpec]:
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
    ]


def build_masks(df: pd.DataFrame, context: AnalysisContext) -> dict[str, pd.Series]:
    role_suffix = context.role_suffix
    base = context.base_mask
    one_group = pd.to_numeric(df[f"n_sedative_hypnotic_groups_{role_suffix}"], errors="coerce").fillna(0).eq(1)
    one_drug = pd.to_numeric(df[f"n_sedative_hypnotic_drugs_{role_suffix}"], errors="coerce").fillna(0).eq(1)

    masks: dict[str, pd.Series] = {"base": base}
    masks["zolpidem_only"] = base & safe_bool(df[exposure_column("zolpidem", role_suffix)]) & one_drug
    masks["z_drug_only"] = base & safe_bool(df[exposure_column("z_drug", role_suffix)]) & one_group
    masks["benzodiazepines_only"] = base & safe_bool(df[exposure_column("benzodiazepine", role_suffix)]) & one_group
    masks["orexin_antagonists_only"] = base & safe_bool(df[exposure_column("orexin_antagonist", role_suffix)]) & one_group
    masks["other_z_drugs_without_zolpidem_only"] = masks["z_drug_only"] & ~safe_bool(
        df[exposure_column("zolpidem", role_suffix)]
    )
    return masks


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


def direction(ror_low: float, ror_high: float) -> str:
    if ror_low > 1:
        return "exposure_higher"
    if ror_high < 1:
        return "exposure_lower"
    return "not_clearly_different"


def analyze_comparison(
    df: pd.DataFrame,
    context: AnalysisContext,
    masks: dict[str, pd.Series],
    spec: ComparisonSpec,
) -> dict[str, object]:
    exposure = masks[spec.exposure_mask]
    comparator = masks[spec.comparator_mask]
    overlap_n = int((exposure & comparator).sum())
    if overlap_n:
        raise ValueError(f"Comparison masks overlap for {context.analysis_type}/{spec.comparison_id}: {overlap_n}")

    outcome = safe_bool(df["strict_fall"])
    analysis_mask = exposure | comparator
    a = int((exposure & outcome).sum())
    b = int((exposure & ~outcome).sum())
    c = int((comparator & outcome).sum())
    d = int((comparator & ~outcome).sum())
    exposure_n = a + b
    comparator_n = c + d

    metrics = calculate_metrics(a, b, c, d)
    ror_low = float(metrics["ROR_95CI_low"])
    ror_high = float(metrics["ROR_95CI_high"])
    row = {
        "analysis_type": context.analysis_type,
        "role_suffix": context.role_suffix,
        "stratum_variable": context.stratum_variable,
        "stratum_value": context.stratum_value,
        "comparison_id": spec.comparison_id,
        "tier": spec.tier,
        "exposure_group": spec.exposure_label,
        "comparator_group": spec.comparator_label,
        "exposure_mask": spec.exposure_mask,
        "comparator_mask": spec.comparator_mask,
        "research_question": spec.research_question,
        "analysis_n": int(analysis_mask.sum()),
        "base_n": int(context.base_mask.sum()),
        "base_strict_fall_n": int((context.base_mask & outcome).sum()),
        "exposure_n": exposure_n,
        "exposure_fall_n": a,
        "exposure_nonfall_n": b,
        "exposure_fall_percent": (a / exposure_n * 100) if exposure_n else np.nan,
        "comparator_n": comparator_n,
        "comparator_fall_n": c,
        "comparator_nonfall_n": d,
        "comparator_fall_percent": (c / comparator_n * 100) if comparator_n else np.nan,
        "small_count": a < 5 or c < 5,
        "positive_signal": ror_low > 1,
        "direction": direction(ror_low, ror_high),
        "check_analysis_total_matches": int(analysis_mask.sum()) == exposure_n + comparator_n,
        "note": context.note,
    }
    row.update(metrics)
    return row


def run_context(df: pd.DataFrame, context: AnalysisContext) -> pd.DataFrame:
    masks = build_masks(df, context)
    rows = [analyze_comparison(df, context, masks, spec) for spec in build_core_comparisons()]
    return pd.DataFrame(rows)


def build_ps_only_results(df: pd.DataFrame) -> pd.DataFrame:
    context = AnalysisContext(
        analysis_type="ps_only",
        role_suffix="ps_only",
        base_mask=safe_bool(df["analysis_eligible_main"]),
        stratum_variable="overall",
        stratum_value="overall",
    )
    return run_context(df, context)


def build_excluding_mixed_results(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        safe_bool(df["analysis_eligible_main"])
        & ~safe_bool(df["mixed_z_drug_ps_ss"])
        & ~safe_bool(df["mixed_sedative_hypnotic_group_ps_ss"])
        & ~safe_bool(df["z_drug_plus_benzo_ps_ss"])
    )
    context = AnalysisContext(
        analysis_type="excluding_mixed_exposure",
        role_suffix="ps_ss",
        base_mask=base,
        stratum_variable="overall",
        stratum_value="overall",
    )
    return run_context(df, context)


def reporting_source_contexts(df: pd.DataFrame) -> list[AnalysisContext]:
    eligible = safe_bool(df["analysis_eligible_main"])
    contexts: list[AnalysisContext] = []

    for value in ["US", "non-US"]:
        stratum_mask = df["country_group"].astype("object").where(df["country_group"].notna(), "missing").astype(str).eq(value)
        contexts.append(
            AnalysisContext(
                analysis_type="reporting_source_stratified",
                role_suffix="ps_ss",
                base_mask=eligible & stratum_mask,
                stratum_variable="country_group",
                stratum_value=value,
            )
        )

    rept = df["rept_cod"].astype("object").where(df["rept_cod"].notna(), "missing").astype(str)
    for value in sorted(rept.loc[eligible].unique()):
        stratum_mask = rept.eq(value)
        contexts.append(
            AnalysisContext(
                analysis_type="reporting_source_stratified",
                role_suffix="ps_ss",
                base_mask=eligible & stratum_mask,
                stratum_variable="rept_cod",
                stratum_value=value,
                note="Raw FAERS report type code; not interpreted as healthcare-professional vs non-healthcare-professional.",
            )
        )
    return contexts


def build_reporting_source_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    included: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    for context in reporting_source_contexts(df):
        result = run_context(df, context)
        has_interpretable_comparison = bool((~result["small_count"]).any())
        if has_interpretable_comparison:
            included.append(result)
        else:
            skipped.append(
                {
                    "analysis_type": context.analysis_type,
                    "stratum_variable": context.stratum_variable,
                    "stratum_value": context.stratum_value,
                    "base_n": int(context.base_mask.sum()),
                    "skip_reason": "All core comparisons had fewer than 5 strict-fall reports in at least one comparison arm.",
                    "note": context.note,
                }
            )
    if not included:
        raise ValueError("No reporting-source strata had an interpretable core comparison.")
    return pd.concat(included, ignore_index=True), pd.DataFrame(skipped)


def build_qc(
    df: pd.DataFrame,
    ps_only: pd.DataFrame,
    excluding_mixed: pd.DataFrame,
    reporting_source: pd.DataFrame,
    skipped_reporting_source: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = safe_bool(df["analysis_eligible_main"])
    strict_fall = safe_bool(df["strict_fall"])

    def add(qc_domain: str, analysis_type: str, stratum_variable: str, stratum_value: str, metric: str, value: object, note: str = "") -> None:
        rows.append(
            {
                "qc_domain": qc_domain,
                "analysis_type": analysis_type,
                "stratum_variable": stratum_variable,
                "stratum_value": stratum_value,
                "comparison_id": "overall",
                "metric": metric,
                "value": value,
                "note": note,
            }
        )

    add("sensitivity", "overall", "overall", "overall", "input_rows", len(df))
    add("sensitivity", "overall", "overall", "overall", "analysis_eligible_rows", int(eligible.sum()))
    add("sensitivity", "overall", "overall", "overall", "analysis_eligible_strict_fall_rows", int((eligible & strict_fall).sum()))
    add(
        "sensitivity",
        "reporting_source_stratified",
        "rept_cod",
        "overall",
        "rept_cod_mapping_status",
        "raw_distribution_only",
        "rept_cod values are report type codes, so they are not recoded to HCP/non-HCP.",
    )

    for value, count in df.loc[eligible, "country_group"].value_counts(dropna=False).sort_index().items():
        mask = eligible & df["country_group"].eq(value)
        add("sensitivity", "reporting_source_stratified", "country_group", str(value), "stratum_rows", int(mask.sum()))
        add("sensitivity", "reporting_source_stratified", "country_group", str(value), "stratum_strict_fall_rows", int((mask & strict_fall).sum()))

    for value, count in df.loc[eligible, "rept_cod"].value_counts(dropna=False).sort_index().items():
        value_string = "missing" if pd.isna(value) else str(value)
        mask = eligible & df["rept_cod"].astype("object").where(df["rept_cod"].notna(), "missing").astype(str).eq(value_string)
        add("sensitivity", "reporting_source_stratified", "rept_cod", value_string, "stratum_rows", int(mask.sum()))
        add("sensitivity", "reporting_source_stratified", "rept_cod", value_string, "stratum_strict_fall_rows", int((mask & strict_fall).sum()))

    for _, row in skipped_reporting_source.iterrows():
        add(
            "sensitivity",
            row["analysis_type"],
            row["stratum_variable"],
            row["stratum_value"],
            "skip_reason",
            row["skip_reason"],
            row["note"],
        )

    for result_set in [ps_only, excluding_mixed, reporting_source]:
        for _, row in result_set.iterrows():
            for metric in [
                "base_n",
                "base_strict_fall_n",
                "analysis_n",
                "exposure_n",
                "exposure_fall_n",
                "comparator_n",
                "comparator_fall_n",
                "small_count",
                "positive_signal",
                "direction",
                "continuity_correction",
                "check_analysis_total_matches",
            ]:
                rows.append(
                    {
                        "qc_domain": "sensitivity_result",
                        "analysis_type": row["analysis_type"],
                        "stratum_variable": row["stratum_variable"],
                        "stratum_value": row["stratum_value"],
                        "comparison_id": row["comparison_id"],
                        "metric": metric,
                        "value": row[metric],
                        "note": row["note"],
                    }
                )
    return pd.DataFrame(rows)


def validate_results(results: pd.DataFrame, label: str) -> None:
    if results.empty:
        raise ValueError(f"{label} results are empty.")
    if not results["check_analysis_total_matches"].all():
        failed = results.loc[~results["check_analysis_total_matches"], ["analysis_type", "stratum_variable", "stratum_value", "comparison_id"]].to_dict("records")
        raise ValueError(f"{label} 2x2 total checks failed: {failed}")

    metric_columns = ["ROR", "ROR_95CI_low", "ROR_95CI_high", "PRR", "PRR_95CI_low", "PRR_95CI_high"]
    if not np.isfinite(results[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite ROR/PRR metrics.")

    unexpected = sorted(set(results["comparison_id"]) - set(CORE_COMPARISON_IDS))
    if unexpected:
        raise ValueError(f"{label} contains unexpected comparisons: {unexpected}")


def validate_cross_checks(df: pd.DataFrame, ps_only: pd.DataFrame, excluding_mixed: pd.DataFrame) -> None:
    eligible = safe_bool(df["analysis_eligible_main"])
    ps_only_rows = int(ps_only.loc[ps_only["comparison_id"].eq("z_drugs_vs_benzodiazepines"), "analysis_n"].iloc[0])
    ps_ss_eligible_exposed = int(
        (
            eligible
            & (
                safe_bool(df["exposure_z_drug_ps_ss"])
                | safe_bool(df["exposure_benzodiazepine_ps_ss"])
            )
        ).sum()
    )
    if ps_only_rows > ps_ss_eligible_exposed:
        raise ValueError("PS-only sensitivity has more Z-drug/benzodiazepine rows than the matching PS+SS exposed set.")

    mixed_clean_n = int(excluding_mixed["base_n"].max())
    if mixed_clean_n > int(eligible.sum()):
        raise ValueError("Excluding-mixed base rows exceed main eligible rows.")


def write_outputs(
    ps_only: pd.DataFrame,
    excluding_mixed: pd.DataFrame,
    reporting_source: pd.DataFrame,
    qc: pd.DataFrame,
    ps_only_out: Path,
    excluding_mixed_out: Path,
    reporting_source_out: Path,
    qc_out: Path,
) -> None:
    for path in [ps_only_out, excluding_mixed_out, reporting_source_out, qc_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    ps_only.to_csv(ps_only_out, index=False, encoding="utf-8-sig")
    excluding_mixed.to_csv(excluding_mixed_out, index=False, encoding="utf-8-sig")
    reporting_source.to_csv(reporting_source_out, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_out, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--ps-only-out", type=Path, default=DEFAULT_PS_ONLY_OUT)
    parser.add_argument("--excluding-mixed-out", type=Path, default=DEFAULT_EXCLUDING_MIXED_OUT)
    parser.add_argument("--reporting-source-out", type=Path, default=DEFAULT_REPORTING_SOURCE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    df = read_main_dataset(args.main_dataset)
    ps_only = build_ps_only_results(df)
    excluding_mixed = build_excluding_mixed_results(df)
    reporting_source, skipped_reporting_source = build_reporting_source_results(df)
    qc = build_qc(df, ps_only, excluding_mixed, reporting_source, skipped_reporting_source)

    validate_results(ps_only, "PS-only sensitivity")
    validate_results(excluding_mixed, "Excluding-mixed sensitivity")
    validate_results(reporting_source, "Reporting-source sensitivity")
    validate_cross_checks(df, ps_only, excluding_mixed)

    write_outputs(
        ps_only,
        excluding_mixed,
        reporting_source,
        qc,
        args.ps_only_out,
        args.excluding_mixed_out,
        args.reporting_source_out,
        args.qc_out,
    )

    print(f"Wrote {args.ps_only_out}")
    print(f"Wrote {args.excluding_mixed_out}")
    print(f"Wrote {args.reporting_source_out}")
    print(f"Wrote {args.qc_out}")
    print(f"PS-only comparisons: {len(ps_only):,}")
    print(f"Excluding-mixed comparisons: {len(excluding_mixed):,}")
    print(f"Reporting-source comparisons: {len(reporting_source):,}")


if __name__ == "__main__":
    main()
