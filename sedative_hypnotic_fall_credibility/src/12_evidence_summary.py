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


PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
DEFAULT_SIGNAL = TABLE_DIR / "table_1_signal_landscape.csv"
DEFAULT_ACTIVE = TABLE_DIR / "table_2_active_comparator_results.csv"
DEFAULT_ADJUSTED = TABLE_DIR / "table_3_adjusted_ror.csv"
DEFAULT_PS_ONLY = TABLE_DIR / "table_s3_ps_only_sensitivity.csv"
DEFAULT_CLEAN = TABLE_DIR / "table_s4_excluding_mixed_exposure_sensitivity.csv"
DEFAULT_REPORTING = TABLE_DIR / "table_s5_reporting_source_stratified_sensitivity.csv"
DEFAULT_PHENOTYPE = TABLE_DIR / "table_4_phenotype_fingerprint_by_drug_group.csv"
DEFAULT_SPEC = PROJECT_DIR / "configs" / "credibility_score_spec.csv"
DEFAULT_EXTERNAL = PROJECT_DIR / "configs" / "credibility_external_evidence.csv"
DEFAULT_TABLE_OUT = TABLE_DIR / "table_5_evidence_summary.csv"
DEFAULT_COMPONENT_OUT = TABLE_DIR / "table_s14_evidence_component_details.csv"
DEFAULT_LOO_OUT = TABLE_DIR / "table_s17_credibility_leave_one_domain_out.csv"
DEFAULT_THRESHOLD_OUT = TABLE_DIR / "table_s18_credibility_threshold_sensitivity.csv"
DEFAULT_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_5_evidence_summary_heatmap.png"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "12_evidence_summary_qc.csv"

CORE_DOMAINS = (
    "traditional_signal_strength",
    "active_comparator_consistency",
    "adjusted_model_stability",
    "exposure_definition_sensitivity",
    "reporting_structure_stability",
)
ALLOWED_STATES = {"supportive", "neutral", "contradictory", "not_available"}


@dataclass(frozen=True)
class EvidenceTarget:
    target_key: str
    target_label: str
    signal_key: str
    signal_level: str
    phenotype_group: str | None
    comparison_ids: tuple[str, ...]
    robustness_comparison_ids: tuple[str, ...] = ()


TARGETS = (
    EvidenceTarget("zolpidem", "Zolpidem", "zolpidem", "drug", "zolpidem_only", (
        "zolpidem_vs_other_z_drugs", "zolpidem_vs_benzodiazepines",
        "zolpidem_vs_orexin_antagonists", "zolpidem_vs_other_insomnia_related"),
        ("zolpidem_vs_other_z_drugs", "zolpidem_vs_benzodiazepines", "zolpidem_vs_orexin_antagonists")),
    EvidenceTarget("z_drug", "Z-drugs", "z_drug", "group", None,
                   ("z_drugs_vs_benzodiazepines", "z_drugs_vs_orexin_antagonists"),
                   ("z_drugs_vs_benzodiazepines", "z_drugs_vs_orexin_antagonists")),
    EvidenceTarget("other_z_drug", "Other Z-drugs", "other_z_drug", "group",
                   "other_z_drugs_without_zolpidem_only", ()),
    EvidenceTarget("benzodiazepine", "Benzodiazepines", "benzodiazepine", "group",
                   "benzodiazepines_only", ("benzodiazepines_vs_orexin_antagonists",)),
    EvidenceTarget("orexin_antagonist", "Orexin receptor antagonists", "orexin_antagonist", "group", None, ()),
    EvidenceTarget("other_insomnia_related", "Other insomnia-related drugs", "other_insomnia_related", "group", None,
                   ("other_insomnia_related_vs_orexin_antagonists",)),
)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required evidence input not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def component(score: int | None, max_points: int, state: str, detail: str) -> dict[str, object]:
    if state not in ALLOWED_STATES:
        raise ValueError(f"Invalid evidence state: {state}")
    if state == "not_available":
        score = None
    return {"score": score, "available_points": 0 if score is None else max_points,
            "max_points": max_points, "status": state, "detail": detail}


def classify_estimates(df: pd.DataFrame, value_col: str, low_col: str, high_col: str) -> dict[str, object]:
    work = df.copy()
    for column in (value_col, low_col, high_col):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[value_col, low_col, high_col])
    n = len(work)
    return {
        "n": n,
        "directional_n": int((work[value_col] > 1).sum()),
        "supported_n": int((work[low_col] > 1).sum()),
        "reverse_n": int((work[high_col] < 1).sum()),
    }


def comparison_detail(df: pd.DataFrame, value_col: str, low_col: str, high_col: str,
                      id_columns: tuple[str, ...] = ("comparison_id",)) -> str:
    details = []
    for _, row in df.iterrows():
        label = "/".join(str(row.get(column, "")) for column in id_columns)
        details.append(f"{label}: {value_col}={float(row[value_col]):.2f} "
                       f"({float(row[low_col]):.2f}-{float(row[high_col]):.2f})")
    return " | ".join(details)


def traditional_component(signal: pd.DataFrame, target: EvidenceTarget, strong_n: int = 10,
                          weak_n: int = 5) -> tuple[dict[str, object], dict[str, object]]:
    rows = signal[(signal["analysis_level"] == target.signal_level) & (signal["target_key"] == target.signal_key)]
    if rows.empty:
        return component(None, 2, "not_available", "No matching signal-landscape row."), {}
    row = rows.iloc[0]
    fall_n, low = int(row["fall_n"]), float(row["ROR_95CI_low"])
    strong = fall_n >= strong_n and low > 1 and float(row["IC025"]) > 0 and float(row["OE05"]) > 1
    weak = fall_n >= weak_n and low > 1
    score = 2 if strong else 1 if weak else 0
    if float(row["ROR_95CI_high"]) < 1:
        state = "contradictory"
    elif score > 0:
        state = "supportive"
    else:
        state = "neutral"
    detail = (f"fall_n={fall_n}; ROR={float(row['ROR']):.2f} "
              f"({low:.2f}-{float(row['ROR_95CI_high']):.2f}); "
              f"IC025={float(row['IC025']):.2f}; OE05={float(row['OE05']):.2f}.")
    descriptive = {
        "fall_n": fall_n, "exposed_n": int(row["exposed_n"]), "fall_percent": float(row["fall_percent"]),
        "ROR": float(row["ROR"]), "ROR_95CI_low": low, "ROR_95CI_high": float(row["ROR_95CI_high"]),
        "IC025": float(row["IC025"]), "OE05": float(row["OE05"]),
        "hospitalization_fall_percent": float(row.get("hospitalization_fall_percent", np.nan)),
        "death_fall_percent": float(row.get("death_fall_percent", np.nan)),
        "life_threatening_fall_percent": float(row.get("life_threatening_fall_percent", np.nan)),
    }
    return component(score, 2, state, detail), descriptive


def consistency_component(df: pd.DataFrame, target: EvidenceTarget, value_col: str, low_col: str,
                          high_col: str, max_points: int = 2, model_id: str | None = None,
                          expected_ids: tuple[str, ...] | None = None,
                          ci_rule: str = "lower_bound") -> dict[str, object]:
    expected = target.comparison_ids if expected_ids is None else expected_ids
    if not expected:
        return component(None, max_points, "not_available", "No comparison was prespecified for this target.")
    subset = df[df["comparison_id"].isin(expected)].copy()
    if model_id is not None:
        subset = subset[(subset["model_id"] == model_id) & subset["fit_status"].astype(str).str.startswith("ok")]
    if subset.empty:
        return component(None, max_points, "not_available", "No matching comparison rows were available.")
    found = set(subset["comparison_id"])
    missing = sorted(set(expected) - found)
    if missing:
        return component(None, max_points, "not_available",
                         f"Missing prespecified comparison rows: {'|'.join(missing)}.")
    counts = classify_estimates(subset, value_col, low_col, high_col)
    all_directional = counts["directional_n"] == counts["n"]
    majority_supported = (counts["supported_n"] >= math.ceil(counts["n"] / 2)
                          if ci_rule == "lower_bound" else all_directional)
    if counts["reverse_n"] > 0:
        score, state = 0, "contradictory"
    elif all_directional and majority_supported:
        score, state = max_points, "supportive"
    elif all_directional:
        score, state = 1, "supportive"
    else:
        score, state = 0, "neutral"
    detail = (f"directional={counts['directional_n']}/{counts['n']}; CI-supported={counts['supported_n']}/{counts['n']}; "
              f"precise-reverse={counts['reverse_n']}/{counts['n']}. " +
              comparison_detail(subset, value_col, low_col, high_col))
    return component(score, max_points, state, detail)


def exposure_sensitivity_component(ps_only: pd.DataFrame, clean: pd.DataFrame,
                                   target: EvidenceTarget, ci_rule: str = "lower_bound") -> dict[str, object]:
    expected = target.robustness_comparison_ids
    subcomponents = [
        ("PS-only", consistency_component(ps_only, target, "ROR", "ROR_95CI_low", "ROR_95CI_high", 1,
                                           expected_ids=expected, ci_rule=ci_rule)),
        ("clean exposure", consistency_component(clean, target, "ROR", "ROR_95CI_low", "ROR_95CI_high", 1,
                                                  expected_ids=expected, ci_rule=ci_rule)),
    ]
    available = [(name, item) for name, item in subcomponents if item["status"] != "not_available"]
    if not available:
        return component(None, 2, "not_available", "Neither PS-only nor clean-exposure analysis was available.")
    score = sum(int(item["score"]) for _, item in available)
    state = "contradictory" if any(item["status"] == "contradictory" for _, item in available) else (
        "supportive" if score > 0 else "neutral")
    detail = " || ".join(f"{name}: {item['status']}, score={item['score']}/1; {item['detail']}" for name, item in subcomponents)
    result = component(score, len(available), state, detail)
    result["max_points"] = 2
    return result


def stratified_component(df: pd.DataFrame, target: EvidenceTarget,
                         minimum_groups: int, directional_fraction: float) -> dict[str, object]:
    expected = target.robustness_comparison_ids
    if df.empty or not expected:
        reason = "No stratified result table was available." if df.empty else "No stratified comparison was prespecified for this target."
        return component(None, 1, "not_available", reason)
    subset = df[df["comparison_id"].isin(expected)].copy()
    if subset.empty:
        return component(None, 1, "not_available", "No matching stratified rows were available.")
    missing = sorted(set(expected) - set(subset["comparison_id"]))
    if missing:
        return component(None, 1, "not_available", f"Missing prespecified stratified comparisons: {'|'.join(missing)}.")
    group_columns = ("stratum_variable", "stratum_value")
    present_columns = tuple(column for column in group_columns if column in subset.columns)
    counts = classify_estimates(subset, "ROR", "ROR_95CI_low", "ROR_95CI_high")
    comparison_checks = []
    for comparison_id, comparison_rows in subset.groupby("comparison_id"):
        comparison_counts = classify_estimates(comparison_rows, "ROR", "ROR_95CI_low", "ROR_95CI_high")
        group_n = comparison_rows[list(present_columns)].drop_duplicates().shape[0] if present_columns else 0
        fraction = comparison_counts["directional_n"] / comparison_counts["n"] if comparison_counts["n"] else 0
        comparison_checks.append((comparison_id, group_n, fraction))
    stable_by_comparison = all(group_n >= minimum_groups and fraction >= directional_fraction
                               for _, group_n, fraction in comparison_checks)
    if counts["reverse_n"] > 0:
        score, state = 0, "contradictory"
    elif stable_by_comparison:
        score, state = 1, "supportive"
    else:
        score, state = 0, "neutral"
    check_text = "; ".join(f"{comparison_id}: groups={group_n}, directional={fraction:.1%}"
                           for comparison_id, group_n, fraction in comparison_checks)
    detail = (f"per-comparison checks=[{check_text}]; "
              f"precise-reverse={counts['reverse_n']}/{counts['n']}. " +
              comparison_detail(subset, "ROR", "ROR_95CI_low", "ROR_95CI_high", present_columns + ("comparison_id",)))
    return component(score, 1, state, detail)


def phenotype_modifier(phenotype: pd.DataFrame, target: EvidenceTarget) -> dict[str, object]:
    if target.phenotype_group is None:
        return component(None, 0, "not_available", "No direct phenotype group was prespecified for this target.")
    group = phenotype[phenotype["drug_group"] == target.phenotype_group]
    none_row = group[group["phenotype_component"] == "no_mechanistic_co_phenotype"]
    if none_row.empty:
        return component(None, 0, "not_available", "No matching phenotype row was available.")
    mechanistic = 100 - float(none_row.iloc[0]["phenotype_percent"])
    state = "supportive" if mechanistic >= 40 else "neutral"
    result = component(0, 0, state, f"Mechanistic co-phenotype percent={mechanistic:.1f}; prespecified threshold=40.0%.")
    result["modifier_value"] = mechanistic
    return result


def external_modifier(external: pd.DataFrame, target: EvidenceTarget) -> dict[str, object]:
    rows = external[external["target_key"] == target.target_key]
    if rows.empty:
        return component(None, 0, "not_available", "No prespecified external-evidence classification was available.")
    row = rows.iloc[0]
    classification = str(row["classification"])
    state_map = {"established": "supportive", "suggestive": "supportive", "absent": "neutral", "conflicting": "contradictory"}
    if classification not in state_map:
        raise ValueError(f"Unsupported external evidence classification: {classification}")
    result = component(0, 0, state_map[classification], f"{row['detail']} Source basis: {row['source_basis']}.")
    result["modifier_classification"] = classification
    return result


def support_class(score: int, available: int, components: dict[str, dict[str, object]]) -> str:
    if available < 6:
        return "insufficient evidence"
    major_contradiction = any(components[name]["status"] == "contradictory" for name in CORE_DOMAINS)
    key_missing = any(components[name]["status"] == "not_available"
                      for name in ("active_comparator_consistency", "adjusted_model_stability"))
    active = components["active_comparator_consistency"]["score"] or 0
    adjusted = components["adjusted_model_stability"]["score"] or 0
    if score >= 8 and active >= 1 and adjusted >= 1 and not major_contradiction and not key_missing:
        return "high support"
    if score >= 5 or score >= 8:
        return "moderate support"
    return "low support"


def build_summary(signal: pd.DataFrame, active: pd.DataFrame, adjusted: pd.DataFrame,
                  ps_only: pd.DataFrame, clean: pd.DataFrame, reporting: pd.DataFrame,
                  phenotype: pd.DataFrame, external: pd.DataFrame,
                  strong_n: int = 10, weak_n: int = 5, ci_rule: str = "lower_bound"
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, detail_rows = [], []
    for target in TARGETS:
        traditional, descriptive = traditional_component(signal, target, strong_n=strong_n, weak_n=weak_n)
        components = {
            "traditional_signal_strength": traditional,
            "active_comparator_consistency": consistency_component(active, target, "ROR", "ROR_95CI_low", "ROR_95CI_high", ci_rule=ci_rule),
            "adjusted_model_stability": consistency_component(adjusted, target, "OR", "OR_95CI_low", "OR_95CI_high", model_id="model_3_full", ci_rule=ci_rule),
            "exposure_definition_sensitivity": exposure_sensitivity_component(ps_only, clean, target, ci_rule=ci_rule),
            "reporting_structure_stability": stratified_component(reporting, target, 2, 0.75),
            "phenotype_coherence": phenotype_modifier(phenotype, target),
            "external_evidence": external_modifier(external, target),
        }
        score = sum(int(components[name]["score"]) for name in CORE_DOMAINS if components[name]["score"] is not None)
        available = sum(int(components[name]["available_points"]) for name in CORE_DOMAINS)
        contradictions = [name for name in CORE_DOMAINS if components[name]["status"] == "contradictory"]
        classification = support_class(score, available, components)
        phenotype_class = components["phenotype_coherence"]["status"]
        external_class = components["external_evidence"].get("modifier_classification", "unavailable")
        interpretation = (f"{classification.title()} for an elevated strict-fall reporting signal; "
                          f"phenotype={phenotype_class}; external evidence={external_class}.")
        row = {
            "target_key": target.target_key, "target_label": target.target_label, **descriptive,
            **{f"{name}_score": components[name]["score"] for name in CORE_DOMAINS},
            **{f"{name}_status": components[name]["status"] for name in CORE_DOMAINS},
            "obtained_score": score, "available_score": available,
            "score_fraction_available": score / available if available else np.nan,
            "contradiction_flag": bool(contradictions), "contradictory_domains": "|".join(contradictions),
            "support_class": classification, "phenotype_modifier": phenotype_class,
            "external_evidence_modifier": external_class, "interpretation": interpretation,
            # Compatibility aliases for downstream packaging; values now mean internal support only.
            "total_score": score, "max_score": available, "credibility_class": classification,
        }
        rows.append(row)
        for name, item in components.items():
            detail_rows.append({
                "target_key": target.target_key, "target_label": target.target_label,
                "evidence_domain": name, "domain_type": "core" if name in CORE_DOMAINS else "modifier",
                "score": item["score"], "available_points": item["available_points"],
                "status": item["status"], "detail": item["detail"],
            })
    summary = pd.DataFrame(rows)
    class_order = pd.CategoricalDtype(
        ["high support", "moderate support", "low support", "insufficient evidence"], ordered=True
    )
    summary["support_class"] = summary["support_class"].astype(class_order)
    summary = summary.sort_values(["support_class", "obtained_score", "fall_n"], ascending=[True, False, False])
    summary["support_class"] = summary["support_class"].astype(str)
    return summary, pd.DataFrame(detail_rows)


def leave_one_domain_out(summary: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, summary_row in summary.iterrows():
        target_details = details[(details["target_key"] == summary_row["target_key"]) & (details["domain_type"] == "core")]
        for _, item in target_details.iterrows():
            remaining = {}
            for _, domain_row in target_details.iterrows():
                name = domain_row["evidence_domain"]
                if name == item["evidence_domain"]:
                    remaining[name] = component(None, 0, "not_available", "Omitted.")
                else:
                    score = None if pd.isna(domain_row["score"]) else int(domain_row["score"])
                    remaining[name] = component(score, int(domain_row["available_points"]), domain_row["status"], "")
                    remaining[name]["available_points"] = int(domain_row["available_points"])
            omitted_available = sum(int(value["available_points"]) for value in remaining.values())
            omitted_score = sum(int(value["score"]) for value in remaining.values() if value["score"] is not None)
            loo_class = support_class(omitted_score, omitted_available, remaining)
            rows.append({
                "target_key": summary_row["target_key"], "target_label": summary_row["target_label"],
                "omitted_domain": item["evidence_domain"], "omitted_domain_status": item["status"],
                "original_score": summary_row["obtained_score"], "original_available_score": summary_row["available_score"],
                "original_support_class": summary_row["support_class"], "loo_score": omitted_score,
                "loo_available_score": omitted_available, "loo_support_class": loo_class,
                "class_changed": loo_class != summary_row["support_class"],
            })
    return pd.DataFrame(rows)


def threshold_sensitivity(inputs: tuple[pd.DataFrame, ...], baseline: pd.DataFrame) -> pd.DataFrame:
    scenarios = (
        ("baseline", 10, 5, "lower_bound", "Prespecified primary rules."),
        ("lenient_case_count", 5, 3, "lower_bound", "Lower fall-report count thresholds."),
        ("conservative_case_count", 20, 10, "lower_bound", "Higher fall-report count thresholds."),
        ("point_estimate_consistency", 10, 5, "point_estimate", "Point-estimate direction used for consistency points; precise reverse results remain contradictions."),
    )
    baseline_classes = baseline.set_index("target_key")["support_class"]
    rows = []
    for scenario, strong_n, weak_n, ci_rule, note in scenarios:
        scenario_summary, _ = build_summary(*inputs, strong_n=strong_n, weak_n=weak_n, ci_rule=ci_rule)
        for _, row in scenario_summary.iterrows():
            rows.append({"scenario": scenario, "target_key": row["target_key"], "target_label": row["target_label"],
                         "strong_case_threshold": strong_n, "weak_case_threshold": weak_n, "ci_rule": ci_rule,
                         "obtained_score": row["obtained_score"], "available_score": row["available_score"],
                         "support_class": row["support_class"], "baseline_support_class": baseline_classes[row["target_key"]],
                         "class_changed_from_baseline": row["support_class"] != baseline_classes[row["target_key"]], "note": note})
    return pd.DataFrame(rows)


def validate_spec(spec: pd.DataFrame) -> None:
    core = spec[spec["domain_type"] == "core"]
    if set(core["domain"]) != set(CORE_DOMAINS):
        raise ValueError("Core domains in the score specification do not match the implementation.")
    if int(core["max_points"].sum()) != 9:
        raise ValueError("Core score must sum to 9 points.")
    expected_thresholds = {
        "traditional_signal_strength": "strong_n=10;weak_n=5",
        "active_comparator_consistency": "supported_fraction=0.5",
        "adjusted_model_stability": "supported_fraction=0.5;model_id=model_3_full",
        "exposure_definition_sensitivity": "subdomains=ps_only|clean_exposure",
        "reporting_structure_stability": "directional_fraction=0.75;minimum_strata=2",
        "phenotype_coherence": "supportive_percent=40",
        "external_evidence": "allowed=established|suggestive|absent|conflicting",
    }
    actual = dict(zip(spec["domain"], spec["threshold"].astype(str)))
    if any(actual.get(name) != value for name, value in expected_thresholds.items()):
        raise ValueError("Configured thresholds do not match the locked implementation constants.")


def validate(summary: pd.DataFrame, details: pd.DataFrame, spec: pd.DataFrame) -> None:
    validate_spec(spec)
    if summary.empty or details.empty:
        raise ValueError("Evidence summary outputs must not be empty.")
    if set(summary["target_key"]) != {target.target_key for target in TARGETS}:
        raise ValueError("Evidence summary targets do not match the prespecified targets.")
    score_columns = [f"{name}_score" for name in CORE_DOMAINS]
    recomputed = summary[score_columns].fillna(0).sum(axis=1)
    if not np.allclose(recomputed, summary["obtained_score"]):
        raise ValueError("Obtained scores do not equal the sum of available core-domain scores.")
    if not details["status"].isin(ALLOWED_STATES).all():
        raise ValueError("An invalid evidence state was emitted.")


def build_qc(summary: pd.DataFrame, details: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "summary_rows", "value": len(summary), "note": ""},
        {"metric": "detail_rows", "value": len(details), "note": ""},
        {"metric": "core_score_max", "value": int(spec.loc[spec["domain_type"] == "core", "max_points"].sum()), "note": "Five-domain internal-support score."},
    ]
    for klass, count in summary["support_class"].value_counts().items():
        rows.append({"metric": f"class_count__{klass}", "value": int(count), "note": ""})
    return pd.DataFrame(rows).assign(qc_domain="evidence_summary")[["qc_domain", "metric", "value", "note"]]


def plot_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    columns = [f"{name}_score" for name in CORE_DOMAINS]
    labels = ["Full database", "Active comparator", "Adjusted model", "Exposure sensitivity", "Reporting structure"]
    plot_df = summary.set_index("target_label")[columns]
    masked = np.ma.masked_invalid(plot_df.to_numpy(dtype=float))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#d9d9d9")
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, len(plot_df) * 0.75)))
    image = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(plot_df)), [f"{label}\n{summary.iloc[i]['obtained_score']}/{summary.iloc[i]['available_score']} available"
                                       for i, label in enumerate(plot_df.index)])
    ax.set_title("Internal support for elevated sedative-hypnotic strict-fall reporting signals")
    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            value = plot_df.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(value) else str(int(value)), ha="center", va="center",
                    color="white" if pd.notna(value) and value >= 1.5 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Core-domain score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the prespecified exploratory strict-fall signal support framework.")
    for flag, default in (("signal", DEFAULT_SIGNAL), ("active", DEFAULT_ACTIVE), ("adjusted", DEFAULT_ADJUSTED),
                          ("ps-only", DEFAULT_PS_ONLY), ("clean", DEFAULT_CLEAN), ("reporting", DEFAULT_REPORTING),
                          ("phenotype", DEFAULT_PHENOTYPE), ("spec", DEFAULT_SPEC), ("external", DEFAULT_EXTERNAL)):
        parser.add_argument(f"--{flag}", type=Path, default=default)
    parser.add_argument("--table-out", type=Path, default=DEFAULT_TABLE_OUT)
    parser.add_argument("--component-out", type=Path, default=DEFAULT_COMPONENT_OUT)
    parser.add_argument("--loo-out", type=Path, default=DEFAULT_LOO_OUT)
    parser.add_argument("--threshold-out", type=Path, default=DEFAULT_THRESHOLD_OUT)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    inputs = (read_csv(args.signal), read_csv(args.active), read_csv(args.adjusted), read_csv(args.ps_only),
              read_csv(args.clean), read_csv(args.reporting, required=False),
              read_csv(args.phenotype), read_csv(args.external))
    summary, details = build_summary(*inputs)
    spec = read_csv(args.spec)
    validate(summary, details, spec)
    loo = leave_one_domain_out(summary, details)
    threshold = threshold_sensitivity(inputs, summary)
    qc = build_qc(summary, details, spec)
    for frame, path in ((summary, args.table_out), (details, args.component_out), (loo, args.loo_out),
                        (threshold, args.threshold_out), (qc, args.qc_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    plot_heatmap(summary, args.figure_out)
    print(summary[["target_label", "obtained_score", "available_score", "support_class", "contradiction_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
