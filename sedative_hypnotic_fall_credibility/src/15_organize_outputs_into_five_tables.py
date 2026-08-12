from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
QC_DIR = ROOT / "outputs" / "qc"
CLUSTER_ML_TABLE_DIR = ROOT / "outputs" / "cluster_serious_ml_1552671" / "tables"
OUT_DIR = ROOT / "outputs" / "organized_five_tables"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def add_source(df: pd.DataFrame, source_table: str, result_block: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "result_block", result_block)
    out.insert(1, "source_table", source_table)
    return out


def ci_text(row: pd.Series, estimate: str, low: str, high: str) -> str:
    if estimate not in row or pd.isna(row.get(estimate)):
        return ""
    if low not in row or high not in row or pd.isna(row.get(low)) or pd.isna(row.get(high)):
        return f"{row[estimate]:.3g}"
    return f"{row[estimate]:.3g} ({row[low]:.3g}-{row[high]:.3g})"


def write_table(df: pd.DataFrame, filename: str) -> Path:
    path = OUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def make_table_1_population_exposure_signal() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    baseline = read_csv(TABLE_DIR / "table_1_baseline_description.csv")
    baseline_out = pd.DataFrame(
        {
            "section": "population_baseline",
            "analysis_level": "baseline",
            "item_key": baseline["variable"],
            "item_label": baseline["category"],
            "drug_group": "",
            "exposed_n": "",
            "exposed_percent": "",
            "strict_fall_n": baseline["strict_fall_n"],
            "strict_fall_percent_within_exposed_or_category": baseline[
                "strict_fall_percent_within_category"
            ],
            "strict_fall_percent_of_all_falls": baseline["strict_fall_percent_of_all_falls"],
            "ROR_95CI": "",
            "PRR_95CI": "",
            "IC025": "",
            "OE05": "",
            "serious_any_percent_among_fall_reports": "",
            "hospitalization_percent_among_fall_reports": "",
            "death_percent_among_fall_reports": "",
            "credibility_class": "",
            "total_score": "",
            "preferred_for_main_text": "",
            "note": "Baseline row; denominator follows the original baseline table.",
        }
    )
    frames.append(add_source(baseline_out, "table_1_baseline_description.csv", "cohort"))

    exposure = read_csv(TABLE_DIR / "table_1b_drug_exposure_description.csv")
    signal = read_csv(TABLE_DIR / "table_1_signal_landscape.csv")
    evidence = read_csv(TABLE_DIR / "table_5_evidence_summary.csv")
    serious = read_csv(TABLE_DIR / "table_6_serious_outcomes_among_fall_reports.csv")

    merged = exposure.merge(
        signal[
            [
                "analysis_level",
                "target_key",
                "ROR",
                "ROR_95CI_low",
                "ROR_95CI_high",
                "PRR",
                "PRR_95CI_low",
                "PRR_95CI_high",
                "IC025",
                "OE05",
                "preferred_for_main_text",
            ]
        ],
        on=["analysis_level", "target_key"],
        how="left",
    )
    merged = merged.merge(
        evidence[["target_key", "credibility_class", "total_score"]],
        on="target_key",
        how="left",
    )
    merged = merged.merge(
        serious[
            [
                "analysis_level",
                "target_key",
                "serious_any_percent_among_fall_reports",
                "serious_hospitalization_percent_among_fall_reports",
                "serious_death_percent_among_fall_reports",
            ]
        ],
        on=["analysis_level", "target_key"],
        how="left",
    )
    drug_out = pd.DataFrame(
        {
            "section": "exposure_signal",
            "analysis_level": merged["analysis_level"],
            "item_key": merged["target_key"],
            "item_label": merged["target_label"],
            "drug_group": merged["drug_group"],
            "exposed_n": merged["exposed_n"],
            "exposed_percent": merged["exposed_percent"],
            "strict_fall_n": merged["strict_fall_n"],
            "strict_fall_percent_within_exposed_or_category": merged[
                "strict_fall_percent_within_exposed"
            ],
            "strict_fall_percent_of_all_falls": "",
            "ROR_95CI": merged.apply(lambda r: ci_text(r, "ROR", "ROR_95CI_low", "ROR_95CI_high"), axis=1),
            "PRR_95CI": merged.apply(lambda r: ci_text(r, "PRR", "PRR_95CI_low", "PRR_95CI_high"), axis=1),
            "IC025": merged["IC025"],
            "OE05": merged["OE05"],
            "serious_any_percent_among_fall_reports": merged[
                "serious_any_percent_among_fall_reports"
            ],
            "hospitalization_percent_among_fall_reports": merged[
                "serious_hospitalization_percent_among_fall_reports"
            ],
            "death_percent_among_fall_reports": merged[
                "serious_death_percent_among_fall_reports"
            ],
            "credibility_class": merged["credibility_class"],
            "total_score": merged["total_score"],
            "preferred_for_main_text": merged["preferred_for_main_text"],
            "note": (
                "strict_fall_percent_within_exposed_or_category is strict_fall_n / exposed_n "
                "for exposure rows; it is a FAERS reporting proportion, not incidence."
            ),
        }
    )
    frames.append(add_source(drug_out, "tables_1b_1_5_6_merged.csv", "main_signal"))

    return pd.concat(frames, ignore_index=True)


def make_table_2_comparative_models() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    active = read_csv(TABLE_DIR / "table_2_active_comparator_results.csv")
    active_out = pd.DataFrame(
        {
            "analysis_family": "active_comparator",
            "analysis_type": "active_comparator",
            "stratum_variable": "",
            "stratum_value": "",
            "comparison_id": active["comparison_id"],
            "tier": active["tier"],
            "model_id": "crude_disproportionality",
            "model_label": "Crude active-comparator disproportionality",
            "exposure_group": active["exposure_group"],
            "comparator_group": active["comparator_group"],
            "analysis_n": active["analysis_n"],
            "exposure_n": active["exposure_n"],
            "exposure_event_n": active["exposure_fall_n"],
            "exposure_event_percent": active["exposure_fall_percent"],
            "comparator_n": active["comparator_n"],
            "comparator_event_n": active["comparator_fall_n"],
            "comparator_event_percent": active["comparator_fall_percent"],
            "estimate_type": "ROR",
            "estimate_95CI": active.apply(lambda r: ci_text(r, "ROR", "ROR_95CI_low", "ROR_95CI_high"), axis=1),
            "p_value": "",
            "direction": active["direction"],
            "preferred_for_main_text": active["preferred_for_main_text"],
            "note": active["research_question"],
        }
    )
    frames.append(add_source(active_out, "table_2_active_comparator_results.csv", "active_comparator"))

    adjusted = read_csv(TABLE_DIR / "table_3_adjusted_ror.csv")
    adjusted_out = pd.DataFrame(
        {
            "analysis_family": "adjusted_model",
            "analysis_type": "adjusted_logistic",
            "stratum_variable": "",
            "stratum_value": "",
            "comparison_id": adjusted["comparison_id"],
            "tier": adjusted["tier"],
            "model_id": adjusted["model_id"],
            "model_label": adjusted["model_label"],
            "exposure_group": adjusted["exposure_group"],
            "comparator_group": adjusted["comparator_group"],
            "analysis_n": adjusted["analysis_n"],
            "exposure_n": adjusted["exposure_n"],
            "exposure_event_n": adjusted["exposure_fall_n"],
            "exposure_event_percent": adjusted["exposure_fall_percent"],
            "comparator_n": adjusted["comparator_n"],
            "comparator_event_n": adjusted["comparator_fall_n"],
            "comparator_event_percent": adjusted["comparator_fall_percent"],
            "estimate_type": "OR",
            "estimate_95CI": adjusted.apply(lambda r: ci_text(r, "OR", "OR_95CI_low", "OR_95CI_high"), axis=1),
            "p_value": adjusted["p_value"],
            "direction": adjusted["direction"],
            "preferred_for_main_text": "",
            "note": "Covariates used: " + adjusted["covariates_used"].fillna("none")
            + "; skipped: "
            + adjusted["covariates_skipped"].fillna("none"),
        }
    )
    frames.append(add_source(adjusted_out, "table_3_adjusted_ror.csv", "adjusted_model"))

    sens_files = [
        "table_s3_ps_only_sensitivity.csv",
        "table_s4_excluding_mixed_exposure_sensitivity.csv",
        "table_s5_reporting_source_stratified_sensitivity.csv",
    ]
    for filename in sens_files:
        sens = read_csv(TABLE_DIR / filename)
        sens_out = pd.DataFrame(
            {
                "analysis_family": "sensitivity",
                "analysis_type": sens["analysis_type"],
                "stratum_variable": sens["stratum_variable"],
                "stratum_value": sens["stratum_value"],
                "comparison_id": sens["comparison_id"],
                "tier": sens["tier"],
                "model_id": "crude_sensitivity",
                "model_label": sens["analysis_type"],
                "exposure_group": sens["exposure_group"],
                "comparator_group": sens["comparator_group"],
                "analysis_n": sens["analysis_n"],
                "exposure_n": sens["exposure_n"],
                "exposure_event_n": sens["exposure_fall_n"],
                "exposure_event_percent": sens["exposure_fall_percent"],
                "comparator_n": sens["comparator_n"],
                "comparator_event_n": sens["comparator_fall_n"],
                "comparator_event_percent": sens["comparator_fall_percent"],
                "estimate_type": "ROR",
                "estimate_95CI": sens.apply(lambda r: ci_text(r, "ROR", "ROR_95CI_low", "ROR_95CI_high"), axis=1),
                "p_value": "",
                "direction": sens["direction"],
                "preferred_for_main_text": "",
                "note": sens["research_question"],
            }
        )
        frames.append(add_source(sens_out, filename, "sensitivity"))

    return pd.concat(frames, ignore_index=True)


def make_table_3_phenotype_support() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    group_fp = read_csv(TABLE_DIR / "table_4_phenotype_fingerprint_by_drug_group.csv")
    group_out = pd.DataFrame(
        {
            "analysis_family": "phenotype_distribution",
            "analysis_set": "drug_group_component",
            "group_key": group_fp["drug_group"],
            "group_label": group_fp["drug_group_label"],
            "phenotype_or_cluster": group_fp["phenotype_component"],
            "phenotype_or_cluster_label": group_fp["phenotype_component_label"],
            "comparison": "",
            "fall_case_n": group_fp["fall_case_n"],
            "event_n": group_fp["phenotype_n"],
            "event_percent": group_fp["phenotype_percent"],
            "estimate_type": "",
            "estimate_95CI": "",
            "p_value": "",
            "fdr_p_value": "",
            "model_label": "",
            "note": "",
        }
    )
    frames.append(add_source(group_out, "table_4_phenotype_fingerprint_by_drug_group.csv", "phenotype_distribution"))

    primary = read_csv(TABLE_DIR / "table_s6_primary_phenotype_distribution.csv")
    primary_out = pd.DataFrame(
        {
            "analysis_family": "primary_phenotype",
            "analysis_set": "drug_group_primary",
            "group_key": primary["drug_group"],
            "group_label": primary["drug_group_label"],
            "phenotype_or_cluster": primary["primary_phenotype"],
            "phenotype_or_cluster_label": primary["primary_phenotype_label"],
            "comparison": "",
            "fall_case_n": primary["fall_case_n"],
            "event_n": primary["phenotype_n"],
            "event_percent": primary["phenotype_percent"],
            "estimate_type": "",
            "estimate_95CI": "",
            "p_value": "",
            "fdr_p_value": "",
            "model_label": "",
            "note": "",
        }
    )
    frames.append(add_source(primary_out, "table_s6_primary_phenotype_distribution.csv", "primary_phenotype"))

    pheno_models = read_csv(TABLE_DIR / "table_s8_phenotype_adjusted_logistic_models.csv")
    model3 = pheno_models[pheno_models["model_id"].eq("model_3_clinical")].copy()
    model_out = pd.DataFrame(
        {
            "analysis_family": "phenotype_adjusted_model",
            "analysis_set": "model_3_clinical",
            "group_key": model3["exposure_group"],
            "group_label": model3["exposure_group"],
            "phenotype_or_cluster": model3["outcome"],
            "phenotype_or_cluster_label": model3["outcome_label"],
            "comparison": model3["comparison"],
            "fall_case_n": model3["analysis_n"],
            "event_n": model3["exposure_phenotype_n"],
            "event_percent": model3["exposure_phenotype_percent"],
            "estimate_type": "adjusted OR",
            "estimate_95CI": model3.apply(lambda r: ci_text(r, "odds_ratio", "ci95_lower", "ci95_upper"), axis=1),
            "p_value": model3["p_value"],
            "fdr_p_value": model3["p_fdr_bh_within_model"],
            "model_label": model3["model_label"],
            "note": model3["note"],
        }
    )
    frames.append(add_source(model_out, "table_s8_phenotype_adjusted_logistic_models.csv", "phenotype_adjusted_model"))

    clusters = read_csv(TABLE_DIR / "table_s12_fall_phenotype_cluster_summary.csv")
    cluster_out = pd.DataFrame(
        {
            "analysis_family": "phenotype_cluster",
            "analysis_set": "cluster_summary",
            "group_key": clusters["phenotype_cluster"],
            "group_label": clusters["cluster_label"],
            "phenotype_or_cluster": clusters["phenotype_cluster"],
            "phenotype_or_cluster_label": clusters["cluster_label"],
            "comparison": "",
            "fall_case_n": clusters["cluster_n"],
            "event_n": "",
            "event_percent": clusters["cluster_percent"],
            "estimate_type": "",
            "estimate_95CI": "",
            "p_value": "",
            "fdr_p_value": "",
            "model_label": "",
            "note": clusters["cluster_rationale"],
        }
    )
    frames.append(add_source(cluster_out, "table_s12_fall_phenotype_cluster_summary.csv", "phenotype_cluster"))

    cluster_dist = read_csv(TABLE_DIR / "table_s13_drug_group_by_cluster_distribution.csv")
    cluster_dist_out = pd.DataFrame(
        {
            "analysis_family": "drug_group_cluster_distribution",
            "analysis_set": "drug_group_by_cluster",
            "group_key": cluster_dist["drug_group"],
            "group_label": cluster_dist["drug_group_label"],
            "phenotype_or_cluster": cluster_dist["phenotype_cluster"],
            "phenotype_or_cluster_label": cluster_dist["phenotype_cluster_label"],
            "comparison": "",
            "fall_case_n": cluster_dist["drug_group_fall_n"],
            "event_n": cluster_dist["cluster_n"],
            "event_percent": cluster_dist["cluster_percent_within_drug_group"],
            "estimate_type": "",
            "estimate_95CI": "",
            "p_value": "",
            "fdr_p_value": "",
            "model_label": "",
            "note": "",
        }
    )
    frames.append(add_source(cluster_dist_out, "table_s13_drug_group_by_cluster_distribution.csv", "drug_group_cluster"))

    return pd.concat(frames, ignore_index=True)


def make_table_4_serious_outcomes_ml() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    serious = read_csv(TABLE_DIR / "table_6_serious_outcomes_among_fall_reports.csv")
    serious_out = pd.DataFrame(
        {
            "analysis_family": "serious_outcomes_among_strict_fall_reports",
            "target": serious["target_key"],
            "target_label": serious["target_label"],
            "scope_or_group": serious["analysis_level"],
            "model": "",
            "rank": "",
            "feature": "",
            "fall_report_n": serious["fall_report_n"],
            "event_n": serious["serious_any_n"],
            "event_percent": serious["serious_any_percent_among_fall_reports"],
            "roc_auc": "",
            "average_precision": "",
            "baseline_positive_percent": "",
            "recall": "",
            "specificity": "",
            "importance_or_shap": "",
            "note": serious["note"],
        }
    )
    frames.append(add_source(serious_out, "table_6_serious_outcomes_among_fall_reports.csv", "serious_outcomes"))

    perf_frames = []
    if CLUSTER_ML_TABLE_DIR.exists():
        perf_frames.extend(sorted(CLUSTER_ML_TABLE_DIR.glob("table_13_ml_serious_outcome_model_performance_*.csv")))
        shap_files = sorted(CLUSTER_ML_TABLE_DIR.glob("table_13_ml_serious_outcome_shap_top_features_*.csv"))
        importance_files = sorted(CLUSTER_ML_TABLE_DIR.glob("table_13_ml_serious_outcome_feature_importance_*.csv"))
    else:
        shap_files = []
        importance_files = []
    perf_frames.extend(sorted(TABLE_DIR.glob("table_13_ml_serious_outcome_model_performance_*.csv")))

    for path in perf_frames:
        perf = read_csv(path)
        perf_out = pd.DataFrame(
            {
                "analysis_family": "ml_model_performance",
                "target": perf["target"],
                "target_label": perf["target"],
                "scope_or_group": perf["scope"],
                "model": perf["model"],
                "rank": "",
                "feature": "",
                "fall_report_n": perf["test_n"],
                "event_n": perf["test_positive_n"],
                "event_percent": perf["test_positive_percent"],
                "roc_auc": perf["roc_auc"],
                "average_precision": perf["average_precision"],
                "baseline_positive_percent": perf["test_positive_percent"],
                "recall": perf["recall"],
                "specificity": perf["specificity"],
                "importance_or_shap": "",
                "note": (
                    "Exploratory report-pattern model; compare average precision with "
                    "baseline_positive_percent before interpreting performance."
                ),
            }
        )
        frames.append(add_source(perf_out, path.name, "ml_performance"))

    for path in shap_files:
        shap = read_csv(path).sort_values("rank").head(10)
        shap_out = pd.DataFrame(
            {
                "analysis_family": "ml_xgboost_top_shap_features",
                "target": shap["target"],
                "target_label": shap["target"],
                "scope_or_group": shap["scope"],
                "model": shap["model"],
                "rank": shap["rank"],
                "feature": shap["feature"],
                "fall_report_n": "",
                "event_n": "",
                "event_percent": "",
                "roc_auc": "",
                "average_precision": "",
                "baseline_positive_percent": "",
                "recall": "",
                "specificity": "",
                "importance_or_shap": shap["mean_abs_shap"],
                "note": shap.get("note", ""),
            }
        )
        frames.append(add_source(shap_out, path.name, "ml_top_shap"))

    for path in importance_files:
        imp = read_csv(path).sort_values("rank").head(10)
        imp_out = pd.DataFrame(
            {
                "analysis_family": "ml_top_model_features",
                "target": imp["target"],
                "target_label": imp["target"],
                "scope_or_group": imp["scope"],
                "model": imp["model"],
                "rank": imp["rank"],
                "feature": imp["feature"],
                "fall_report_n": "",
                "event_n": "",
                "event_percent": "",
                "roc_auc": "",
                "average_precision": "",
                "baseline_positive_percent": "",
                "recall": "",
                "specificity": "",
                "importance_or_shap": imp["abs_importance"],
                "note": "",
            }
        )
        frames.append(add_source(imp_out, path.name, "ml_top_importance"))

    return pd.concat(frames, ignore_index=True)


def make_table_5_definitions_qc_index() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    definitions = read_csv(TABLE_DIR / "table_s2_outcome_phenotype_definitions.csv")
    definitions_out = pd.DataFrame(
        {
            "section": "outcome_phenotype_definition",
            "item_key": definitions["module_key"],
            "item_label": definitions["module_label"],
            "category": definitions["analysis_role"],
            "value": definitions["meddra_pt_terms"],
            "status_or_role": definitions["use_in_main_outcome"],
            "detail": definitions["interpretation_note"],
        }
    )
    frames.append(add_source(definitions_out, "table_s2_outcome_phenotype_definitions.csv", "definitions"))

    dictionary = read_csv(TABLE_DIR / "table_s1_drug_dictionary.csv")
    dictionary_out = pd.DataFrame(
        {
            "section": "drug_dictionary",
            "item_key": dictionary["drug_key"],
            "item_label": dictionary["generic_name"],
            "category": dictionary["drug_group"],
            "value": dictionary["brand_names"],
            "status_or_role": dictionary["analysis_role"],
            "detail": dictionary["manual_review_status"].fillna("")
            + "; "
            + dictionary["source_note"].fillna(""),
        }
    )
    frames.append(add_source(dictionary_out, "table_s1_drug_dictionary.csv", "drug_dictionary"))

    evidence_details = read_csv(TABLE_DIR / "table_s14_evidence_component_details.csv")
    evidence_out = pd.DataFrame(
        {
            "section": "evidence_component",
            "item_key": evidence_details["target_key"],
            "item_label": evidence_details["target_label"],
            "category": evidence_details["evidence_domain"],
            "value": evidence_details["score"].fillna("NA").astype(str)
            + "/"
            + evidence_details["available_points"].astype(str)
            + " available",
            "status_or_role": evidence_details["status"],
            "detail": evidence_details["detail"],
        }
    )
    frames.append(add_source(evidence_out, "table_s14_evidence_component_details.csv", "evidence_components"))

    qc_frames = []
    for path in sorted(QC_DIR.glob("*.csv")):
        if path.name.startswith(("00_", "01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_", "12_", "14_")):
            qc = read_csv(path)
            if {"metric", "value"}.issubset(qc.columns):
                small = qc.head(20).copy()
                qc_frames.append((path.name, small))
    for filename, qc in qc_frames:
        qc_out = pd.DataFrame(
            {
                "section": "qc_key_metric",
                "item_key": qc.get("metric", ""),
                "item_label": filename,
                "category": qc.get("qc_domain", ""),
                "value": qc.get("value", ""),
                "status_or_role": "",
                "detail": qc.get("note", ""),
            }
        )
        frames.append(add_source(qc_out, filename, "qc_key_metrics"))

    source_map_rows = [
        ("organized_table_1_population_exposure_signal.csv", "cohort + exposure + disproportionality + credibility", "main text candidate"),
        ("organized_table_2_comparative_models.csv", "active comparator + adjusted models + sensitivity", "main/supplement"),
        ("organized_table_3_phenotype_support.csv", "phenotype distribution + adjusted phenotype + clusters", "supporting evidence"),
        ("organized_table_4_serious_outcomes_ml.csv", "serious outcome proportions + exploratory ML summaries", "supporting/exploratory"),
        ("organized_table_5_definitions_qc_index.csv", "definitions + dictionary + evidence/QC index", "traceability"),
    ]
    source_map = pd.DataFrame(
        source_map_rows,
        columns=["item_key", "item_label", "status_or_role"],
    )
    source_map.insert(0, "section", "organized_output_index")
    source_map["category"] = "organized_table"
    source_map["value"] = ""
    source_map["detail"] = "Generated by src/15_organize_outputs_into_five_tables.py"
    frames.append(add_source(source_map, "generated_index", "organized_output_index"))

    return pd.concat(frames, ignore_index=True)


def write_excel(tables: dict[str, pd.DataFrame]) -> Path | None:
    path = OUT_DIR / "organized_five_tables.xlsx"
    try:
        import openpyxl  # noqa: F401

        engine = "openpyxl"
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401

            engine = "xlsxwriter"
        except ImportError:
            return None

    with pd.ExcelWriter(path, engine=engine) as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return path


def write_readme(paths: list[Path], workbook: Path | None) -> Path:
    readme = OUT_DIR / "README_整理说明.md"
    lines = [
        "# outputs 结果整理版（五张表以内）",
        "",
        "这批文件是从 `outputs/tables`、`outputs/qc` 和 `outputs/cluster_serious_ml_1552671/tables` 重新归类生成的。",
        "原始结果没有删除；这里只是把论文/汇报最常用的证据链压缩成 5 张表。",
        "",
        "注意：所有比例都是 FAERS 报告中的报告比例，不是临床发生率；所有信号结果都不能写成因果结论。",
        "",
    ]
    if workbook is None:
        lines.append("- Excel 总表：本机当前 Python 环境缺少 openpyxl/xlsxwriter，本次未生成。")
    else:
        lines.append(f"- Excel 总表：`{workbook.name}`")
    for path in paths:
        lines.append(f"- CSV：`{path.name}`")
    lines.extend(
        [
            "",
            "## 五张表怎么用",
            "",
            "1. `organized_table_1_population_exposure_signal.csv`：人群基线、药物暴露、strict_fall 主信号和可信度分级。",
            "2. `organized_table_2_comparative_models.csv`：主动比较、调整模型和敏感性分析，主要看 ROR/OR 方向是否一致。",
            "3. `organized_table_3_phenotype_support.csv`：表型分布、表型调整模型和跌倒报告聚类，用来支持机制解释。",
            "4. `organized_table_4_serious_outcomes_ml.csv`：strict_fall 报告中的严重结局比例，以及探索性 ML 的性能和重要特征。",
            "5. `organized_table_5_definitions_qc_index.csv`：结局/表型定义、药物词典、证据打分细节和关键 QC 指标，主要用于追溯。",
        ]
    )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "table1_population_signal": make_table_1_population_exposure_signal(),
        "table2_comparative_models": make_table_2_comparative_models(),
        "table3_phenotype_support": make_table_3_phenotype_support(),
        "table4_serious_ml": make_table_4_serious_outcomes_ml(),
        "table5_definitions_qc": make_table_5_definitions_qc_index(),
    }
    csv_paths = [
        write_table(tables["table1_population_signal"], "organized_table_1_population_exposure_signal.csv"),
        write_table(tables["table2_comparative_models"], "organized_table_2_comparative_models.csv"),
        write_table(tables["table3_phenotype_support"], "organized_table_3_phenotype_support.csv"),
        write_table(tables["table4_serious_ml"], "organized_table_4_serious_outcomes_ml.csv"),
        write_table(tables["table5_definitions_qc"], "organized_table_5_definitions_qc_index.csv"),
    ]
    workbook = write_excel(tables)
    readme = write_readme(csv_paths, workbook)

    print(f"Wrote {len(csv_paths)} CSV files")
    for path in csv_paths:
        print(path.relative_to(ROOT))
    if workbook is None:
        print("Excel workbook skipped: openpyxl/xlsxwriter is not installed")
    else:
        print(workbook.relative_to(ROOT))
    print(readme.relative_to(ROOT))


if __name__ == "__main__":
    main()
