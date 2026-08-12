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
from scipy.stats import chi2_contingency
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_CASE_LABELS_OUT = PROJECT_DIR / "outputs" / "intermediate" / "10_phenotype_fingerprint_case_labels.parquet"
DEFAULT_PROFILE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_4_phenotype_fingerprint_by_drug_group.csv"
DEFAULT_PRIMARY_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s6_primary_phenotype_distribution.csv"
DEFAULT_CHISQ_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s7_phenotype_chi_square_tests.csv"
DEFAULT_LOGIT_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s8_phenotype_adjusted_logistic_models.csv"
DEFAULT_DRUG_PROFILE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s9_drug_level_phenotype_fingerprint.csv"
DEFAULT_DRUG_CONTRAST_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s10_drug_level_phenotype_crude_contrasts.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "10_phenotype_fingerprint_analysis_qc.csv"
DEFAULT_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_4_phenotype_fingerprint_heatmap.png"

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
    "melatonin",
]
DRUG_LABELS = {key: key for key in DRUG_KEYS}

BASE_COLUMNS = [
    "caseid",
    "analysis_eligible_main",
    "strict_fall",
    "year",
    "quarter",
    "age_group_3",
    "sex_clean",
    "country_group",
    "rept_cod",
    "e_sub",
    "exposure_zolpidem_ps_ss",
    "exposure_z_drug_ps_ss",
    "exposure_other_z_drug_ps_ss",
    "exposure_benzodiazepine_ps_ss",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
    "pheno_sedation",
    "pheno_neurocognitive",
    "pheno_dizziness_syncope",
    "pheno_gait_balance",
    "pheno_hypotension",
    "pheno_visual_disturbance",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
    "polypharmacy",
    "polypharmacy_5",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
    "serious_any",
    "serious_death",
    "serious_hospitalization",
    "serious_disability",
    "serious_life_threatening",
] + [f"exposure_{key}_ps_ss" for key in DRUG_KEYS] + [f"exposure_{key}_ps_only" for key in DRUG_KEYS]

CATEGORICAL_COVARIATES = [
    "age_group_3",
    "sex_clean",
    "year",
    "quarter",
    "country_group",
    "rept_cod",
    "e_sub",
]
BINARY_FULL_COVARIATES = [
    "cns_polypharmacy_marker",
    "polypharmacy",
    "polypharmacy_5",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
    "comparison_ps_only_marker",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_label: str
    categorical_covariates: tuple[str, ...]
    binary_covariates: tuple[str, ...]

DRUG_GROUP_ORDER = [
    "zolpidem_only",
    "other_z_drugs_without_zolpidem_only",
    "benzodiazepines_only",
]
DRUG_GROUP_LABELS = {
    "zolpidem_only": "Zolpidem only",
    "other_z_drugs_without_zolpidem_only": "Other Z-drugs only",
    "benzodiazepines_only": "Benzodiazepines only",
}

PHENOTYPE_COMPONENTS = [
    ("phenotype_sedation", "Sedation/somnolence"),
    ("phenotype_neurocognitive", "Neurocognitive/consciousness"),
    ("phenotype_dizziness_syncope_hypotension", "Dizziness/syncope/hypotension"),
    ("phenotype_gait_balance", "Gait/balance"),
    ("phenotype_visual_disturbance", "Visual disturbance"),
]

PRIMARY_PHENOTYPE_ORDER = [
    "sedation_only",
    "neurocognitive_only",
    "dizziness_syncope_hypotension_only",
    "gait_balance_only",
    "visual_disturbance_only",
    "mixed_phenotype",
    "no_mechanistic_co_phenotype",
]
PRIMARY_PHENOTYPE_LABELS = {
    "sedation_only": "Sedation/somnolence only",
    "neurocognitive_only": "Neurocognitive/consciousness only",
    "dizziness_syncope_hypotension_only": "Dizziness/syncope/hypotension only",
    "gait_balance_only": "Gait/balance only",
    "visual_disturbance_only": "Visual disturbance only",
    "mixed_phenotype": "Mixed phenotype",
    "no_mechanistic_co_phenotype": "No mechanistic co-phenotype",
}


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def read_main_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")
    available = pq.ParquetFile(path).schema.names
    required_columns = list(dict.fromkeys(BASE_COLUMNS))
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"Main analysis dataset is missing required columns: {missing}")
    return pd.read_parquet(path, columns=required_columns)


def assign_drug_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eligible = safe_bool(df["analysis_eligible_main"]) & safe_bool(df["strict_fall"])
    one_group = pd.to_numeric(df["n_sedative_hypnotic_groups_ps_ss"], errors="coerce").fillna(0).eq(1)
    one_drug = pd.to_numeric(df["n_sedative_hypnotic_drugs_ps_ss"], errors="coerce").fillna(0).eq(1)

    masks = {
        "zolpidem_only": eligible & safe_bool(df["exposure_zolpidem_ps_ss"]) & one_drug,
        "other_z_drugs_without_zolpidem_only": (
            eligible
            & safe_bool(df["exposure_z_drug_ps_ss"])
            & ~safe_bool(df["exposure_zolpidem_ps_ss"])
            & one_group
        ),
        "benzodiazepines_only": eligible & safe_bool(df["exposure_benzodiazepine_ps_ss"]) & one_group,
    }

    df["fingerprint_drug_group"] = pd.NA
    for group in DRUG_GROUP_ORDER:
        df.loc[masks[group], "fingerprint_drug_group"] = group
    return df[df["fingerprint_drug_group"].notna()].copy()


def assign_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phenotype_sedation"] = safe_bool(df["pheno_sedation"])
    df["phenotype_neurocognitive"] = safe_bool(df["pheno_neurocognitive"])
    df["phenotype_dizziness_syncope_hypotension"] = (
        safe_bool(df["pheno_dizziness_syncope"]) | safe_bool(df["pheno_hypotension"])
    )
    df["phenotype_gait_balance"] = safe_bool(df["pheno_gait_balance"])
    df["phenotype_visual_disturbance"] = safe_bool(df["pheno_visual_disturbance"])
    df["cns_polypharmacy_marker"] = (
        safe_bool(df["is_antidepressant"])
        | safe_bool(df["is_antipsychotic"])
        | safe_bool(df["is_opioid"])
        | safe_bool(df["is_antiepileptic"])
    )

    component_columns = [column for column, _ in PHENOTYPE_COMPONENTS]
    df["phenotype_component_count"] = df[component_columns].sum(axis=1).astype("int8")

    df["primary_phenotype"] = "no_mechanistic_co_phenotype"
    single = df["phenotype_component_count"].eq(1)
    df.loc[single & df["phenotype_sedation"], "primary_phenotype"] = "sedation_only"
    df.loc[single & df["phenotype_neurocognitive"], "primary_phenotype"] = "neurocognitive_only"
    df.loc[
        single & df["phenotype_dizziness_syncope_hypotension"],
        "primary_phenotype",
    ] = "dizziness_syncope_hypotension_only"
    df.loc[single & df["phenotype_gait_balance"], "primary_phenotype"] = "gait_balance_only"
    df.loc[single & df["phenotype_visual_disturbance"], "primary_phenotype"] = "visual_disturbance_only"
    df.loc[df["phenotype_component_count"].gt(1), "primary_phenotype"] = "mixed_phenotype"
    return df


def percent(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator * 100 if denominator else np.nan


def build_models() -> list[ModelSpec]:
    return [
        ModelSpec("crude", "Crude", (), ()),
        ModelSpec("model_1_demographic_time", "Model 1: demographic and time", ("age_group_3", "sex_clean", "year", "quarter"), ()),
        ModelSpec(
            "model_2_reporting",
            "Model 2: plus reporting source",
            ("age_group_3", "sex_clean", "year", "quarter", "country_group", "rept_cod", "e_sub"),
            (),
        ),
        ModelSpec(
            "model_3_full",
            "Model 3: plus medication, indication, and PS-only marker",
            ("age_group_3", "sex_clean", "year", "quarter", "country_group", "rept_cod", "e_sub"),
            tuple(BINARY_FULL_COVARIATES),
        ),
    ]


def prepare_model_frame(df: pd.DataFrame, outcome: str, exposure: pd.Series, comparison_ps_only: pd.Series) -> pd.DataFrame:
    columns = list(dict.fromkeys([outcome, *CATEGORICAL_COVARIATES, *BINARY_FULL_COVARIATES]))
    model_df = df.copy()
    model_df["outcome_int"] = safe_bool(model_df[outcome]).astype(int)
    model_df["exposure_int"] = exposure.astype(int).to_numpy()
    model_df["comparison_ps_only_marker"] = safe_bool(comparison_ps_only).astype(int).to_numpy()

    for column in CATEGORICAL_COVARIATES:
        model_df[column] = model_df[column].astype("object").where(model_df[column].notna(), "missing").astype(str)
    for column in BINARY_FULL_COVARIATES:
        model_df[column] = safe_bool(model_df[column]).astype(int)
    return model_df[["outcome_int", "exposure_int", *columns[1:]]].copy()


def usable_terms(model_df: pd.DataFrame, model: ModelSpec) -> tuple[list[str], list[str]]:
    terms: list[str] = []
    skipped: list[str] = []
    for column in model.categorical_covariates:
        if model_df[column].nunique(dropna=False) >= 2:
            terms.append(f"C({column})")
        else:
            skipped.append(column)
    for column in model.binary_covariates:
        if model_df[column].nunique(dropna=False) >= 2:
            terms.append(column)
        else:
            skipped.append(column)
    return terms, skipped


def model_direction(or_low: float, or_high: float) -> str:
    if or_low > 1:
        return "exposure_higher"
    if or_high < 1:
        return "exposure_lower"
    return "not_clearly_different"


def exp_allow_extreme(value: float) -> float:
    if value > 709:
        return np.inf
    if value < -745:
        return 0.0
    return math.exp(value)


def fit_binary_logit(model_df: pd.DataFrame, model: ModelSpec) -> dict[str, object]:
    terms, skipped = usable_terms(model_df, model)
    formula = "outcome_int ~ exposure_int"
    if terms:
        formula += " + " + " + ".join(terms)

    result: dict[str, object] = {
        "model_id": model.model_id,
        "model_label": model.model_label,
        "formula": formula,
        "covariates_used": ";".join(terms),
        "covariates_skipped": ";".join(skipped),
        "odds_ratio": np.nan,
        "ci95_lower": np.nan,
        "ci95_upper": np.nan,
        "p_value": np.nan,
        "nobs": len(model_df),
        "model_converged": False,
        "fit_status": "not_fit",
        "direction": "not_available",
    }

    if model_df["outcome_int"].nunique(dropna=False) < 2 or model_df["exposure_int"].nunique(dropna=False) < 2:
        result["fit_status"] = "skipped:no_outcome_or_exposure_variation"
        return result

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            warnings.simplefilter("always", PerfectSeparationWarning)
            fitted = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial()).fit(maxiter=200)

        coef = float(fitted.params["exposure_int"])
        se = float(fitted.bse["exposure_int"])
        or_value = exp_allow_extreme(coef)
        or_low = exp_allow_extreme(coef - 1.96 * se)
        or_high = exp_allow_extreme(coef + 1.96 * se)
        warning_names = sorted({warning.category.__name__ for warning in caught})
        unstable_extreme = not np.isfinite([or_value, or_low, or_high]).all() or max(abs(coef), abs(se)) > 20
        if unstable_extreme:
            warning_names.append("extreme_coefficient")
            or_value = np.nan
            or_low = np.nan
            or_high = np.nan
        result.update(
            {
                "odds_ratio": or_value,
                "ci95_lower": or_low,
                "ci95_upper": or_high,
                "p_value": np.nan if unstable_extreme else float(fitted.pvalues["exposure_int"]),
                "model_converged": bool(getattr(fitted, "converged", False)),
                "fit_status": "ok" if not warning_names else "ok_with_warning:" + ";".join(warning_names),
                "direction": "not_available" if unstable_extreme else model_direction(or_low, or_high),
            }
        )
    except Exception as exc:
        result["fit_status"] = f"failed:{type(exc).__name__}:{exc}"
    return result


def bh_fdr(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return adjusted
    m = len(valid)
    ranked = valid.rename("p_value").reset_index()
    ranked["rank"] = np.arange(1, m + 1)
    ranked["raw_adjusted"] = ranked["p_value"] * m / ranked["rank"]
    ranked["p_fdr_bh"] = ranked["raw_adjusted"][::-1].cummin()[::-1].clip(upper=1.0)
    adjusted.loc[ranked["index"]] = ranked["p_fdr_bh"].to_numpy()
    return adjusted


def build_component_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in DRUG_GROUP_ORDER:
        group_df = df[df["fingerprint_drug_group"].eq(group)]
        total = len(group_df)
        for column, label in PHENOTYPE_COMPONENTS:
            n = int(group_df[column].sum())
            rows.append(
                {
                    "drug_group": group,
                    "drug_group_label": DRUG_GROUP_LABELS[group],
                    "phenotype_component": column,
                    "phenotype_component_label": label,
                    "fall_case_n": total,
                    "phenotype_n": n,
                    "phenotype_percent": percent(n, total),
                }
            )
        mixed_n = int(group_df["primary_phenotype"].eq("mixed_phenotype").sum())
        none_n = int(group_df["primary_phenotype"].eq("no_mechanistic_co_phenotype").sum())
        rows.extend(
            [
                {
                    "drug_group": group,
                    "drug_group_label": DRUG_GROUP_LABELS[group],
                    "phenotype_component": "mixed_phenotype",
                    "phenotype_component_label": "Mixed phenotype",
                    "fall_case_n": total,
                    "phenotype_n": mixed_n,
                    "phenotype_percent": percent(mixed_n, total),
                },
                {
                    "drug_group": group,
                    "drug_group_label": DRUG_GROUP_LABELS[group],
                    "phenotype_component": "no_mechanistic_co_phenotype",
                    "phenotype_component_label": "No mechanistic co-phenotype",
                    "fall_case_n": total,
                    "phenotype_n": none_n,
                    "phenotype_percent": percent(none_n, total),
                },
            ]
        )
    return pd.DataFrame(rows)


def build_primary_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in DRUG_GROUP_ORDER:
        group_df = df[df["fingerprint_drug_group"].eq(group)]
        total = len(group_df)
        counts = group_df["primary_phenotype"].value_counts()
        for phenotype in PRIMARY_PHENOTYPE_ORDER:
            n = int(counts.get(phenotype, 0))
            rows.append(
                {
                    "drug_group": group,
                    "drug_group_label": DRUG_GROUP_LABELS[group],
                    "primary_phenotype": phenotype,
                    "primary_phenotype_label": PRIMARY_PHENOTYPE_LABELS[phenotype],
                    "fall_case_n": total,
                    "phenotype_n": n,
                    "phenotype_percent": percent(n, total),
                }
            )
    return pd.DataFrame(rows)


def chi_square_from_table(table: pd.DataFrame) -> tuple[float, float, int]:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, 0
    chi2, p_value, dof, _ = chi2_contingency(table)
    return chi2, p_value, dof


def build_chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in PHENOTYPE_COMPONENTS:
        table = pd.crosstab(df["fingerprint_drug_group"], df[column])
        table = table.reindex(index=DRUG_GROUP_ORDER, columns=[False, True], fill_value=0)
        chi2, p_value, dof = chi_square_from_table(table)
        rows.append(
            {
                "test_id": f"{column}_by_drug_group",
                "test_label": f"{label} by drug group",
                "table_type": "3x2",
                "chi_square": chi2,
                "df": dof,
                "p_value": p_value,
            }
        )

    primary_table = pd.crosstab(df["fingerprint_drug_group"], df["primary_phenotype"])
    primary_table = primary_table.reindex(index=DRUG_GROUP_ORDER, columns=PRIMARY_PHENOTYPE_ORDER, fill_value=0)
    chi2, p_value, dof = chi_square_from_table(primary_table)
    rows.append(
        {
            "test_id": "primary_phenotype_by_drug_group",
            "test_label": "Primary phenotype distribution by drug group",
            "table_type": "3x7",
            "chi_square": chi2,
            "df": dof,
            "p_value": p_value,
        }
    )
    return pd.DataFrame(rows)


def build_adjusted_logistic_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("Zolpidem only vs benzodiazepines only", "zolpidem_only", "benzodiazepines_only"),
        (
            "Other Z-drugs only vs benzodiazepines only",
            "other_z_drugs_without_zolpidem_only",
            "benzodiazepines_only",
        ),
        ("Zolpidem only vs other Z-drugs only", "zolpidem_only", "other_z_drugs_without_zolpidem_only"),
    ]
    for outcome, label in PHENOTYPE_COMPONENTS:
        for comparison_label, exposure_group, comparator_group in comparisons:
            subset = df[df["fingerprint_drug_group"].isin([exposure_group, comparator_group])].copy()
            exposure = subset["fingerprint_drug_group"].eq(exposure_group)
            comparison_ps_only = pd.Series(False, index=subset.index)
            model_frame = prepare_model_frame(subset, outcome, exposure, comparison_ps_only)
            exposure_rows = model_frame["exposure_int"].eq(1)
            base = {
                "outcome": outcome,
                "outcome_label": label,
                "comparison": comparison_label,
                "exposure_group": exposure_group,
                "comparator_group": comparator_group,
                "analysis_n": len(model_frame),
                "exposure_n": int(exposure_rows.sum()),
                "exposure_phenotype_n": int(model_frame.loc[exposure_rows, "outcome_int"].sum()),
                "exposure_phenotype_percent": percent(
                    int(model_frame.loc[exposure_rows, "outcome_int"].sum()),
                    int(exposure_rows.sum()),
                ),
                "comparator_n": int((~exposure_rows).sum()),
                "comparator_phenotype_n": int(model_frame.loc[~exposure_rows, "outcome_int"].sum()),
                "comparator_phenotype_percent": percent(
                    int(model_frame.loc[~exposure_rows, "outcome_int"].sum()),
                    int((~exposure_rows).sum()),
                ),
            }
            for model in build_models():
                row = dict(base)
                row.update(fit_binary_logit(model_frame, model))
                row["note"] = (
                    "Sequential phenotype model; Model 3 adjusts for age group, sex, year, quarter, country group, "
                    "reporter type, e_sub, CNS co-medication marker, polypharmacy, indications, and PS-only marker "
                    "when estimable."
                )
                rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr_bh_within_model"] = result.groupby(["model_id", "outcome"], group_keys=False)["p_value"].apply(bh_fdr)
    return result


def build_drug_level_profiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = safe_bool(df["analysis_eligible_main"]) & safe_bool(df["strict_fall"])
    single_drug = base & pd.to_numeric(df["n_sedative_hypnotic_drugs_ps_ss"], errors="coerce").fillna(0).eq(1)
    for analysis_set, base_mask in [("all_exposed", base), ("single_drug_clean", single_drug)]:
        for drug in DRUG_KEYS:
            column = f"exposure_{drug}_ps_ss"
            if column not in df.columns:
                continue
            drug_mask = base_mask & safe_bool(df[column])
            drug_df = df.loc[drug_mask]
            total = len(drug_df)
            if total == 0:
                continue
            for phenotype_column, phenotype_label in PHENOTYPE_COMPONENTS:
                n = int(drug_df[phenotype_column].sum())
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "drug": drug,
                        "drug_label": DRUG_LABELS.get(drug, drug),
                        "fall_case_n": total,
                        "phenotype_component": phenotype_column,
                        "phenotype_component_label": phenotype_label,
                        "phenotype_n": n,
                        "phenotype_percent": percent(n, total),
                    }
                )
            mixed_n = int(drug_df["primary_phenotype"].eq("mixed_phenotype").sum())
            none_n = int(drug_df["primary_phenotype"].eq("no_mechanistic_co_phenotype").sum())
            rows.extend(
                [
                    {
                        "analysis_set": analysis_set,
                        "drug": drug,
                        "drug_label": DRUG_LABELS.get(drug, drug),
                        "fall_case_n": total,
                        "phenotype_component": "mixed_phenotype",
                        "phenotype_component_label": "Mixed phenotype",
                        "phenotype_n": mixed_n,
                        "phenotype_percent": percent(mixed_n, total),
                    },
                    {
                        "analysis_set": analysis_set,
                        "drug": drug,
                        "drug_label": DRUG_LABELS.get(drug, drug),
                        "fall_case_n": total,
                        "phenotype_component": "no_mechanistic_co_phenotype",
                        "phenotype_component_label": "No mechanistic co-phenotype",
                        "phenotype_n": none_n,
                        "phenotype_percent": percent(none_n, total),
                    },
                ]
            )
    result = pd.DataFrame(rows)
    return result.sort_values(["analysis_set", "fall_case_n", "drug", "phenotype_component"], ascending=[True, False, True, True])


def crude_or_with_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    # Haldane-Anscombe correction keeps sparse drug-phenotype cells usable for screening tables.
    ah, bh, ch, dh = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    odds_ratio = (ah * dh) / (bh * ch)
    se = math.sqrt(1 / ah + 1 / bh + 1 / ch + 1 / dh)
    log_or = math.log(odds_ratio)
    return odds_ratio, math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)


def build_drug_level_crude_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = safe_bool(df["analysis_eligible_main"]) & safe_bool(df["strict_fall"])
    any_target = pd.Series(False, index=df.index)
    for drug in DRUG_KEYS:
        column = f"exposure_{drug}_ps_ss"
        if column in df.columns:
            any_target = any_target | safe_bool(df[column])

    base = base & any_target
    single_drug = base & pd.to_numeric(df["n_sedative_hypnotic_drugs_ps_ss"], errors="coerce").fillna(0).eq(1)
    for analysis_set, base_mask in [("single_drug_clean", single_drug), ("all_exposed_screening", base)]:
        for drug in DRUG_KEYS:
            column = f"exposure_{drug}_ps_ss"
            if column not in df.columns:
                continue
            drug_mask = base_mask & safe_bool(df[column])
            comparator_mask = base_mask & ~safe_bool(df[column])
            drug_n = int(drug_mask.sum())
            comparator_n = int(comparator_mask.sum())
            if drug_n == 0 or comparator_n == 0:
                continue
            subset = df.loc[drug_mask | comparator_mask].copy()
            exposure = drug_mask.loc[subset.index]
            ps_only_column = f"exposure_{drug}_ps_only"
            comparison_ps_only = safe_bool(subset[ps_only_column]) if ps_only_column in subset.columns else pd.Series(False, index=subset.index)
            for phenotype_column, phenotype_label in PHENOTYPE_COMPONENTS:
                a = int((drug_mask & safe_bool(df[phenotype_column])).sum())
                b = drug_n - a
                c = int((comparator_mask & safe_bool(df[phenotype_column])).sum())
                d = comparator_n - c
                or_value, ci_low, ci_high = crude_or_with_ci(a, b, c, d)
                model_frame = prepare_model_frame(subset, phenotype_column, exposure, comparison_ps_only)
                base_row = {
                    "analysis_set": analysis_set,
                    "drug": drug,
                    "drug_label": DRUG_LABELS.get(drug, drug),
                    "comparison": f"{drug} vs other target drugs",
                    "phenotype_component": phenotype_column,
                    "phenotype_component_label": phenotype_label,
                    "drug_fall_case_n": drug_n,
                    "drug_phenotype_n": a,
                    "drug_phenotype_percent": percent(a, drug_n),
                    "comparator_fall_case_n": comparator_n,
                    "comparator_phenotype_n": c,
                    "comparator_phenotype_percent": percent(c, comparator_n),
                    "haldane_crude_or": or_value,
                    "haldane_crude_ci95_lower": ci_low,
                    "haldane_crude_ci95_upper": ci_high,
                }
                for model in build_models():
                    row = dict(base_row)
                    row.update(fit_binary_logit(model_frame, model))
                    row["note"] = (
                        "Sequential drug-level phenotype contrast. Prefer single_drug_clean rows for interpretation; "
                        "all_exposed_screening rows can count reports with multiple sedative-hypnotic drugs under the exposed drug."
                    )
                    rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr_bh_within_model"] = result.groupby(["analysis_set", "model_id", "phenotype_component"], group_keys=False)[
            "p_value"
        ].apply(bh_fdr)
    return result.sort_values(
        ["analysis_set", "drug_fall_case_n", "drug", "phenotype_component", "model_id"],
        ascending=[True, False, True, True, True],
    )


def build_qc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "phenotype_fingerprint", "metric": metric, "value": value, "note": note})

    add("analysis_rows", len(df))
    add("duplicate_caseid", int(df["caseid"].duplicated().sum()))
    for group in DRUG_GROUP_ORDER:
        add(f"{group}__strict_fall_rows", int(df["fingerprint_drug_group"].eq(group).sum()))
    for column, _ in PHENOTYPE_COMPONENTS:
        add(f"{column}__true", int(df[column].sum()))
    add("mixed_phenotype_rows", int(df["primary_phenotype"].eq("mixed_phenotype").sum()))
    add(
        "no_mechanistic_co_phenotype_rows",
        int(df["primary_phenotype"].eq("no_mechanistic_co_phenotype").sum()),
    )
    add(
        "injury_consequence_not_available",
        "not_estimated",
        "Current main dataset has no full all-PT REAC list or locked injury PT map; do not infer fracture/head injury phenotype from fall_pt_list.",
    )
    return pd.DataFrame(rows)


def plot_heatmap(profile: pd.DataFrame, output_path: Path) -> None:
    plot_df = profile[
        profile["phenotype_component"].isin([column for column, _ in PHENOTYPE_COMPONENTS])
    ].pivot(index="phenotype_component_label", columns="drug_group_label", values="phenotype_percent")
    plot_df = plot_df[[DRUG_GROUP_LABELS[group] for group in DRUG_GROUP_ORDER]]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(plot_df.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=25, ha="right")
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Phenotype fingerprint among strict fall reports")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.iloc[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Percent of strict fall reports")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--case-labels-out", type=Path, default=DEFAULT_CASE_LABELS_OUT)
    parser.add_argument("--profile-out", type=Path, default=DEFAULT_PROFILE_OUT)
    parser.add_argument("--primary-out", type=Path, default=DEFAULT_PRIMARY_OUT)
    parser.add_argument("--chisq-out", type=Path, default=DEFAULT_CHISQ_OUT)
    parser.add_argument("--logit-out", type=Path, default=DEFAULT_LOGIT_OUT)
    parser.add_argument("--drug-profile-out", type=Path, default=DEFAULT_DRUG_PROFILE_OUT)
    parser.add_argument("--drug-contrast-out", type=Path, default=DEFAULT_DRUG_CONTRAST_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--figure-out", type=Path, default=DEFAULT_FIGURE_OUT)
    args = parser.parse_args()

    df = read_main_dataset(args.main_dataset)
    analysis_df = assign_phenotypes(assign_drug_groups(df))

    keep_columns = [
        "caseid",
        "fingerprint_drug_group",
        "primary_phenotype",
        "phenotype_component_count",
        "year",
        "age_group_3",
        "sex_clean",
        "country_group",
        "rept_cod",
        "cns_polypharmacy_marker",
        "polypharmacy_5",
        "serious_any",
        "serious_death",
        "serious_hospitalization",
        "serious_disability",
        "serious_life_threatening",
    ] + [column for column, _ in PHENOTYPE_COMPONENTS]
    case_labels = analysis_df[keep_columns].copy()

    profile = build_component_profile(analysis_df)
    primary = build_primary_distribution(analysis_df)
    chisq = build_chi_square_tests(analysis_df)
    logit = build_adjusted_logistic_models(analysis_df)
    full_labeled_df = assign_phenotypes(df)
    drug_profile = build_drug_level_profiles(full_labeled_df)
    drug_contrast = build_drug_level_crude_contrasts(full_labeled_df)
    qc = build_qc(analysis_df)

    for path in [
        args.case_labels_out,
        args.profile_out,
        args.primary_out,
        args.chisq_out,
        args.logit_out,
        args.drug_profile_out,
        args.drug_contrast_out,
        args.qc_out,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    case_labels.to_parquet(args.case_labels_out, index=False)
    profile.to_csv(args.profile_out, index=False, encoding="utf-8-sig")
    primary.to_csv(args.primary_out, index=False, encoding="utf-8-sig")
    chisq.to_csv(args.chisq_out, index=False, encoding="utf-8-sig")
    logit.to_csv(args.logit_out, index=False, encoding="utf-8-sig")
    drug_profile.to_csv(args.drug_profile_out, index=False, encoding="utf-8-sig")
    drug_contrast.to_csv(args.drug_contrast_out, index=False, encoding="utf-8-sig")
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    plot_heatmap(profile, args.figure_out)

    print(f"Wrote {args.case_labels_out}")
    print(f"Wrote {args.profile_out}")
    print(f"Wrote {args.primary_out}")
    print(f"Wrote {args.chisq_out}")
    print(f"Wrote {args.logit_out}")
    print(f"Wrote {args.drug_profile_out}")
    print(f"Wrote {args.drug_contrast_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.figure_out}")
    print(f"Phenotype fingerprint rows: {len(analysis_df):,}")
    for group in DRUG_GROUP_ORDER:
        print(f"{DRUG_GROUP_LABELS[group]}: {int(analysis_df['fingerprint_drug_group'].eq(group).sum()):,}")


if __name__ == "__main__":
    main()
