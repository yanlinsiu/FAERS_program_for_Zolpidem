from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2_contingency, norm, spearmanr

from analysis_common import OUTCOME_SPECS, ensure_output_dir, merge_signal_and_feature


DEFAULT_SIGNAL_FILE = Path(
    r"D:\program_FAERS\OUTPUT_GLOBAL\datasets\signal_dataset_2004_2025.parquet"
)
DEFAULT_FEATURE_FILE = Path(
    r"D:\program_FAERS\OUTPUT_GLOBAL\datasets\drug_feature_2004_2025_case.parquet"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\program_FAERS\OUTPUT_GLOBAL\analysis")

AGE_GROUP_ORDER = ["65-74", "75-84", ">=85"]
AGE_GROUP_TO_SCORE = {label: idx for idx, label in enumerate(AGE_GROUP_ORDER)}
COHORT_SPECS = [
    ("primary_ps_ss", "is_zolpidem_suspect"),
    ("sensitivity_ps_only", "is_zolpidem_suspect_ps"),
]
BOOL_COVARIATES = [
    "serious",
    "polypharmacy_5",
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
]


def _prepare_subset(
    df: pd.DataFrame, exposure_col: str, outcome_col: str
) -> pd.DataFrame:
    subset = df[df[exposure_col].fillna(False).astype(bool)].copy()
    subset = subset[subset["age_group"].isin(AGE_GROUP_ORDER)].copy()

    subset["age_group"] = pd.Categorical(
        subset["age_group"], categories=AGE_GROUP_ORDER, ordered=True
    )
    subset["age_order"] = subset["age_group"].map(AGE_GROUP_TO_SCORE).astype(int)
    subset[outcome_col] = subset[outcome_col].fillna(False).astype(int)
    subset["sex_clean"] = (
        subset["sex_clean"]
        .where(subset["sex_clean"].notna(), "unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    subset["year_centered"] = subset["year"].astype(int) - int(subset["year"].min())

    for col in BOOL_COVARIATES:
        if col not in subset.columns:
            subset[col] = False
        subset[col] = subset[col].fillna(False).astype(int)

    return subset


def _build_age_rate_rows(
    subset: pd.DataFrame, analysis_name: str, outcome_name: str
) -> pd.DataFrame:
    grouped = (
        subset.groupby("age_group", observed=True)
        .agg(
            n_cases=("caseid", "size"),
            n_outcome=(outcome_name, "sum"),
        )
        .reindex(AGE_GROUP_ORDER)
        .fillna(0)
        .reset_index()
    )
    grouped["outcome_reporting_rate"] = grouped["n_outcome"] / grouped["n_cases"]
    grouped.insert(0, "analysis", analysis_name)
    grouped.insert(1, "outcome_name", outcome_name)
    return grouped


def _cochran_armitage_test(
    n_cases: np.ndarray, n_outcome: np.ndarray
) -> dict[str, float]:
    scores = np.arange(len(n_cases), dtype=float)
    total_n = float(n_cases.sum())
    total_outcome = float(n_outcome.sum())
    outcome_rate = total_outcome / total_n

    numerator = float(np.sum(scores * (n_outcome - n_cases * outcome_rate)))
    weighted_score_mean = float(np.sum(n_cases * scores) / total_n)
    variance = float(
        outcome_rate
        * (1.0 - outcome_rate)
        * np.sum(n_cases * (scores - weighted_score_mean) ** 2)
    )
    if variance <= 0:
        return {
            "cochran_armitage_z": 0.0,
            "cochran_armitage_p_two_sided": 1.0,
            "cochran_armitage_p_increasing": 1.0,
        }

    z_value = numerator / np.sqrt(variance)
    return {
        "cochran_armitage_z": float(z_value),
        "cochran_armitage_p_two_sided": float(2.0 * norm.sf(abs(z_value))),
        "cochran_armitage_p_increasing": float(norm.sf(z_value)),
    }


def _build_test_row(
    subset: pd.DataFrame,
    analysis_name: str,
    outcome_name: str,
    outcome_label: str,
    exposure_col: str,
) -> dict[str, object]:
    contingency = pd.crosstab(
        subset["age_group"], subset[outcome_name].astype(int)
    ).reindex(AGE_GROUP_ORDER, fill_value=0)
    contingency = contingency.reindex(columns=[0, 1], fill_value=0)
    chi2_stat, chi2_p_value, dof, _ = chi2_contingency(contingency.to_numpy())

    spearman_result = spearmanr(subset["age_order"], subset[outcome_name].astype(int))
    trend_stats = _cochran_armitage_test(
        contingency.sum(axis=1).to_numpy(dtype=float),
        contingency[1].to_numpy(dtype=float),
    )

    return {
        "analysis": analysis_name,
        "exposure_col": exposure_col,
        "outcome_name": outcome_name,
        "outcome_definition": outcome_label,
        "n_cases": int(len(subset)),
        "n_outcome": int(subset[outcome_name].sum()),
        "pearson_chi_square": float(chi2_stat),
        "pearson_chi_square_df": int(dof),
        "pearson_chi_square_p_value": float(chi2_p_value),
        "spearman_rho": float(spearman_result.statistic),
        "spearman_p_value": float(spearman_result.pvalue),
        **trend_stats,
    }


def _build_logistic_design_matrix(subset: pd.DataFrame) -> pd.DataFrame:
    design = pd.DataFrame(index=subset.index)
    design["age_order"] = subset["age_order"].astype(float)
    design["year_centered"] = subset["year_centered"].astype(float)
    design["sex_male"] = subset["sex_clean"].eq("M").astype(float)
    design["sex_unknown"] = (~subset["sex_clean"].isin(["F", "M"])).astype(float)

    for col in BOOL_COVARIATES:
        design[col] = subset[col].astype(float)

    non_constant_cols = [
        col for col in design.columns if design[col].nunique(dropna=False) > 1
    ]
    return design[non_constant_cols].copy()


def _fit_logistic_with_inference(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, dict[str, object]]:
    X_matrix = np.column_stack([np.ones(len(X), dtype=float), X.to_numpy(dtype=float)])
    y_array = y.astype(int).to_numpy(dtype=float)
    feature_names = ["intercept", *X.columns.tolist()]

    def objective(beta: np.ndarray) -> float:
        probabilities = np.clip(expit(X_matrix @ beta), 1e-9, 1.0 - 1e-9)
        return float(
            -np.sum(
                y_array * np.log(probabilities)
                + (1.0 - y_array) * np.log(1.0 - probabilities)
            )
        )

    def gradient(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(X_matrix @ beta)
        return X_matrix.T @ (probabilities - y_array)

    def hessian_fn(beta: np.ndarray) -> np.ndarray:
        probabilities = expit(X_matrix @ beta)
        weights = probabilities * (1.0 - probabilities)
        return X_matrix.T @ (X_matrix * weights[:, None])

    result = minimize(
        objective,
        x0=np.zeros(X_matrix.shape[1], dtype=float),
        jac=gradient,
        hess=hessian_fn,
        method="Newton-CG",
        options={"xtol": 1e-8, "maxiter": 200},
    )

    beta = result.x
    hessian = hessian_fn(beta)
    gradient_at_optimum = gradient(beta)
    covariance = np.linalg.pinv(hessian)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), a_min=0.0, a_max=None))
    z_values = np.divide(
        beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0
    )
    p_values = 2.0 * norm.sf(np.abs(z_values))

    def safe_exp(values: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(values, -50.0, 50.0))

    model_df = pd.DataFrame(
        {
            "term": feature_names,
            "coefficient": beta,
            "std_error": standard_errors,
            "z_value": z_values,
            "p_value": p_values,
            "odds_ratio": safe_exp(beta),
            "ci_low": safe_exp(beta - 1.96 * standard_errors),
            "ci_high": safe_exp(beta + 1.96 * standard_errors),
        }
    )

    diagnostics = {
        "optimization_success": bool(
            result.success or np.max(np.abs(gradient_at_optimum)) < 1e-5
        ),
        "optimization_message": str(result.message),
        "n_iterations": int(getattr(result, "nit", 0)),
        "negative_log_likelihood": float(result.fun),
        "gradient_inf_norm": float(np.max(np.abs(gradient_at_optimum))),
    }
    return model_df, diagnostics


def build_age_trend_analysis(
    signal_file: str | Path = DEFAULT_SIGNAL_FILE,
    feature_file: str | Path = DEFAULT_FEATURE_FILE,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_df = merge_signal_and_feature(
        signal_file=signal_file,
        feature_file=feature_file,
    )

    rate_frames: list[pd.DataFrame] = []
    test_rows: list[dict[str, object]] = []
    logistic_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]
        outcome_col = outcome_spec["outcome_col"]
        outcome_label = outcome_spec["outcome_label"]
        if outcome_col not in merged_df.columns:
            continue

        for analysis_name, exposure_col in COHORT_SPECS:
            if exposure_col not in merged_df.columns:
                continue

            exposed_df = merged_df[
                merged_df[exposure_col].fillna(False).astype(bool)
            ].copy()
            subset = _prepare_subset(merged_df, exposure_col, outcome_col)
            if subset.empty:
                continue

            analysis_df = subset.rename(columns={outcome_col: outcome_name}).copy()

            rate_frames.append(
                _build_age_rate_rows(analysis_df, analysis_name, outcome_name)
            )
            test_rows.append(
                _build_test_row(
                    analysis_df,
                    analysis_name=analysis_name,
                    outcome_name=outcome_name,
                    outcome_label=outcome_label,
                    exposure_col=exposure_col,
                )
            )

            logistic_df, diagnostics = _fit_logistic_with_inference(
                _build_logistic_design_matrix(analysis_df),
                analysis_df[outcome_name],
            )
            logistic_df.insert(0, "analysis", analysis_name)
            logistic_df.insert(1, "outcome_name", outcome_name)
            logistic_df.insert(2, "outcome_definition", outcome_label)
            logistic_df["n_cases"] = int(len(analysis_df))
            logistic_df["n_outcome"] = int(analysis_df[outcome_name].sum())
            logistic_df["optimization_success"] = diagnostics["optimization_success"]
            logistic_df["optimization_message"] = diagnostics["optimization_message"]
            logistic_df["n_iterations"] = diagnostics["n_iterations"]
            logistic_df["negative_log_likelihood"] = diagnostics[
                "negative_log_likelihood"
            ]
            logistic_df["gradient_inf_norm"] = diagnostics["gradient_inf_norm"]
            logistic_frames.append(logistic_df)

            age_counts = (
                analysis_df.groupby("age_group", observed=True)
                .size()
                .reindex(AGE_GROUP_ORDER)
                .fillna(0)
            )
            qc_rows.append(
                {
                    "analysis": analysis_name,
                    "outcome_name": outcome_name,
                    "outcome_definition": outcome_label,
                    "n_exposed_total": int(len(exposed_df)),
                    "n_in_age_analysis": int(len(analysis_df)),
                    "n_excluded_non_target_age": int(
                        len(exposed_df) - len(analysis_df)
                    ),
                    "n_outcome": int(analysis_df[outcome_name].sum()),
                    "n_age_65_74": int(age_counts.get("65-74", 0)),
                    "n_age_75_84": int(age_counts.get("75-84", 0)),
                    "n_age_gte_85": int(age_counts.get(">=85", 0)),
                }
            )

    rates_df = (
        pd.concat(rate_frames, ignore_index=True) if rate_frames else pd.DataFrame()
    )
    tests_df = pd.DataFrame(test_rows)
    logistic_df = (
        pd.concat(logistic_frames, ignore_index=True)
        if logistic_frames
        else pd.DataFrame()
    )
    qc_df = pd.DataFrame(qc_rows)

    output_root = ensure_output_dir(output_dir)
    rates_df.to_csv(
        output_root / "04_age_trend_analysis_rates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tests_df.to_csv(
        output_root / "04_age_trend_analysis_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )
    logistic_df.to_csv(
        output_root / "04_age_trend_analysis_logistic.csv",
        index=False,
        encoding="utf-8-sig",
    )
    qc_df.to_csv(
        output_root / "04_age_trend_analysis_qc.csv", index=False, encoding="utf-8-sig"
    )

    return rates_df, tests_df, logistic_df, qc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Age trend analysis for fall outcomes among zolpidem-exposed cases."
    )
    parser.add_argument(
        "--signal-file",
        default=str(DEFAULT_SIGNAL_FILE),
        help="Signal dataset parquet file.",
    )
    parser.add_argument(
        "--feature-file",
        default=str(DEFAULT_FEATURE_FILE),
        help="Feature dataset parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save age trend outputs.",
    )
    args = parser.parse_args()

    rates, tests, logistic_results, qc = build_age_trend_analysis(
        signal_file=args.signal_file,
        feature_file=args.feature_file,
        output_dir=args.output_dir,
    )
    print("saved age rate rows:", len(rates))
    print("saved age test rows:", len(tests))
    print("saved logistic rows:", len(logistic_results))
    print("saved QC rows:", len(qc))
