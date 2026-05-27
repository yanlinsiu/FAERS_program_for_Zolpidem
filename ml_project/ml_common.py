from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import (
    DatasetBundle,
    extract_token,
    resolve_signal_feature_bundle,
    token_sort_key,
)

GLOBAL_DATASET_DIR = Path(
    os.environ.get("FAERS_GLOBAL_DATASET_DIR", PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets")
)
OUTPUT_ML_ROOT = Path(os.environ.get("FAERS_ML_OUTPUT_ROOT", PROJECT_ROOT / "OUTPUT_ML"))
FEATURE_V2_DATASET_DIR = OUTPUT_ML_ROOT / "features_v2" / "datasets"

TARGET_OPTIONS = ("is_fall_narrow", "serious")
SEARCH_MODES = ("none", "fast", "full")
COHORT_OPTIONS = ("all", "zolpidem", "zdrug")
FEATURE_VERSION_OPTIONS = ("v1", "v2")

V1_BOOL_FEATURES = [
    "is_zolpidem",
    "is_zaleplon",
    "is_zopiclone",
    "is_eszopiclone",
    "is_benzo",
    "is_antidepressant",
    "is_antipsychotic",
    "is_opioid",
    "is_antiepileptic",
    "polypharmacy_5",
    "is_other_zdrug",
    "multiple_zdrug",
    "any_cns_coprescription",
    "high_drug_burden_10",
    "very_high_drug_burden_20",
]

V1_NUMERIC_FEATURES = [
    "year",
    "drug_n",
    "distinct_drug_n",
    "log_drug_n",
    "log_distinct_drug_n",
    "zdrug_count",
    "cns_coprescription_count",
]

V1_CATEGORICAL_FEATURES = [
    "age_group",
    "sex_clean",
    "quarter",
    "drug_n_bucket",
    "distinct_drug_n_bucket",
    "cns_coprescription_bucket",
]

V2_BASE_BOOL_FEATURES = [
    "event_date_known",
    "has_ps_drug",
    "has_ss_drug",
    "zolpidem_as_ps",
    "zolpidem_as_suspect",
    "other_zdrug_as_suspect",
    "indi_insomnia",
    "indi_anxiety",
    "indi_depression",
    "indi_pain",
    "indi_epilepsy",
    "indi_dizziness_vertigo",
    "has_rpsr",
    "has_start_dt",
    "has_end_dt",
    "duration_known",
]

V2_BASE_NUMERIC_FEATURES = [
    "age_years",
    "ps_drug_n",
    "ss_drug_n",
    "concomitant_drug_n",
    "interacting_drug_n",
    "indi_n",
    "distinct_indi_n",
    "indi_mapped_n",
    "indi_unmapped_n",
    "therapy_record_n",
]

V2_BASE_CATEGORICAL_FEATURES = [
    "rept_cod",
    "e_sub",
    "reporter_country",
    "occr_country",
    "rpsr_cod",
]

BOOL_FEATURES = V1_BOOL_FEATURES.copy()
NUMERIC_FEATURES = V1_NUMERIC_FEATURES.copy()
CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES.copy()
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES

SEARCH_SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "neg_brier_score": "neg_brier_score",
}
REFIT_METRIC = "average_precision"
EVALUATION_METRICS = [
    "roc_auc",
    "average_precision",
    "brier_score",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
]


@dataclass(frozen=True)
class ExperimentConfig:
    period_token: str | None
    feature_version: str
    target_col: str
    cohort: str
    train_end_year: int
    valid_year: int
    test_year: int
    train_sample_n: int
    search_mode: str
    cv_folds: int
    bootstrap_iterations: int
    random_state: int


@dataclass(frozen=True)
class SearchSpec:
    strategy: Literal["grid", "random"]
    param_space_by_mode: dict[str, dict[str, list[Any]] | list[dict[str, list[Any]]]]
    n_iter_by_mode: dict[str, int] | None = None


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    bundle: DatasetBundle
    run_dir: Path
    train_full_df: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    pipeline: Pipeline
    search_summary: dict[str, Any]
    search_results_df: pd.DataFrame | None
    cv_metrics_df: pd.DataFrame
    cv_summary: dict[str, Any]
    threshold_selection: dict[str, float]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    validation_metrics_raw: dict[str, Any]
    test_metrics_raw: dict[str, Any]
    valid_raw_scores: np.ndarray
    valid_scores: np.ndarray
    test_raw_scores: np.ndarray
    test_scores: np.ndarray


def log_step(message: str) -> None:
    print(f"[ml] {message}", flush=True)


def add_common_arguments(
    parser: Any,
    *,
    default_train_sample_n: int,
    default_search_mode: str,
) -> None:
    parser.add_argument(
        "--period-token",
        default=None,
        help="Dataset token such as 2004_2025. Defaults to the latest available bundle.",
    )
    parser.add_argument(
        "--target-col",
        default="is_fall_narrow",
        choices=TARGET_OPTIONS,
        help="Target column to predict.",
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        choices=FEATURE_VERSION_OPTIONS,
        help="Feature table version. v1 uses current global datasets; v2 uses OUTPUT_ML/features_v2/datasets.",
    )
    parser.add_argument(
        "--cohort",
        default="all",
        choices=COHORT_OPTIONS,
        help=(
            "Study population. all keeps all eligible elderly cases; "
            "zolpidem keeps zolpidem-exposed cases; zdrug keeps any Z-drug-exposed cases."
        ),
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2023,
        help="Use all cases up to this year for model training.",
    )
    parser.add_argument(
        "--valid-year",
        type=int,
        default=2024,
        help="Validation year used for calibration and threshold selection.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        help="Holdout test year used only for final evaluation.",
    )
    parser.add_argument(
        "--train-sample-n",
        type=int,
        default=default_train_sample_n,
        help="Optional stratified training sample size. Use 0 to keep the full training set.",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default=default_search_mode,
        help="Hyperparameter search depth. none skips tuning, fast is a small search, full is the full configured search.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds used inside the training period.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations used for final test-set confidence intervals.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling, tuning, and model fitting.",
    )


def config_from_args(args: Any) -> ExperimentConfig:
    return ExperimentConfig(
        period_token=args.period_token,
        feature_version=args.feature_version,
        target_col=args.target_col,
        cohort=args.cohort,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
        train_sample_n=args.train_sample_n,
        search_mode=args.search_mode,
        cv_folds=args.cv_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )


def resolve_dataset_bundle(
    dataset_dir: Path = GLOBAL_DATASET_DIR,
    period_token: str | None = None,
    feature_version: str = "v1",
) -> DatasetBundle:
    if feature_version == "v2":
        feature_files = sorted(FEATURE_V2_DATASET_DIR.glob("ml_feature_v2_*.parquet"))
        if not feature_files:
            raise FileNotFoundError(
                f"No ML-v2 feature dataset found in {FEATURE_V2_DATASET_DIR}. "
                "Run ml_project/features_v2/07_build_ml_feature_v2.py first."
            )
        feature_by_token = {
            extract_token(path, "ml_feature_v2_"): path for path in feature_files
        }
        selected_token = period_token or max(feature_by_token, key=token_sort_key)
        if selected_token not in feature_by_token:
            raise FileNotFoundError(
                f"ML-v2 period token not found in {FEATURE_V2_DATASET_DIR}: {selected_token}"
            )
        selected_file = feature_by_token[selected_token]
        return DatasetBundle(
            period_token=selected_token,
            signal_file=selected_file,
            feature_file=selected_file,
            feature_version="v2",
        )

    return resolve_signal_feature_bundle(
        dataset_dir=dataset_dir,
        period_token=period_token,
    )


def apply_cohort_filter(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort not in COHORT_OPTIONS:
        raise ValueError(f"Unsupported cohort: {cohort}")

    if cohort == "all":
        log_step(f"Cohort filter: all eligible cases kept ({len(df):,} rows)")
        return df.copy()

    if cohort == "zolpidem":
        filtered = df[df["is_zolpidem"]].copy()
        label = "zolpidem-exposed"
    else:
        zdrug_cols = ["is_zolpidem", "is_zaleplon", "is_zopiclone", "is_eszopiclone"]
        filtered = df[df[zdrug_cols].any(axis=1)].copy()
        label = "any Z-drug-exposed"

    if filtered.empty:
        raise ValueError(f"Cohort filter produced no rows: {cohort}")
    log_step(
        f"Cohort filter: {label} cases kept ({len(filtered):,} of {len(df):,} rows)"
    )
    return filtered


def configure_feature_schema(feature_version: str, available_columns: list[str] | None = None) -> None:
    global BOOL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES

    if feature_version == "v1":
        BOOL_FEATURES = V1_BOOL_FEATURES.copy()
        NUMERIC_FEATURES = V1_NUMERIC_FEATURES.copy()
        CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES.copy()
    elif feature_version == "v2":
        available = set(available_columns or [])
        dynamic_soc_features = sorted(
            column for column in available if column.startswith("indi_soc_")
        )
        BOOL_FEATURES = (
            V1_BOOL_FEATURES
            + [column for column in V2_BASE_BOOL_FEATURES if column in available]
            + dynamic_soc_features
        )
        NUMERIC_FEATURES = V1_NUMERIC_FEATURES + [
            column for column in V2_BASE_NUMERIC_FEATURES if column in available
        ]
        CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES + [
            column for column in V2_BASE_CATEGORICAL_FEATURES if column in available
        ]
    else:
        raise ValueError(f"Unsupported feature version: {feature_version}")

    MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES


def load_modeling_frame(bundle: DatasetBundle, target_col: str, cohort: str) -> pd.DataFrame:
    if target_col not in TARGET_OPTIONS:
        raise ValueError(f"Unsupported target column: {target_col}")

    if bundle.feature_version == "v2":
        return load_modeling_frame_v2(bundle, target_col=target_col, cohort=cohort)

    configure_feature_schema("v1")

    raw_bool_features = [
        "is_zolpidem",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "polypharmacy_5",
    ]
    raw_numeric_features = ["drug_n", "distinct_drug_n"]
    signal_columns = list(
        dict.fromkeys(["caseid", target_col, "age_group", "sex_clean", "quarter", "year"])
    )
    feature_columns = list(
        dict.fromkeys(["caseid", *raw_numeric_features, *raw_bool_features])
    )

    log_step(
        f"Loading signal dataset: {bundle.signal_file.name} and feature dataset: {bundle.feature_file.name}"
    )
    signal_df = pd.read_parquet(bundle.signal_file, columns=signal_columns)
    feature_df = pd.read_parquet(bundle.feature_file, columns=feature_columns)

    signal_df["caseid"] = signal_df["caseid"].astype(str).str.strip()
    feature_df["caseid"] = feature_df["caseid"].astype(str).str.strip()

    signal_df = signal_df.drop_duplicates(subset=["caseid"]).copy()
    feature_df = feature_df.drop_duplicates(subset=["caseid"]).copy()

    merged = signal_df.merge(feature_df, on="caseid", how="inner")
    merged = merged[merged["caseid"] != ""].copy()

    for col in raw_bool_features + [target_col]:
        merged[col] = merged[col].fillna(False).astype(bool)

    for col in ["year", *raw_numeric_features]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    merged = add_derived_features(merged)

    for col in CATEGORICAL_FEATURES:
        merged[col] = (
            merged[col]
            .where(merged[col].notna(), "unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    final_df = merged[["caseid", target_col, *MODEL_FEATURES]].copy()
    final_df = apply_cohort_filter(final_df, cohort=cohort)
    log_step(f"Modeling frame ready with {len(final_df):,} rows")
    return final_df


def load_modeling_frame_v2(bundle: DatasetBundle, target_col: str, cohort: str) -> pd.DataFrame:
    log_step(f"Loading ML-v2 feature dataset: {bundle.feature_file.name}")
    df = pd.read_parquet(bundle.feature_file)
    if target_col not in df.columns:
        raise ValueError(f"ML-v2 feature dataset missing target column: {target_col}")

    leakage_cols = {"fall_pt_list", "fall_narrow_pt_count"}
    present_leakage = sorted(leakage_cols & set(df.columns))
    if present_leakage:
        raise ValueError(f"ML-v2 feature dataset contains leakage columns: {present_leakage}")

    df["caseid"] = df["caseid"].astype(str).str.strip()
    df = df[df["caseid"] != ""].drop_duplicates(subset=["caseid"]).copy()

    base_required = [
        "is_zolpidem",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "polypharmacy_5",
        "drug_n",
        "distinct_drug_n",
        "age_group",
        "sex_clean",
        "quarter",
        "year",
    ]
    missing = [column for column in base_required if column not in df.columns]
    if missing:
        raise ValueError(f"ML-v2 feature dataset missing required base columns: {missing}")

    for col in V1_BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in [target_col]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["year", "drug_n", "distinct_drug_n"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = add_derived_features(df)
    configure_feature_schema("v2", available_columns=list(df.columns))

    for col in BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = (
            df[col]
            .where(df[col].notna(), "unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    final_df = df[["caseid", target_col, *MODEL_FEATURES]].copy()
    final_df = apply_cohort_filter(final_df, cohort=cohort)
    log_step(
        f"ML-v2 modeling frame ready with {len(final_df):,} rows and {len(MODEL_FEATURES):,} features"
    )
    return final_df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    zdrug_cols = ["is_zolpidem", "is_zaleplon", "is_zopiclone", "is_eszopiclone"]
    cns_cols = [
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
    ]

    for col in zdrug_cols + cns_cols:
        frame[col] = frame[col].fillna(False).astype(bool)

    frame["is_other_zdrug"] = frame[["is_zaleplon", "is_zopiclone", "is_eszopiclone"]].any(axis=1)
    frame["zdrug_count"] = frame[zdrug_cols].sum(axis=1).astype(float)
    frame["multiple_zdrug"] = frame["zdrug_count"] >= 2
    frame["cns_coprescription_count"] = frame[cns_cols].sum(axis=1).astype(float)
    frame["any_cns_coprescription"] = frame["cns_coprescription_count"] >= 1

    frame["log_drug_n"] = np.log1p(frame["drug_n"].clip(lower=0))
    frame["log_distinct_drug_n"] = np.log1p(frame["distinct_drug_n"].clip(lower=0))
    frame["high_drug_burden_10"] = frame["distinct_drug_n"] >= 10
    frame["very_high_drug_burden_20"] = frame["distinct_drug_n"] >= 20

    frame["drug_n_bucket"] = pd.cut(
        frame["drug_n"],
        bins=[-np.inf, 1, 2, 4, 9, 19, np.inf],
        labels=["0-1", "2", "3-4", "5-9", "10-19", "20+"],
    ).astype("string")
    frame["distinct_drug_n_bucket"] = pd.cut(
        frame["distinct_drug_n"],
        bins=[-np.inf, 1, 2, 4, 9, 19, np.inf],
        labels=["0-1", "2", "3-4", "5-9", "10-19", "20+"],
    ).astype("string")
    frame["cns_coprescription_bucket"] = pd.cut(
        frame["cns_coprescription_count"],
        bins=[-np.inf, 0, 1, 2, np.inf],
        labels=["0", "1", "2", "3+"],
    ).astype("string")
    return frame


def sample_training_frame(
    df: pd.DataFrame,
    target_col: str,
    sample_n: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_n is None or sample_n <= 0 or len(df) <= sample_n:
        log_step(f"Training uses full dataset with {len(df):,} rows")
        return df.copy()

    labels = df[target_col].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Training frame must contain both positive and negative cases.")

    try:
        sampled_idx, _ = train_test_split(
            df.index.to_numpy(),
            train_size=sample_n,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError:
        sampled_idx = df.sample(
            n=sample_n, random_state=random_state, replace=False
        ).index.to_numpy()

    sampled = df.loc[sampled_idx].copy()
    sampled = sampled.sort_values(["year", "caseid"]).reset_index(drop=True)

    if sampled[target_col].astype(int).sum() == 0:
        raise ValueError(
            "Training sample contains no positive cases. Increase --train-sample-n."
        )
    log_step(
        f"Training sampled down from {len(df):,} to {len(sampled):,} rows with stratification"
    )
    return sampled


def temporal_split(
    df: pd.DataFrame,
    train_end_year: int,
    valid_year: int,
    test_year: int,
) -> dict[str, pd.DataFrame]:
    train_df = df[df["year"] <= train_end_year].copy()
    valid_df = df[df["year"] == valid_year].copy()
    test_df = df[df["year"] == test_year].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"train_end_year={train_end_year}, valid_year={valid_year}, test_year={test_year}"
        )
    log_step(
        "Temporal split ready: "
        f"train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}"
    )
    return {"train": train_df, "valid": valid_df, "test": test_df}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(with_mean=False), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def _determine_cv_folds(y: pd.Series | np.ndarray, requested_folds: int) -> int:
    y_arr = np.asarray(pd.Series(y).astype(int))
    positives = int(y_arr.sum())
    negatives = int(len(y_arr) - positives)
    folds = min(requested_folds, positives, negatives)
    if folds < 2:
        raise ValueError("Cross-validation requires at least 2 positives and 2 negatives.")
    return folds


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _safe_brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(brier_score_loss(y_true, y_score))


def _top_risk_metrics(y_true_arr: np.ndarray, y_score_arr: np.ndarray) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if len(y_true_arr) == 0:
        return rows
    order = np.argsort(-y_score_arr)
    baseline = float(y_true_arr.mean()) if len(y_true_arr) else 0.0
    for pct in [0.05, 0.10]:
        n_top = max(1, int(np.ceil(len(y_true_arr) * pct)))
        top_labels = y_true_arr[order[:n_top]]
        rate = float(top_labels.mean()) if n_top else 0.0
        label = f"top_{int(pct * 100)}pct"
        rows[f"{label}_n"] = int(n_top)
        rows[f"{label}_positive_cases"] = int(top_labels.sum())
        rows[f"{label}_positive_rate"] = rate
        rows[f"{label}_lift"] = rate / baseline if baseline > 0 else None
    return rows


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    y_pred = (y_score_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "n_rows": int(len(y_true_arr)),
        "positive_cases": int(y_true_arr.sum()),
        "positive_rate": float(y_true_arr.mean()),
        "threshold": float(threshold),
        "roc_auc": _safe_roc_auc(y_true_arr, y_score_arr),
        "average_precision": _safe_average_precision(y_true_arr, y_score_arr),
        "brier_score": _safe_brier_score(y_true_arr, y_score_arr),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    metrics.update(_top_risk_metrics(y_true_arr, y_score_arr))
    return metrics


def build_roc_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    if np.unique(y_true_arr).size < 2:
        return pd.DataFrame(
            columns=["threshold", "fpr", "tpr", "specificity", "youden_index"]
        )

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    roc_df = pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})
    roc_df = roc_df[np.isfinite(roc_df["threshold"])].copy()
    roc_df["specificity"] = 1.0 - roc_df["fpr"]
    roc_df["youden_index"] = roc_df["tpr"] - roc_df["fpr"]
    return roc_df.reset_index(drop=True)


def select_threshold_by_youden(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    roc_df = build_roc_table(y_true, y_score)
    if roc_df.empty:
        return {
            "threshold": 0.5,
            "youden_index": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "fpr": 0.0,
            "tpr": 0.0,
        }

    best_idx = int(roc_df["youden_index"].idxmax())
    best_row = roc_df.loc[best_idx]
    return {
        "threshold": float(best_row["threshold"]),
        "youden_index": float(best_row["youden_index"]),
        "sensitivity": float(best_row["tpr"]),
        "specificity": float(best_row["specificity"]),
        "fpr": float(best_row["fpr"]),
        "tpr": float(best_row["tpr"]),
    }


def fit_platt_calibrator(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    random_state: int,
) -> LogisticRegression:
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    if np.unique(y_true_arr).size < 2:
        raise ValueError("Validation labels must contain both classes for Platt scaling.")

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(np.asarray(y_score, dtype=float).reshape(-1, 1), y_true_arr)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    y_score: pd.Series | np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(y_score, dtype=float).reshape(-1, 1))[
        :, 1
    ]


def build_calibration_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target": np.asarray(pd.Series(y_true).astype(int)),
            "score": np.asarray(pd.Series(y_score).astype(float)),
        }
    )

    unique_scores = int(frame["score"].nunique())
    if unique_scores <= 1:
        return pd.DataFrame(
            [
                {
                    "bin": 1,
                    "n_rows": int(len(frame)),
                    "mean_predicted_probability": float(frame["score"].mean()),
                    "observed_rate": float(frame["target"].mean()),
                }
            ]
        )

    bin_count = min(n_bins, unique_scores)
    frame["bin_interval"] = pd.qcut(frame["score"], q=bin_count, duplicates="drop")
    calibration_df = (
        frame.groupby("bin_interval", observed=True)
        .agg(
            n_rows=("target", "size"),
            mean_predicted_probability=("score", "mean"),
            observed_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )
    calibration_df.insert(0, "bin", np.arange(1, len(calibration_df) + 1))
    return calibration_df


def bootstrap_metric_intervals(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))

    pos_idx = np.flatnonzero(y_true_arr == 1)
    neg_idx = np.flatnonzero(y_true_arr == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap requires both positive and negative cases.")

    metric_names = metrics or EVALUATION_METRICS
    point_estimates = evaluate_predictions(y_true_arr, y_score_arr, threshold=threshold)
    rng = np.random.default_rng(random_state)

    samples_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(n_bootstrap):
        sampled_pos_idx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg_idx = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
        rng.shuffle(sampled_idx)

        sampled_metrics = evaluate_predictions(
            y_true_arr[sampled_idx],
            y_score_arr[sampled_idx],
            threshold=threshold,
        )
        for metric in metric_names:
            value = sampled_metrics.get(metric)
            if value is not None:
                samples_by_metric[metric].append(float(value))

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        point_estimate = point_estimates.get(metric)
        metric_samples = np.asarray(samples_by_metric[metric], dtype=float)
        if point_estimate is None or metric_samples.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
            continue
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point_estimate),
                "ci_low": float(np.quantile(metric_samples, 0.025)),
                "ci_high": float(np.quantile(metric_samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def _count_search_candidates(search_spec: SearchSpec, search_mode: str) -> int | None:
    if search_mode == "none":
        return None
    param_space = search_spec.param_space_by_mode[search_mode]
    if search_spec.strategy == "grid":
        return len(ParameterGrid(param_space))
    if search_spec.n_iter_by_mode is None:
        return None
    return int(search_spec.n_iter_by_mode[search_mode])


def _normalize_cv_results(cv_results: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(cv_results).sort_values(
        by="rank_test_average_precision", na_position="last"
    )


def _fit_search(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    target_col: str,
    search_spec: SearchSpec,
    search_mode: str,
    cv_folds: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame | None]:
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[target_col].astype(int)

    if search_mode == "none":
        log_step("Search mode is none, fitting base model directly")
        pipeline.fit(X_train, y_train)
        return (
            pipeline,
            {
                "search_mode": "none",
                "search_strategy": "none",
                "refit_metric": REFIT_METRIC,
                "candidate_count": None,
                "best_score": None,
                "best_params": {},
            },
            None,
        )

    effective_folds = _determine_cv_folds(y_train, cv_folds)
    cv = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=random_state
    )
    param_space = search_spec.param_space_by_mode[search_mode]
    candidate_count = _count_search_candidates(search_spec, search_mode)
    log_step(
        "Starting hyperparameter search: "
        f"mode={search_mode}, strategy={search_spec.strategy}, "
        f"cv_folds={effective_folds}, candidates={candidate_count}"
    )

    if search_spec.strategy == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_space,
            scoring=SEARCH_SCORING,
            refit=REFIT_METRIC,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
            verbose=2,
        )
    else:
        if search_spec.n_iter_by_mode is None:
            raise ValueError("Random search requires n_iter_by_mode.")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=search_spec.n_iter_by_mode[search_mode],
            scoring=SEARCH_SCORING,
            refit=REFIT_METRIC,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
            random_state=random_state,
            verbose=2,
        )

    search.fit(X_train, y_train)
    log_step(
        f"Search finished, best {REFIT_METRIC}={search.best_score_:.6f}"
    )
    search_results_df = _normalize_cv_results(search.cv_results_)
    search_summary = {
        "search_mode": search_mode,
        "search_strategy": search_spec.strategy,
        "refit_metric": REFIT_METRIC,
        "cv_folds_used": effective_folds,
        "candidate_count": candidate_count,
        "best_score": float(search.best_score_),
        "best_params": {key: _json_safe(value) for key, value in search.best_params_.items()},
    }
    return search.best_estimator_, search_summary, search_results_df


def run_cross_validation_pipeline(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    target_col: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    y = train_df[target_col].astype(int)
    effective_folds = _determine_cv_folds(y, n_splits)
    splitter = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=random_state
    )

    rows: list[dict[str, Any]] = []
    log_step(f"Running post-search cross-validation summary with {effective_folds} folds")
    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(train_df[MODEL_FEATURES], y), start=1
    ):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(
            fold_train[MODEL_FEATURES], fold_train[target_col].astype(int)
        )
        fold_scores = fold_pipeline.predict_proba(fold_valid[MODEL_FEATURES])[:, 1]

        metrics = evaluate_predictions(
            fold_valid[target_col], fold_scores, threshold=0.5
        )
        metrics.update(
            {
                "fold": fold_idx,
                "train_rows": int(len(fold_train)),
                "valid_rows": int(len(fold_valid)),
                "train_positive_rate": float(fold_train[target_col].astype(int).mean()),
                "valid_positive_rate": float(fold_valid[target_col].astype(int).mean()),
            }
        )
        log_step(
            f"Cross-validation fold {fold_idx}/{effective_folds} done: "
            f"ap={metrics['average_precision']}, roc_auc={metrics['roc_auc']}"
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_cv_metrics(cv_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_folds": int(len(cv_df)),
        "train_rows_mean": float(cv_df["train_rows"].mean()),
        "valid_rows_mean": float(cv_df["valid_rows"].mean()),
    }
    for metric in EVALUATION_METRICS:
        metric_series = cv_df[metric].dropna()
        if metric_series.empty:
            summary[metric] = {"mean": None, "std": None}
            continue
        summary[metric] = {
            "mean": float(metric_series.mean()),
            "std": float(metric_series.std(ddof=1)) if len(metric_series) > 1 else 0.0,
        }
    return summary


def make_run_dir(
    model_name: str,
    target_col: str,
    period_token: str,
    cohort: str,
    feature_version: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_suffix = "" if feature_version == "v1" else f"_{feature_version}"
    run_dir = (
        OUTPUT_ML_ROOT
        / model_name
        / f"{target_col}_{cohort}_{period_token}{version_suffix}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        cast_value = float(value)
        return None if not np.isfinite(cast_value) else cast_value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_split_summary(
    splits: dict[str, pd.DataFrame],
    target_col: str,
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(split_df)),
                "positive_cases": int(split_df[target_col].astype(int).sum()),
                "positive_rate": float(split_df[target_col].astype(int).mean()),
                "min_year": int(split_df["year"].min()),
                "max_year": int(split_df["year"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_prediction_table(
    df: pd.DataFrame,
    target_col: str,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    pd.DataFrame(
        {
            "caseid": df["caseid"].astype(str),
            "year": df["year"].astype(int),
            "target": df[target_col].astype(int),
            "predicted_probability_raw": raw_scores,
            "predicted_probability_calibrated": calibrated_scores,
            "predicted_label_optimal": (calibrated_scores >= threshold).astype(int),
        }
    ).to_csv(output_path, index=False, encoding="utf-8-sig")


def _extract_model_params(pipeline: Pipeline) -> dict[str, Any]:
    model = pipeline.named_steps["model"]
    params = model.get_params(deep=False)
    if isinstance(model, LogisticRegression) and params.get("penalty") == "deprecated":
        params["penalty"] = "l2 (scikit-learn default)"
    return {key: _json_safe(value) for key, value in sorted(params.items())}


def _build_search_payload(
    result: ExperimentResult,
    model_name: str,
    display_name: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "display_name": display_name,
        "search_mode": result.search_summary["search_mode"],
        "search_strategy": result.search_summary["search_strategy"],
        "refit_metric": result.search_summary["refit_metric"],
        "cv_folds_used": result.search_summary.get("cv_folds_used"),
        "candidate_count": result.search_summary.get("candidate_count"),
        "best_score": result.search_summary.get("best_score"),
        "best_params": result.search_summary.get("best_params", {}),
        "selected_model_params": _extract_model_params(result.pipeline),
    }


def _build_metrics_payload(
    result: ExperimentResult,
    model_name: str,
    display_name: str,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "display_name": display_name,
        "feature_version": result.config.feature_version,
        "target_col": result.config.target_col,
        "cohort": result.config.cohort,
        "period_token": result.bundle.period_token,
        "signal_file": str(result.bundle.signal_file),
        "feature_file": str(result.bundle.feature_file),
        "train_end_year": result.config.train_end_year,
        "valid_year": result.config.valid_year,
        "test_year": result.config.test_year,
        "train_sample_n": result.config.train_sample_n,
        "search_mode": result.config.search_mode,
        "cv_folds_requested": result.config.cv_folds,
        "bootstrap_iterations": result.config.bootstrap_iterations,
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "bool_features": BOOL_FEATURES,
        "search_summary": result.search_summary,
        "cross_validation_summary": result.cv_summary,
        "threshold_selection": result.threshold_selection,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "validation_metrics_raw_threshold_0_5": result.validation_metrics_raw,
        "test_metrics_raw_threshold_0_5": result.test_metrics_raw,
        "calibration_method": "platt",
    }


def run_model_experiment(
    *,
    config: ExperimentConfig,
    model_name: str,
    display_name: str,
    estimator_factory: Callable[[pd.DataFrame, ExperimentConfig], BaseEstimator],
    search_spec: SearchSpec,
) -> ExperimentResult:
    log_step(
        "Resolving dataset bundle for "
        f"period token: {config.period_token or 'latest'}, feature_version={config.feature_version}"
    )
    bundle = resolve_dataset_bundle(
        period_token=config.period_token,
        feature_version=config.feature_version,
    )
    modeling_df = load_modeling_frame(
        bundle=bundle,
        target_col=config.target_col,
        cohort=config.cohort,
    )
    splits = temporal_split(
        modeling_df,
        train_end_year=config.train_end_year,
        valid_year=config.valid_year,
        test_year=config.test_year,
    )

    train_full_df = splits["train"]
    train_df = sample_training_frame(
        train_full_df,
        target_col=config.target_col,
        sample_n=config.train_sample_n,
        random_state=config.random_state,
    )
    valid_df = splits["valid"]
    test_df = splits["test"]

    pipeline = build_pipeline(estimator_factory(train_df, config))
    fitted_pipeline, search_summary, search_results_df = _fit_search(
        pipeline=pipeline,
        train_df=train_df,
        target_col=config.target_col,
        search_spec=search_spec,
        search_mode=config.search_mode,
        cv_folds=config.cv_folds,
        random_state=config.random_state,
    )

    cv_metrics_df = run_cross_validation_pipeline(
        pipeline=fitted_pipeline,
        train_df=train_df,
        target_col=config.target_col,
        n_splits=config.cv_folds,
        random_state=config.random_state,
    )
    cv_summary = summarize_cv_metrics(cv_metrics_df)

    valid_raw_scores = fitted_pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
    test_raw_scores = fitted_pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]
    log_step("Validation and test probabilities generated")

    calibrator = fit_platt_calibrator(
        valid_df[config.target_col], valid_raw_scores, config.random_state
    )
    valid_scores = apply_platt_calibrator(calibrator, valid_raw_scores)
    test_scores = apply_platt_calibrator(calibrator, test_raw_scores)
    log_step("Probability calibration completed with Platt scaling")

    threshold_selection = select_threshold_by_youden(
        valid_df[config.target_col], valid_scores
    )
    threshold = threshold_selection["threshold"]

    validation_metrics = evaluate_predictions(
        valid_df[config.target_col], valid_scores, threshold=threshold
    )
    test_metrics = evaluate_predictions(
        test_df[config.target_col], test_scores, threshold=threshold
    )
    validation_metrics_raw = evaluate_predictions(
        valid_df[config.target_col], valid_raw_scores, threshold=0.5
    )
    test_metrics_raw = evaluate_predictions(
        test_df[config.target_col], test_raw_scores, threshold=0.5
    )

    run_dir = make_run_dir(
        model_name=model_name,
        target_col=config.target_col,
        period_token=bundle.period_token,
        cohort=config.cohort,
        feature_version=bundle.feature_version,
    )
    log_step(f"Writing outputs to {run_dir}")

    valid_roc_df = build_roc_table(valid_df[config.target_col], valid_scores)
    test_roc_df = build_roc_table(test_df[config.target_col], test_scores)
    valid_calibration_df = build_calibration_table(
        valid_df[config.target_col], valid_scores
    )
    test_calibration_df = build_calibration_table(
        test_df[config.target_col], test_scores
    )
    bootstrap_df = bootstrap_metric_intervals(
        test_df[config.target_col],
        test_scores,
        threshold=threshold,
        n_bootstrap=config.bootstrap_iterations,
        random_state=config.random_state,
    )

    result = ExperimentResult(
        config=config,
        bundle=bundle,
        run_dir=run_dir,
        train_full_df=train_full_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        pipeline=fitted_pipeline,
        search_summary=search_summary,
        search_results_df=search_results_df,
        cv_metrics_df=cv_metrics_df,
        cv_summary=cv_summary,
        threshold_selection=threshold_selection,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_metrics_raw=validation_metrics_raw,
        test_metrics_raw=test_metrics_raw,
        valid_raw_scores=valid_raw_scores,
        valid_scores=valid_scores,
        test_raw_scores=test_raw_scores,
        test_scores=test_scores,
    )

    cv_metrics_df.to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    if search_results_df is not None:
        search_results_df.to_csv(
            run_dir / "search_results.csv", index=False, encoding="utf-8-sig"
        )
    save_json(
        run_dir / "best_params.json",
        _build_search_payload(result, model_name=model_name, display_name=display_name),
    )
    save_json(
        run_dir / "metrics.json",
        _build_metrics_payload(result, model_name=model_name, display_name=display_name),
    )
    save_split_summary(
        {
            "train_full": train_full_df,
            "train_sampled": train_df,
            "valid": valid_df,
            "test": test_df,
        },
        target_col=config.target_col,
        output_path=run_dir / "split_summary.csv",
    )
    save_prediction_table(
        valid_df,
        config.target_col,
        valid_raw_scores,
        valid_scores,
        threshold,
        run_dir / "validation_predictions.csv",
    )
    save_prediction_table(
        test_df,
        config.target_col,
        test_raw_scores,
        test_scores,
        threshold,
        run_dir / "test_predictions.csv",
    )
    valid_roc_df.to_csv(
        run_dir / "validation_roc_curve.csv", index=False, encoding="utf-8-sig"
    )
    test_roc_df.to_csv(
        run_dir / "test_roc_curve.csv", index=False, encoding="utf-8-sig"
    )
    valid_calibration_df.to_csv(
        run_dir / "validation_calibration_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_calibration_df.to_csv(
        run_dir / "test_calibration_curve.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap_df.to_csv(
        run_dir / "test_bootstrap_metrics.csv", index=False, encoding="utf-8-sig"
    )
    log_step("All outputs saved")
    return result


def summarize_importance_highlights(
    feature_df: pd.DataFrame,
    *,
    feature_col: str,
    score_col: str,
    top_n: int = 10,
) -> list[str]:
    top_df = feature_df.sort_values(score_col, ascending=False).head(top_n)
    highlights = []
    for _, row in top_df.iterrows():
        highlights.append(f"{row[feature_col]}: {row[score_col]:.4f}")
    return highlights


def summarize_logistic_highlights(coefficients_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    positive_df = coefficients_df.sort_values("coefficient", ascending=False).head(top_n)
    negative_df = coefficients_df.sort_values("coefficient", ascending=True).head(top_n)
    highlights: list[str] = []
    for _, row in positive_df.iterrows():
        highlights.append(
            f"Positive association: {row['feature']} coefficient={row['coefficient']:.4f}, odds_ratio={row['odds_ratio']:.4f}"
        )
    for _, row in negative_df.iterrows():
        highlights.append(
            f"Negative association: {row['feature']} coefficient={row['coefficient']:.4f}, odds_ratio={row['odds_ratio']:.4f}"
        )
    return highlights


def save_model_card(
    *,
    output_path: Path,
    display_name: str,
    model_name: str,
    result: ExperimentResult,
    feature_highlights: list[str],
    notes: list[str] | None = None,
) -> None:
    notes = notes or []
    best_params_payload = _build_search_payload(
        result, model_name=model_name, display_name=display_name
    )
    best_params_lines = json.dumps(
        best_params_payload["selected_model_params"], ensure_ascii=False, indent=2
    )

    lines = [
        f"# {display_name} model card",
        "",
        "## Task",
        f"- Predict `{result.config.target_col}` from the FAERS global case-level bundle.",
        "- Use the model as a research ranking layer on top of the existing signal detection workflow.",
        "",
        "## Data",
        f"- Signal file: `{result.bundle.signal_file}`",
        f"- Feature file: `{result.bundle.feature_file}`",
        f"- Period token: `{result.bundle.period_token}`",
        f"- Cohort: `{result.config.cohort}`",
        "",
        "## Time split",
        f"- Train: years <= {result.config.train_end_year}",
        f"- Validation: {result.config.valid_year}",
        f"- Test: {result.config.test_year}",
        "",
        "## Search",
        f"- Search mode: `{result.search_summary['search_mode']}`",
        f"- Search strategy: `{result.search_summary['search_strategy']}`",
        f"- Refit metric: `{result.search_summary['refit_metric']}`",
        f"- Candidate count: `{result.search_summary.get('candidate_count')}`",
        "",
        "## Selected parameters",
        "```json",
        best_params_lines,
        "```",
        "",
        "## Final metrics",
        f"- Validation average precision: `{result.validation_metrics['average_precision']}`",
        f"- Validation ROC-AUC: `{result.validation_metrics['roc_auc']}`",
        f"- Test average precision: `{result.test_metrics['average_precision']}`",
        f"- Test ROC-AUC: `{result.test_metrics['roc_auc']}`",
        f"- Test Brier score: `{result.test_metrics['brier_score']}`",
        "",
        "## Feature highlights",
    ]
    lines.extend(f"- {highlight}" for highlight in feature_highlights)
    lines.extend(
        [
            "",
            "## Limitations",
            "- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.",
            "- The output reflects reporting patterns in FAERS, not causal drug effects.",
            "- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.",
        ]
    )
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
