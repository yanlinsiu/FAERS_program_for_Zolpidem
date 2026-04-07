from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
OUTPUT_ML_ROOT = PROJECT_ROOT / "OUTPUT_ML"

TARGET_OPTIONS = {"is_fall", "has_fall_related_broad", "serious"}

BOOL_FEATURES = [
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

NUMERIC_FEATURES = [
    "year",
    "drug_n",
    "distinct_drug_n",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "sex_clean",
    "quarter",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES

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
class DatasetBundle:
    period_token: str
    signal_file: Path
    feature_file: Path


def _extract_token(path: Path, prefix: str, suffix: str = "") -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected dataset file name: {path.name}")
    token = stem[len(prefix) :]
    if suffix:
        if not token.endswith(suffix):
            raise ValueError(f"Unexpected dataset file name: {path.name}")
        token = token[: -len(suffix)]
    return token


def _token_sort_key(token: str) -> tuple[int, int, int, str]:
    parts = token.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        start_year = int(parts[0])
        end_year = int(parts[1])
        return (end_year - start_year, end_year, -start_year, token)
    return (0, 0, 0, token)


def resolve_dataset_bundle(
    dataset_dir: Path = GLOBAL_DATASET_DIR,
    period_token: str | None = None,
) -> DatasetBundle:
    signal_files = sorted(dataset_dir.glob("signal_dataset_*.parquet"))
    feature_files = sorted(dataset_dir.glob("drug_feature_*_case.parquet"))

    if not signal_files:
        raise FileNotFoundError(f"No signal dataset found in {dataset_dir}")
    if not feature_files:
        raise FileNotFoundError(f"No feature dataset found in {dataset_dir}")

    signal_by_token = {
        _extract_token(path, "signal_dataset_"): path for path in signal_files
    }
    feature_by_token = {
        _extract_token(path, "drug_feature_", "_case"): path for path in feature_files
    }
    shared_tokens = sorted(set(signal_by_token) & set(feature_by_token))
    if not shared_tokens:
        raise RuntimeError(f"No matching signal/feature bundle found in {dataset_dir}")

    selected_token = period_token
    if selected_token is None:
        selected_token = max(shared_tokens, key=_token_sort_key)

    if selected_token not in signal_by_token or selected_token not in feature_by_token:
        raise FileNotFoundError(
            f"Period token not found in {dataset_dir}: {selected_token}"
        )

    return DatasetBundle(
        period_token=selected_token,
        signal_file=signal_by_token[selected_token],
        feature_file=feature_by_token[selected_token],
    )


def load_modeling_frame(bundle: DatasetBundle, target_col: str) -> pd.DataFrame:
    if target_col not in TARGET_OPTIONS:
        raise ValueError(f"Unsupported target_col: {target_col}")

    signal_df = pd.read_parquet(bundle.signal_file)
    feature_df = pd.read_parquet(bundle.feature_file)

    signal_df["caseid"] = signal_df["caseid"].astype(str).str.strip()
    feature_df["caseid"] = feature_df["caseid"].astype(str).str.strip()

    signal_df = signal_df.drop_duplicates(subset=["caseid"]).copy()
    feature_df = feature_df.drop_duplicates(subset=["caseid"]).copy()

    merged = signal_df.merge(
        feature_df, on="caseid", how="inner", suffixes=("", "_feature")
    )
    merged = merged[merged["caseid"] != ""].copy()

    if target_col not in merged.columns:
        raise ValueError(f"Target column not present in merged dataset: {target_col}")

    for col in BOOL_FEATURES + [target_col]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(False).astype(bool)

    for col in NUMERIC_FEATURES:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    for col in CATEGORICAL_FEATURES:
        merged[col] = (
            merged[col]
            .where(merged[col].notna(), "unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    return merged[["caseid", target_col, *MODEL_FEATURES]].copy()


def sample_training_frame(
    df: pd.DataFrame,
    target_col: str,
    sample_n: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_n is None or sample_n <= 0 or len(df) <= sample_n:
        return df.copy()

    sampled = df.sample(n=sample_n, random_state=random_state, replace=False)
    sampled = sampled.sort_values(["year", "caseid"]).reset_index(drop=True)

    if sampled[target_col].astype(int).sum() == 0:
        raise ValueError(
            "Random training sample contains no positive cases. Increase --train-sample-n."
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

    return {"train": train_df, "valid": valid_df, "test": test_df}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_int = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    y_pred = (y_score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "n_rows": int(len(y_true_int)),
        "positive_cases": int(y_true_int.sum()),
        "positive_rate": float(y_true_int.mean()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true_int, y_score_arr)),
        "average_precision": float(average_precision_score(y_true_int, y_score_arr)),
        "brier_score": float(brier_score_loss(y_true_int, y_score_arr)),
        "accuracy": float(accuracy_score(y_true_int, y_pred)),
        "precision": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_roc_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    y_true_int = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    fpr, tpr, thresholds = roc_curve(y_true_int, y_score_arr)
    roc_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "fpr": fpr,
            "tpr": tpr,
        }
    )
    roc_df = roc_df[np.isfinite(roc_df["threshold"])].copy()
    roc_df["specificity"] = 1.0 - roc_df["fpr"]
    roc_df["youden_index"] = roc_df["tpr"] - roc_df["fpr"]
    return roc_df.reset_index(drop=True)


def select_threshold_by_youden(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    roc_df = build_roc_table(y_true, y_score)
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
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(
        np.asarray(y_score, dtype=float).reshape(-1, 1), np.asarray(y_true, dtype=int)
    )
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
            "target": pd.Series(y_true).astype(int),
            "score": pd.Series(y_score).astype(float),
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
    y_true_int = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))

    pos_idx = np.flatnonzero(y_true_int == 1)
    neg_idx = np.flatnonzero(y_true_int == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap requires both positive and negative cases.")

    metric_names = metrics or EVALUATION_METRICS
    point_estimates = evaluate_predictions(y_true_int, y_score_arr, threshold=threshold)

    rng = np.random.default_rng(random_state)
    samples_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for _ in range(n_bootstrap):
        sampled_pos_idx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg_idx = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
        rng.shuffle(sampled_idx)

        sampled_metrics = evaluate_predictions(
            y_true_int[sampled_idx],
            y_score_arr[sampled_idx],
            threshold=threshold,
        )
        for metric in metric_names:
            samples_by_metric[metric].append(float(sampled_metrics[metric]))

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        metric_samples = np.asarray(samples_by_metric[metric], dtype=float)
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point_estimates[metric]),
                "ci_low": float(np.quantile(metric_samples, 0.025)),
                "ci_high": float(np.quantile(metric_samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def run_cross_validation(
    build_pipeline: Callable[[], Pipeline],
    train_df: pd.DataFrame,
    target_col: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    y = train_df[target_col].astype(int)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(train_df[MODEL_FEATURES], y), start=1
    ):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        pipeline = build_pipeline()
        pipeline.fit(fold_train[MODEL_FEATURES], fold_train[target_col].astype(int))
        fold_scores = pipeline.predict_proba(fold_valid[MODEL_FEATURES])[:, 1]

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
        rows.append(metrics)

    return pd.DataFrame(rows)


def summarize_cv_metrics(cv_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_folds": int(len(cv_df)),
        "train_rows_mean": float(cv_df["train_rows"].mean()),
        "valid_rows_mean": float(cv_df["valid_rows"].mean()),
    }
    for metric in EVALUATION_METRICS:
        summary[metric] = {
            "mean": float(cv_df[metric].mean()),
            "std": float(cv_df[metric].std(ddof=1)) if len(cv_df) > 1 else 0.0,
        }
    return summary


def make_run_dir(model_name: str, target_col: str, period_token: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ML_ROOT / model_name / f"{target_col}_{period_token}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_split_summary(
    splits: dict[str, pd.DataFrame], target_col: str, output_path: Path
) -> None:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(split_df)),
                "positive_cases": int(split_df[target_col].sum()),
                "positive_rate": float(split_df[target_col].mean()),
                "min_year": int(split_df["year"].min()),
                "max_year": int(split_df["year"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
