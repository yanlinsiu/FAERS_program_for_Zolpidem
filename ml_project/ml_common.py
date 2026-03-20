from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
        _extract_token(path, "signal_dataset_"): path
        for path in signal_files
    }
    feature_by_token = {
        _extract_token(path, "drug_feature_", "_case"): path
        for path in feature_files
    }
    shared_tokens = sorted(set(signal_by_token) & set(feature_by_token))
    if not shared_tokens:
        raise RuntimeError(f"No matching signal/feature bundle found in {dataset_dir}")

    selected_token = period_token
    if selected_token is None:
        selected_token = shared_tokens[-1]

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

    merged = signal_df.merge(feature_df, on="caseid", how="inner", suffixes=("", "_feature"))
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

    positives = df[df[target_col]].copy()
    negatives = df[~df[target_col]].copy()

    positive_target = min(len(positives), max(1, int(sample_n * 0.5)))
    negative_target = max(0, sample_n - positive_target)

    sampled_parts = [
        positives.sample(n=positive_target, random_state=random_state, replace=False),
    ]
    if negative_target > 0:
        sampled_parts.append(
            negatives.sample(n=negative_target, random_state=random_state, replace=False)
        )

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


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
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_int = y_true.astype(int)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_int, y_pred, labels=[0, 1]).ravel()

    return {
        "n_rows": int(len(y_true)),
        "positive_rate": float(y_true_int.mean()),
        "roc_auc": float(roc_auc_score(y_true_int, y_score)),
        "average_precision": float(average_precision_score(y_true_int, y_score)),
        "precision_at_0_5": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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


def save_split_summary(splits: dict[str, pd.DataFrame], target_col: str, output_path: Path) -> None:
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
