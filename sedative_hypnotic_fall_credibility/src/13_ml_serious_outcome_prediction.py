from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_PERFORMANCE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_13_ml_serious_outcome_model_performance.csv"
DEFAULT_IMPORTANCE_OUT = PROJECT_DIR / "outputs" / "tables" / "table_13_ml_serious_outcome_feature_importance.csv"
DEFAULT_TUNING_OUT = PROJECT_DIR / "outputs" / "tables" / "table_13_ml_serious_outcome_tuning_results.csv"
DEFAULT_SHAP_OUT = PROJECT_DIR / "outputs" / "tables" / "table_13_ml_serious_outcome_shap_top_features.csv"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "13_ml_serious_outcome_prediction_qc.csv"
DEFAULT_CURVE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_13_ml_serious_outcome_roc_pr_curves.png"
DEFAULT_SHAP_FIGURE_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_13_ml_serious_outcome_shap_summary.png"

TARGET_OPTIONS = {
    "serious_any",
    "serious_hospitalization",
    "serious_death",
    "serious_disability",
    "serious_life_threatening",
    "serious_required_intervention",
    "serious_congenital_anomaly",
    "serious_other",
}
DEFAULT_TARGETS = [
    "serious_any",
    "serious_hospitalization",
    "serious_death",
    "serious_life_threatening",
]
MODEL_OPTIONS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
}
DEFAULT_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
]
FEATURE_SET_OPTIONS = {
    "full",
    "no_reporting_structure",
}
REPORTING_STRUCTURE_FEATURES = {
    "year",
    "quarter",
    "regulatory_period",
    "country_group",
    "reporter_country",
    "occr_country",
    "rept_cod",
    "e_sub",
}

BASE_COLUMNS = [
    "caseid",
    "analysis_eligible_main",
    "strict_fall",
    "age_years",
    "age_group_3",
    "sex_clean",
    "year",
    "quarter",
    "regulatory_period",
    "country_group",
    "reporter_country",
    "occr_country",
    "rept_cod",
    "e_sub",
]

NUMERIC_FEATURES = [
    "age_years",
    "year",
    "quarter",
    "drug_n",
    "distinct_drug_n",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
    "n_sedative_hypnotic_drugs_ps_only",
    "n_sedative_hypnotic_groups_ps_only",
]

CATEGORICAL_FEATURES = [
    "age_group_3",
    "sex_clean",
    "regulatory_period",
    "country_group",
    "reporter_country",
    "occr_country",
    "rept_cod",
    "e_sub",
]

BINARY_FEATURES = [
    "exposure_zolpidem_ps_ss",
    "exposure_zolpidem_ps_only",
    "exposure_eszopiclone_ps_ss",
    "exposure_eszopiclone_ps_only",
    "exposure_zaleplon_ps_ss",
    "exposure_zaleplon_ps_only",
    "exposure_zopiclone_ps_ss",
    "exposure_zopiclone_ps_only",
    "exposure_temazepam_ps_ss",
    "exposure_temazepam_ps_only",
    "exposure_triazolam_ps_ss",
    "exposure_triazolam_ps_only",
    "exposure_lorazepam_ps_ss",
    "exposure_lorazepam_ps_only",
    "exposure_diazepam_ps_ss",
    "exposure_diazepam_ps_only",
    "exposure_alprazolam_ps_ss",
    "exposure_alprazolam_ps_only",
    "exposure_clonazepam_ps_ss",
    "exposure_clonazepam_ps_only",
    "exposure_suvorexant_ps_ss",
    "exposure_suvorexant_ps_only",
    "exposure_lemborexant_ps_ss",
    "exposure_lemborexant_ps_only",
    "exposure_daridorexant_ps_ss",
    "exposure_daridorexant_ps_only",
    "exposure_trazodone_ps_ss",
    "exposure_trazodone_ps_only",
    "exposure_mirtazapine_ps_ss",
    "exposure_mirtazapine_ps_only",
    "exposure_doxepin_ps_ss",
    "exposure_doxepin_ps_only",
    "exposure_ramelteon_ps_ss",
    "exposure_ramelteon_ps_only",
    "exposure_melatonin_ps_ss",
    "exposure_melatonin_ps_only",
    "exposure_z_drug_ps_ss",
    "exposure_z_drug_ps_only",
    "exposure_other_z_drug_ps_ss",
    "exposure_other_z_drug_ps_only",
    "exposure_benzodiazepine_ps_ss",
    "exposure_benzodiazepine_ps_only",
    "exposure_orexin_antagonist_ps_ss",
    "exposure_orexin_antagonist_ps_only",
    "exposure_other_insomnia_related_ps_ss",
    "exposure_other_insomnia_related_ps_only",
    "mixed_z_drug_ps_ss",
    "mixed_sedative_hypnotic_group_ps_ss",
    "z_drug_plus_benzo_ps_ss",
    "mixed_z_drug_ps_only",
    "mixed_sedative_hypnotic_group_ps_only",
    "z_drug_plus_benzo_ps_only",
    "polypharmacy",
    "polypharmacy_5",
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

LEAKAGE_PREFIXES = ("serious_", "pheno_injury", "fall_pt")
LEAKAGE_COLUMNS = {
    "strict_fall",
    "broad_fall",
    "fall_pt_count",
    "fall_pt_list",
    "serious_any",
    "serious_death",
    "serious_hospitalization",
    "serious_disability",
    "serious_life_threatening",
    "serious_required_intervention",
    "serious_congenital_anomaly",
    "serious_other",
}


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        kwargs["sparse_output"] = False
    else:
        kwargs["sparse"] = False
    return OneHotEncoder(**kwargs)


def read_main_dataset(path: Path, target: str) -> pd.DataFrame:
    if target not in TARGET_OPTIONS:
        raise ValueError(f"Unsupported target: {target}. Choose one of {sorted(TARGET_OPTIONS)}")
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")

    available = pq.ParquetFile(path).schema.names
    requested = BASE_COLUMNS + NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [target]
    columns = [column for column in dict.fromkeys(requested) if column in available]
    missing_required = [column for column in ["caseid", "analysis_eligible_main", "strict_fall", target] if column not in columns]
    if missing_required:
        raise ValueError(f"Main dataset is missing required columns: {missing_required}")
    return pd.read_parquet(path, columns=columns)


def build_analysis_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    eligible = safe_bool(df["analysis_eligible_main"])
    strict_fall = safe_bool(df["strict_fall"])

    if scope == "strict_fall":
        return eligible & strict_fall
    if scope == "sedative_fall":
        group_columns = [
            "exposure_z_drug_ps_ss",
            "exposure_benzodiazepine_ps_ss",
            "exposure_orexin_antagonist_ps_ss",
            "exposure_other_insomnia_related_ps_ss",
        ]
        present = [column for column in group_columns if column in df.columns]
        if not present:
            raise ValueError("No sedative-hypnotic group exposure columns are available.")
        any_sedative = pd.Series(False, index=df.index)
        for column in present:
            any_sedative = any_sedative | safe_bool(df[column])
        return eligible & strict_fall & any_sedative
    raise ValueError("scope must be 'sedative_fall' or 'strict_fall'")


def collapse_rare_categories(series: pd.Series, min_count: int) -> pd.Series:
    values = series.astype("object").where(series.notna(), "missing").astype(str)
    counts = values.value_counts(dropna=False)
    rare = counts[counts < min_count].index
    return values.where(~values.isin(rare), "other_rare")


def select_features(model_df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SET_OPTIONS:
        raise ValueError(f"Unsupported feature set: {feature_set}. Choose from {sorted(FEATURE_SET_OPTIONS)}")

    features = [
        column
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
        if column in model_df.columns
    ]
    if feature_set == "no_reporting_structure":
        features = [column for column in features if column not in REPORTING_STRUCTURE_FEATURES]
    return features


def prepare_dataset(
    df: pd.DataFrame,
    target: str,
    scope: str,
    feature_set: str,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    mask = build_analysis_mask(df, scope)
    model_df = df.loc[mask].copy()
    model_df = model_df.loc[model_df[target].notna()].copy()
    model_df[target] = safe_bool(model_df[target]).astype(int)

    if max_rows and len(model_df) > max_rows:
        model_df, _ = train_test_split(
            model_df,
            train_size=max_rows,
            stratify=model_df[target],
            random_state=random_state,
        )

    features = select_features(model_df, feature_set)
    validate_no_leakage(features)

    X = model_df[features].copy()
    for column in [c for c in NUMERIC_FEATURES if c in X.columns]:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in [c for c in CATEGORICAL_FEATURES if c in X.columns]:
        X[column] = collapse_rare_categories(X[column], min_count=25)
    for column in [c for c in BINARY_FEATURES if c in X.columns]:
        X[column] = safe_bool(X[column]).astype(int)

    all_missing = [column for column in X.columns if X[column].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    y = model_df[target].astype(int)
    metadata = model_df[["caseid", target]].copy()
    return X, y, metadata


def validate_no_leakage(features: list[str]) -> None:
    leaked = []
    for column in features:
        if column in LEAKAGE_COLUMNS or any(column.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            leaked.append(column)
    if leaked:
        raise ValueError(f"Potential target leakage columns were selected as features: {leaked}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in NUMERIC_FEATURES if column in X.columns]
    categorical = [column for column in CATEGORICAL_FEATURES if column in X.columns]
    binary = [column for column in BINARY_FEATURES if column in X.columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", one_hot_encoder()),
                    ]
                ),
                categorical,
            ),
            (
                "binary",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                binary,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def xgboost_scale_pos_weight(y_train: pd.Series) -> float:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    return negative / positive if positive else 1.0


def build_models(random_state: int, n_estimators: int, scale_pos_weight: float) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )
    return models


def build_param_distributions(n_estimators: int, scale_pos_weight: float) -> dict[str, dict[str, list[object]]]:
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__solver": ["lbfgs"],
            "model__class_weight": ["balanced"],
        },
        "random_forest": {
            "model__n_estimators": [n_estimators, max(500, n_estimators)],
            "model__max_depth": [5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [5, 10, 20, 50],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__bootstrap": [True],
            "model__class_weight": ["balanced_subsample"],
        },
        "xgboost": {
            "model__n_estimators": [n_estimators, max(500, n_estimators)],
            "model__max_depth": [2, 3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 5, 10],
            "model__gamma": [0, 0.5, 1.0],
            "model__reg_alpha": [0, 0.1, 1.0],
            "model__reg_lambda": [1.0, 5.0, 10.0],
            "model__scale_pos_weight": [scale_pos_weight],
        },
    }


def threshold_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float | int]:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "brier_score": brier_score_loss(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if name == "categorical":
                encoder = transformer.named_steps["onehot"]
                names.extend(encoder.get_feature_names_out(columns))
            else:
                names.extend(columns)
        return np.asarray(names)


def signed_importance(model_name: str, estimator: object) -> np.ndarray | None:
    if model_name == "logistic_regression":
        return estimator.coef_.ravel()
    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_
    return None


def fit_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    scope: str,
    selected_models: list[str],
    random_state: int,
    n_estimators: int,
    tune: bool,
    search_iter: int,
    cv_folds: int,
    tune_scoring: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Pipeline], tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    performance_rows = []
    importance_rows = []
    tuning_rows = []
    fitted: dict[str, Pipeline] = {}

    scale_pos_weight = xgboost_scale_pos_weight(y_train)
    param_distributions = build_param_distributions(n_estimators, scale_pos_weight)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    available_models = build_models(random_state, n_estimators, scale_pos_weight)
    missing_models = [model for model in selected_models if model not in available_models]
    if missing_models:
        raise ValueError(f"Requested models are unavailable: {missing_models}")

    for model_name in selected_models:
        estimator = available_models[model_name]
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        best_cv_score = np.nan
        best_params: dict[str, object] = {}
        tuning_status = "not_tuned"

        if tune:
            grid_size = len(ParameterGrid(param_distributions[model_name]))
            n_iter = min(search_iter, grid_size)
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions[model_name],
                n_iter=n_iter,
                scoring=tune_scoring,
                cv=cv,
                n_jobs=-1,
                random_state=random_state,
                refit=True,
                verbose=1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_cv_score = float(search.best_score_)
            best_params = search.best_params_
            tuning_status = "tuned"
            tuning_table = pd.DataFrame(search.cv_results_)
            tuning_table["target"] = target
            tuning_table["scope"] = scope
            tuning_table["model"] = model_name
            tuning_rows.extend(tuning_table.to_dict("records"))
        else:
            pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = threshold_metrics(y_test, proba)
        performance_rows.append(
            {
                "target": target,
                "scope": scope,
                "model": model_name,
                "tuning_status": tuning_status,
                "tune_scoring": tune_scoring if tune else "",
                "cv_best_score": best_cv_score,
                "best_params_json": json.dumps(best_params, ensure_ascii=False),
                "train_n": len(X_train),
                "test_n": len(X_test),
                "train_positive_n": int(y_train.sum()),
                "test_positive_n": int(y_test.sum()),
                "train_positive_percent": float(y_train.mean() * 100),
                "test_positive_percent": float(y_test.mean() * 100),
                **metrics,
            }
        )

        names = get_feature_names(pipeline.named_steps["preprocess"])
        values = signed_importance(model_name, pipeline.named_steps["model"])
        if values is not None:
            order = np.argsort(np.abs(values))[::-1][:50]
            for rank, idx in enumerate(order, start=1):
                importance_rows.append(
                    {
                        "target": target,
                        "scope": scope,
                        "model": model_name,
                        "rank": rank,
                        "feature": names[idx],
                        "importance": float(values[idx]),
                        "abs_importance": float(abs(values[idx])),
                    }
                )
        fitted[model_name] = pipeline

    return pd.DataFrame(performance_rows), pd.DataFrame(importance_rows), pd.DataFrame(tuning_rows), fitted, (X_train, X_test, y_train, y_test)


def plot_curves(fitted: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model_name, pipeline in fitted.items():
        proba = pipeline.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, name=model_name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, name=model_name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_title("ROC curve")
    axes[1].set_title("Precision-recall curve")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_xgboost_shap(
    fitted: dict[str, Pipeline],
    X_test: pd.DataFrame,
    target: str,
    scope: str,
    shap_sample: int,
    random_state: int,
    table_out: Path,
    figure_out: Path,
) -> pd.DataFrame:
    if "xgboost" not in fitted:
        return pd.DataFrame(
            [
                {
                    "target": target,
                    "scope": scope,
                    "model": "xgboost",
                    "rank": pd.NA,
                    "feature": "shap_not_run",
                    "mean_abs_shap": pd.NA,
                    "note": "xgboost is not installed.",
                }
            ]
        )

    try:
        import shap
    except ImportError:
        shap = None

    pipeline = fitted["xgboost"]
    sample_n = min(shap_sample, len(X_test))
    sample = X_test.sample(n=sample_n, random_state=random_state) if len(X_test) > sample_n else X_test.copy()
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    names = get_feature_names(pipeline.named_steps["preprocess"])
    transformed_df = pd.DataFrame(transformed, columns=names)

    if shap is not None:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        shap_values = explainer.shap_values(transformed_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        note = ""
    else:
        import xgboost as xgb

        booster = pipeline.named_steps["model"].get_booster()
        matrix = xgb.DMatrix(transformed_df, feature_names=list(names))
        contributions = booster.predict(matrix, pred_contribs=True)
        shap_values = contributions[:, :-1]
        note = "Computed with XGBoost pred_contribs because the shap package is not installed."

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:50]
    rows = [
        {
            "target": target,
            "scope": scope,
            "model": "xgboost",
            "rank": rank,
            "feature": names[idx],
            "mean_abs_shap": float(mean_abs[idx]),
            "note": note,
        }
        for rank, idx in enumerate(order, start=1)
    ]

    table = pd.DataFrame(rows)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_out, index=False, encoding="utf-8-sig")

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    if shap is not None:
        shap.summary_plot(shap_values, transformed_df, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(figure_out, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plot_df = table.head(20).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(plot_df) + 1.5)))
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#4f6f52")
        ax.set_xlabel("Mean absolute SHAP contribution")
        ax.set_title("XGBoost SHAP top features")
        fig.tight_layout()
        fig.savefig(figure_out, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return table


def build_qc(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    scope: str,
    feature_set: str,
    performance: pd.DataFrame,
    features: list[str],
    max_rows: int | None,
    tune: bool,
    search_iter: int,
    cv_folds: int,
    tune_scoring: str,
) -> pd.DataFrame:
    rows = [
        {"qc_domain": "ml_serious_outcome", "metric": "target", "value": target, "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "scope", "value": scope, "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "feature_set", "value": feature_set, "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "analysis_n", "value": len(X), "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "positive_n", "value": int(y.sum()), "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "positive_percent", "value": float(y.mean() * 100), "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "feature_n", "value": len(features), "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "features_json", "value": json.dumps(features), "note": "Leakage-screened input features."},
        {"qc_domain": "ml_serious_outcome", "metric": "max_rows", "value": max_rows if max_rows else "", "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "hyperparameter_tuning", "value": tune, "note": "RandomizedSearchCV on training set only."},
        {"qc_domain": "ml_serious_outcome", "metric": "search_iter", "value": search_iter if tune else "", "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "cv_folds", "value": cv_folds if tune else "", "note": ""},
        {"qc_domain": "ml_serious_outcome", "metric": "tune_scoring", "value": tune_scoring if tune else "", "note": ""},
    ]
    for row in performance.itertuples():
        rows.append(
            {
                "qc_domain": "ml_serious_outcome",
                "metric": f"{row.model}_roc_auc",
                "value": row.roc_auc,
                "note": "",
            }
        )
        rows.append(
            {
                "qc_domain": "ml_serious_outcome",
                "metric": f"{row.model}_average_precision",
                "value": row.average_precision,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def parse_target_list(targets: list[str]) -> list[str]:
    selected: list[str] = []
    for item in targets:
        for target in item.split(","):
            target = target.strip()
            if target:
                selected.append(target)
    invalid = sorted(set(selected) - TARGET_OPTIONS)
    if invalid:
        raise ValueError(f"Unsupported targets: {invalid}. Choose from {sorted(TARGET_OPTIONS)}")
    return list(dict.fromkeys(selected))


def parse_model_list(models: list[str]) -> list[str]:
    selected: list[str] = []
    for item in models:
        for model in item.split(","):
            model = model.strip()
            if model:
                selected.append(model)
    invalid = sorted(set(selected) - MODEL_OPTIONS)
    if invalid:
        raise ValueError(f"Unsupported models: {invalid}. Choose from {sorted(MODEL_OPTIONS)}")
    return list(dict.fromkeys(selected))


def target_output_path(path: Path, target: str, selected_models: list[str], multiple_targets: bool) -> Path:
    if not multiple_targets and len(selected_models) != 1:
        return path
    model_part = f"_{selected_models[0]}" if len(selected_models) == 1 else ""
    return path.with_name(f"{path.stem}_{target}{model_part}{path.suffix}")


def run_target(args: argparse.Namespace, target: str, selected_models: list[str], multiple_targets: bool, tune: bool) -> None:
    performance_out = target_output_path(args.performance_out, target, selected_models, multiple_targets)
    importance_out = target_output_path(args.importance_out, target, selected_models, multiple_targets)
    tuning_out = target_output_path(args.tuning_out, target, selected_models, multiple_targets)
    shap_out = target_output_path(args.shap_out, target, selected_models, multiple_targets)
    qc_out = target_output_path(args.qc_out, target, selected_models, multiple_targets)
    curve_out = target_output_path(args.curve_out, target, selected_models, multiple_targets)
    shap_figure_out = target_output_path(args.shap_figure_out, target, selected_models, multiple_targets)

    df = read_main_dataset(args.main_dataset, target)
    X, y, _ = prepare_dataset(df, target, args.scope, args.feature_set, args.max_rows, args.random_state)
    if y.nunique() < 2:
        raise ValueError(f"Target has only one class after filtering: {target}, scope={args.scope}")

    performance, importance, tuning, fitted, split = fit_and_evaluate(
        X,
        y,
        target=target,
        scope=args.scope,
        selected_models=selected_models,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        tune=tune,
        search_iter=args.search_iter,
        cv_folds=args.cv_folds,
        tune_scoring=args.tune_scoring,
    )
    _, X_test, _, y_test = split
    shap_table = pd.DataFrame()
    if "xgboost" in fitted:
        shap_table = compute_xgboost_shap(
            fitted,
            X_test,
            target=target,
            scope=args.scope,
            shap_sample=args.shap_sample,
            random_state=args.random_state,
            table_out=shap_out,
            figure_out=shap_figure_out,
        )
    qc = build_qc(
        X,
        y,
        target,
        args.scope,
        args.feature_set,
        performance,
        list(X.columns),
        args.max_rows,
        tune=tune,
        search_iter=args.search_iter,
        cv_folds=args.cv_folds,
        tune_scoring=args.tune_scoring,
    )

    performance_out.parent.mkdir(parents=True, exist_ok=True)
    importance_out.parent.mkdir(parents=True, exist_ok=True)
    tuning_out.parent.mkdir(parents=True, exist_ok=True)
    qc_out.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(performance_out, index=False, encoding="utf-8-sig")
    importance.to_csv(importance_out, index=False, encoding="utf-8-sig")
    tuning.to_csv(tuning_out, index=False, encoding="utf-8-sig")
    if "xgboost" in fitted and not shap_out.exists():
        shap_out.parent.mkdir(parents=True, exist_ok=True)
        shap_table.to_csv(shap_out, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_out, index=False, encoding="utf-8-sig")
    plot_curves(fitted, X_test, y_test, curve_out)

    print(f"Wrote {performance_out}")
    print(f"Wrote {importance_out}")
    print(f"Wrote {tuning_out}")
    if "xgboost" in fitted:
        print(f"Wrote {shap_out}")
    print(f"Wrote {qc_out}")
    print(f"Wrote {curve_out}")
    if shap_figure_out.exists():
        print(f"Wrote {shap_figure_out}")
    print(f"Target: {target}")
    print(f"Models: {', '.join(selected_models)}")
    print(f"Scope: {args.scope}")
    print(f"Feature set: {args.feature_set}")
    print(f"Analysis rows: {len(X):,}")
    print(f"Positive cases: {int(y.sum()):,} ({y.mean() * 100:.2f}%)")
    print(f"Features used: {len(X.columns):,}")
    print(f"Hyperparameter tuning: {tune}")
    if tune:
        print(f"CV folds: {args.cv_folds}")
        print(f"Search iterations per model: {args.search_iter}")
        print(f"Tuning scoring: {args.tune_scoring}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Serious outcome targets to run. Accepts space-separated values or comma-separated values.",
    )
    parser.add_argument("--target", choices=sorted(TARGET_OPTIONS), default=None, help="Optional single-target shortcut.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to run. Accepts space-separated values or comma-separated values.",
    )
    parser.add_argument("--scope", choices=["sedative_fall", "strict_fall"], default="sedative_fall")
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SET_OPTIONS),
        default="full",
        help="Use 'no_reporting_structure' to exclude report time/source variables from model features.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified downsample for quick test runs.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning and use the base model settings.")
    parser.add_argument("--search-iter", type=int, default=20, help="RandomizedSearchCV candidates per model.")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--tune-scoring", default="average_precision")
    parser.add_argument("--shap-sample", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=20260608)
    parser.add_argument("--performance-out", type=Path, default=DEFAULT_PERFORMANCE_OUT)
    parser.add_argument("--importance-out", type=Path, default=DEFAULT_IMPORTANCE_OUT)
    parser.add_argument("--tuning-out", type=Path, default=DEFAULT_TUNING_OUT)
    parser.add_argument("--shap-out", type=Path, default=DEFAULT_SHAP_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--curve-out", type=Path, default=DEFAULT_CURVE_OUT)
    parser.add_argument("--shap-figure-out", type=Path, default=DEFAULT_SHAP_FIGURE_OUT)
    args = parser.parse_args()

    selected_targets = [args.target] if args.target else parse_target_list(args.targets)
    selected_models = parse_model_list(args.models)
    multiple_targets = len(selected_targets) > 1
    tune = not args.no_tune
    print(f"Targets: {', '.join(selected_targets)}")
    print(f"Models: {', '.join(selected_models)}")
    for target in selected_targets:
        print(f"\n=== Running target: {target} ===")
        run_target(args, target, selected_models, multiple_targets, tune)


if __name__ == "__main__":
    main()
