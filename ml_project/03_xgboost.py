from __future__ import annotations

import argparse

import pandas as pd

from ml_common import (
    ExperimentConfig,
    SearchSpec,
    add_common_arguments,
    config_from_args,
    get_feature_names,
    print_run_summary,
    run_model_experiment,
    save_interpretation_summary,
    save_model_card,
    summarize_importance_highlights,
)


DISPLAY_NAME = "XGBoost"
MODEL_NAME = "xgboost"

SEARCH_SPEC = SearchSpec(
    strategy="random",
    param_space_by_mode={
        "fast": {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 5],
            "model__reg_lambda": [1.0, 5.0],
        },
        "full": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 5],
            "model__reg_lambda": [1.0, 5.0],
        },
    },
    n_iter_by_mode={
        "fast": 12,
        "full": 18,
    },
)


def _positive_class_weight(y: pd.Series) -> float:
    positives = int(y.astype(int).sum())
    negatives = int(len(y) - positives)
    if positives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def build_estimator(train_df: pd.DataFrame, config: ExperimentConfig):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - handled as runtime exit
        raise SystemExit(
            "xgboost is not installed. Run `.\\.venv\\Scripts\\python -m pip install xgboost` first."
        ) from exc

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=_positive_class_weight(train_df[config.target_col]),
        random_state=config.random_state,
        n_jobs=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tuned XGBoost model on FAERS global datasets."
    )
    add_common_arguments(
        parser,
        default_train_sample_n=400000,
        default_search_mode="full",
    )
    args = parser.parse_args()
    config = config_from_args(args)

    result = run_model_experiment(
        config=config,
        model_name=MODEL_NAME,
        display_name=DISPLAY_NAME,
        estimator_factory=build_estimator,
        search_spec=SEARCH_SPEC,
    )

    importances_df = pd.DataFrame(
        {
            "feature": get_feature_names(result.pipeline),
            "importance": result.pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances_df.to_csv(
        result.run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    feature_highlights = summarize_importance_highlights(
        importances_df,
        feature_col="feature",
        score_col="importance",
    )
    save_model_card(
        output_path=result.run_dir / "model_card.md",
        display_name=DISPLAY_NAME,
        model_name=MODEL_NAME,
        result=result,
        feature_highlights=feature_highlights,
        notes=[
            "XGBoost is the strongest nonlinear benchmark in this repository, but it remains an auxiliary model.",
            "The positive-class weight is derived from the training period only, so tuning stays leakage-safe.",
        ],
    )
    save_interpretation_summary(
        output_path=result.run_dir / "interpretation_summary.md",
        display_name=DISPLAY_NAME,
        model_name=MODEL_NAME,
        result=result,
        feature_highlights=feature_highlights,
        notes=[
            "XGBoost importance is a model-internal ranking and should be checked alongside the Logistic Regression coefficients.",
        ],
    )
    print_run_summary(
        display_name=DISPLAY_NAME,
        result=result,
        feature_highlights=feature_highlights,
    )

    print(f"Saved XGBoost outputs to: {result.run_dir}")


if __name__ == "__main__":
    main()
