from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml_common import (
    ExperimentConfig,
    SearchSpec,
    add_common_arguments,
    config_from_args,
    get_feature_names,
    run_model_experiment,
    save_model_card,
    summarize_importance_highlights,
)


DISPLAY_NAME = "Random Forest"
MODEL_NAME = "random_forest"

SEARCH_SPEC = SearchSpec(
    strategy="grid",
    param_space_by_mode={
        "fast": {
            "model__n_estimators": [200],
            "model__max_depth": [8, 12],
            "model__min_samples_leaf": [20, 50],
            "model__max_features": ["sqrt"],
            "model__class_weight": ["balanced_subsample"],
        },
        "full": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 12],
            "model__min_samples_leaf": [10, 20, 50],
            "model__max_features": ["sqrt", 0.5],
            "model__class_weight": ["balanced", "balanced_subsample"],
        },
    },
)


def build_estimator(_: pd.DataFrame, config: ExperimentConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_jobs=-1,
        random_state=config.random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tuned random forest model on FAERS global datasets."
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

    save_model_card(
        output_path=result.run_dir / "model_card.md",
        display_name=DISPLAY_NAME,
        model_name=MODEL_NAME,
        result=result,
        feature_highlights=summarize_importance_highlights(
            importances_df,
            feature_col="feature",
            score_col="importance",
        ),
        notes=[
            "Random Forest is used as a nonlinear benchmark against the main logistic regression model.",
            "Feature importance here is impurity-based and should be read as a rough ranking, not a causal explanation.",
        ],
    )

    print(f"Saved Random Forest outputs to: {result.run_dir}")


if __name__ == "__main__":
    main()
