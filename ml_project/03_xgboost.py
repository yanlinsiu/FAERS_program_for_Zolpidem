from __future__ import annotations

import argparse

import pandas as pd
from sklearn.pipeline import Pipeline

from ml_common import (
    MODEL_FEATURES,
    build_preprocessor,
    evaluate_predictions,
    get_feature_names,
    load_modeling_frame,
    make_run_dir,
    resolve_dataset_bundle,
    sample_training_frame,
    save_json,
    save_split_summary,
    temporal_split,
)


def _positive_class_weight(y: pd.Series) -> float:
    positives = int(y.astype(int).sum())
    negatives = int(len(y) - positives)
    if positives <= 0:
        return 1.0
    return max(1.0, negatives / positives)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost baseline on FAERS global datasets.")
    parser.add_argument("--period-token", default=None, help="Dataset token such as 2004_2025.")
    parser.add_argument("--target-col", default="is_fall", help="Target column: is_fall, has_fall_related_broad, or serious.")
    parser.add_argument("--train-end-year", type=int, default=2023, help="Train on years <= this value.")
    parser.add_argument("--valid-year", type=int, default=2024, help="Validation year.")
    parser.add_argument("--test-year", type=int, default=2025, help="Test year.")
    parser.add_argument("--train-sample-n", type=int, default=300000, help="Optional sampled training size for faster baseline fitting.")
    parser.add_argument("--n-estimators", type=int, default=400, help="Number of boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=5, help="Tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "xgboost is not installed. Run `.\\.venv\\Scripts\\python -m pip install xgboost` first."
        ) from exc

    bundle = resolve_dataset_bundle(period_token=args.period_token)
    modeling_df = load_modeling_frame(bundle=bundle, target_col=args.target_col)
    splits = temporal_split(
        modeling_df,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )

    train_df = sample_training_frame(
        splits["train"],
        target_col=args.target_col,
        sample_n=args.train_sample_n,
        random_state=args.random_state,
    )
    valid_df = splits["valid"]
    test_df = splits["test"]

    scale_pos_weight = _positive_class_weight(train_df[args.target_col])

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=5,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(train_df[MODEL_FEATURES], train_df[args.target_col].astype(int))

    valid_scores = pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
    test_scores = pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]

    valid_metrics = evaluate_predictions(valid_df[args.target_col], pd.Series(valid_scores))
    test_metrics = evaluate_predictions(test_df[args.target_col], pd.Series(test_scores))

    run_dir = make_run_dir(
        model_name="xgboost",
        target_col=args.target_col,
        period_token=bundle.period_token,
    )

    importances = pd.DataFrame(
        {
            "feature": get_feature_names(pipeline),
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "caseid": test_df["caseid"].astype(str),
            "year": test_df["year"].astype(int),
            "target": test_df[args.target_col].astype(int),
            "predicted_probability": test_scores,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_split_summary(
        {
            "train_sampled": train_df,
            "valid": valid_df,
            "test": test_df,
        },
        target_col=args.target_col,
        output_path=run_dir / "split_summary.csv",
    )

    save_json(
        run_dir / "metrics.json",
        {
            "model": "xgboost",
            "target_col": args.target_col,
            "period_token": bundle.period_token,
            "signal_file": str(bundle.signal_file),
            "feature_file": str(bundle.feature_file),
            "train_end_year": args.train_end_year,
            "valid_year": args.valid_year,
            "test_year": args.test_year,
            "train_sample_n": args.train_sample_n,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "model_features": MODEL_FEATURES,
            "validation_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
    )

    print(f"Saved XGBoost outputs to: {run_dir}")


if __name__ == "__main__":
    main()
