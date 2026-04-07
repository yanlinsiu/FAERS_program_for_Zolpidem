from __future__ import annotations

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml_common import (
    MODEL_FEATURES,
    apply_platt_calibrator,
    bootstrap_metric_intervals,
    build_calibration_table,
    build_preprocessor,
    build_roc_table,
    evaluate_predictions,
    fit_platt_calibrator,
    get_feature_names,
    load_modeling_frame,
    make_run_dir,
    resolve_dataset_bundle,
    run_cross_validation,
    sample_training_frame,
    save_json,
    save_split_summary,
    select_threshold_by_youden,
    summarize_cv_metrics,
    temporal_split,
)


def build_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest baseline on FAERS global datasets."
    )
    parser.add_argument(
        "--period-token", default=None, help="Dataset token such as 2004_2025."
    )
    parser.add_argument(
        "--target-col",
        default="is_fall",
        help="Target column: is_fall, has_fall_related_broad, or serious.",
    )
    parser.add_argument(
        "--train-end-year", type=int, default=2023, help="Train on years <= this value."
    )
    parser.add_argument("--valid-year", type=int, default=2024, help="Validation year.")
    parser.add_argument("--test-year", type=int, default=2025, help="Test year.")
    parser.add_argument(
        "--train-sample-n",
        type=int,
        default=200000,
        help="Optional random sample size from the training period.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300, help="Number of trees."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation inside the training period.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations for final test-set confidence intervals.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    bundle = resolve_dataset_bundle(period_token=args.period_token)
    modeling_df = load_modeling_frame(bundle=bundle, target_col=args.target_col)
    splits = temporal_split(
        modeling_df,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )

    train_full_df = splits["train"]
    train_df = sample_training_frame(
        train_full_df,
        target_col=args.target_col,
        sample_n=args.train_sample_n,
        random_state=args.random_state,
    )
    valid_df = splits["valid"]
    test_df = splits["test"]

    cv_metrics = run_cross_validation(
        build_pipeline=lambda: build_pipeline(args.n_estimators, args.random_state),
        train_df=train_df,
        target_col=args.target_col,
        n_splits=args.cv_folds,
        random_state=args.random_state,
    )
    cv_summary = summarize_cv_metrics(cv_metrics)

    pipeline = build_pipeline(args.n_estimators, args.random_state)
    pipeline.fit(train_df[MODEL_FEATURES], train_df[args.target_col].astype(int))

    valid_raw_scores = pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
    test_raw_scores = pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]

    calibrator = fit_platt_calibrator(
        valid_df[args.target_col], valid_raw_scores, args.random_state
    )
    valid_scores = apply_platt_calibrator(calibrator, valid_raw_scores)
    test_scores = apply_platt_calibrator(calibrator, test_raw_scores)

    threshold_selection = select_threshold_by_youden(
        valid_df[args.target_col], valid_scores
    )
    optimal_threshold = threshold_selection["threshold"]

    validation_metrics = evaluate_predictions(
        valid_df[args.target_col], valid_scores, threshold=optimal_threshold
    )
    test_metrics = evaluate_predictions(
        test_df[args.target_col], test_scores, threshold=optimal_threshold
    )
    validation_metrics_raw = evaluate_predictions(
        valid_df[args.target_col], valid_raw_scores, threshold=0.5
    )
    test_metrics_raw = evaluate_predictions(
        test_df[args.target_col], test_raw_scores, threshold=0.5
    )

    valid_roc_df = build_roc_table(valid_df[args.target_col], valid_scores)
    test_roc_df = build_roc_table(test_df[args.target_col], test_scores)
    valid_calibration_df = build_calibration_table(
        valid_df[args.target_col], valid_scores
    )
    test_calibration_df = build_calibration_table(test_df[args.target_col], test_scores)
    bootstrap_df = bootstrap_metric_intervals(
        test_df[args.target_col],
        test_scores,
        threshold=optimal_threshold,
        n_bootstrap=args.bootstrap_iterations,
        random_state=args.random_state,
    )

    run_dir = make_run_dir(
        model_name="random_forest",
        target_col=args.target_col,
        period_token=bundle.period_token,
    )

    importances = pd.DataFrame(
        {
            "feature": get_feature_names(pipeline),
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(
        run_dir / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    cv_metrics.to_csv(run_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig")
    valid_roc_df.to_csv(
        run_dir / "validation_roc_curve.csv", index=False, encoding="utf-8-sig"
    )
    test_roc_df.to_csv(
        run_dir / "test_roc_curve.csv", index=False, encoding="utf-8-sig"
    )
    valid_calibration_df.to_csv(
        run_dir / "validation_calibration_curve.csv", index=False, encoding="utf-8-sig"
    )
    test_calibration_df.to_csv(
        run_dir / "test_calibration_curve.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap_df.to_csv(
        run_dir / "test_bootstrap_metrics.csv", index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(
        {
            "caseid": valid_df["caseid"].astype(str),
            "year": valid_df["year"].astype(int),
            "target": valid_df[args.target_col].astype(int),
            "predicted_probability_raw": valid_raw_scores,
            "predicted_probability_calibrated": valid_scores,
            "predicted_label_optimal": (valid_scores >= optimal_threshold).astype(int),
        }
    ).to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        {
            "caseid": test_df["caseid"].astype(str),
            "year": test_df["year"].astype(int),
            "target": test_df[args.target_col].astype(int),
            "predicted_probability_raw": test_raw_scores,
            "predicted_probability_calibrated": test_scores,
            "predicted_label_optimal": (test_scores >= optimal_threshold).astype(int),
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_split_summary(
        {
            "train_full": train_full_df,
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
            "model": "random_forest",
            "target_col": args.target_col,
            "period_token": bundle.period_token,
            "signal_file": str(bundle.signal_file),
            "feature_file": str(bundle.feature_file),
            "train_end_year": args.train_end_year,
            "valid_year": args.valid_year,
            "test_year": args.test_year,
            "train_sample_n": args.train_sample_n,
            "n_estimators": args.n_estimators,
            "cv_folds": args.cv_folds,
            "bootstrap_iterations": args.bootstrap_iterations,
            "model_features": MODEL_FEATURES,
            "cross_validation_summary": cv_summary,
            "threshold_selection": threshold_selection,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "validation_metrics_raw_threshold_0_5": validation_metrics_raw,
            "test_metrics_raw_threshold_0_5": test_metrics_raw,
            "calibration_method": "platt",
        },
    )

    print(f"Saved Random Forest outputs to: {run_dir}")


if __name__ == "__main__":
    main()
