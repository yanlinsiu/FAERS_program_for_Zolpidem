from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

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
    summarize_logistic_highlights,
)


DISPLAY_NAME = "Logistic Regression"
MODEL_NAME = "logistic_regression"

SEARCH_SPEC = SearchSpec(
    strategy="grid",
    param_space_by_mode={
        "fast": [
            {
                "model__penalty": ["elasticnet"],
                "model__l1_ratio": [0.0],
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
                "model__class_weight": [None, "balanced"],
            },
        ],
        "full": [
            {
                "model__penalty": ["elasticnet"],
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "model__l1_ratio": [0.0, 0.15, 0.3, 0.5, 0.7, 1.0],
                "model__class_weight": [None, "balanced"],
            },
        ],
    },
)


def build_estimator(_: pd.DataFrame, config: ExperimentConfig) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.0,
        max_iter=10000,
        random_state=config.random_state,
    )


def save_logistic_inference(
    *,
    result,
    coefficients_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if top_n <= 0:
        return pd.DataFrame()

    top_features = (
        coefficients_df.assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(top_n)
        ["feature"]
        .tolist()
    )
    feature_names = get_feature_names(result.pipeline)
    feature_index = {feature: idx for idx, feature in enumerate(feature_names)}
    selected_indices = [feature_index[feature] for feature in top_features]

    preprocessor = result.pipeline.named_steps["preprocessor"]
    x_sparse = preprocessor.transform(result.train_df[preprocessor.feature_names_in_])
    x_selected = x_sparse[:, selected_indices].toarray()
    x_selected = sm.add_constant(x_selected, has_constant="add")
    y = result.train_df[result.config.target_col].astype(int).to_numpy()

    model = sm.GLM(y, x_selected, family=sm.families.Binomial())
    fitted = model.fit(maxiter=100, disp=0, cov_type="HC0")

    params = np.asarray(fitted.params)
    pvalues = np.asarray(fitted.pvalues)
    conf_int = np.asarray(fitted.conf_int(alpha=0.05))

    rows = []
    for offset, feature in enumerate(top_features, start=1):
        coef = float(params[offset])
        ci_low = float(conf_int[offset, 0])
        ci_high = float(conf_int[offset, 1])
        rows.append(
            {
                "feature": feature,
                "inference_model_coefficient": coef,
                "odds_ratio": float(np.exp(coef)),
                "or_95ci_low": float(np.exp(ci_low)),
                "or_95ci_high": float(np.exp(ci_high)),
                "p_value": float(pvalues[offset]),
                "selected_by": "top_abs_regularized_coefficient",
                "inference_note": (
                    "Post-hoc unpenalized GLM on selected top features; "
                    "use as explanatory support, not as the tuned ML model."
                ),
            }
        )

    inference_df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
    inference_df.to_csv(
        result.run_dir / "logistic_inference_top_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return inference_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tuned logistic regression model on FAERS global datasets."
    )
    add_common_arguments(
        parser,
        default_train_sample_n=0,
        default_search_mode="fast",
    )
    parser.add_argument(
        "--inference-top-n",
        type=int,
        default=30,
        help=(
            "Fit a post-hoc unpenalized GLM for the top N absolute regularized "
            "coefficients and save OR, 95% CI, and P values. Use 0 to skip."
        ),
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

    model = result.pipeline.named_steps["model"]
    coefficients_df = pd.DataFrame(
        {
            "feature": get_feature_names(result.pipeline),
            "coefficient": model.coef_[0],
        }
    ).sort_values("coefficient", ascending=False)
    coefficients_df["odds_ratio"] = np.exp(coefficients_df["coefficient"])

    coefficients_df.to_csv(
        result.run_dir / "coefficients.csv", index=False, encoding="utf-8-sig"
    )
    coefficients_df[["feature", "odds_ratio"]].sort_values(
        "odds_ratio", ascending=False
    ).to_csv(result.run_dir / "odds_ratios.csv", index=False, encoding="utf-8-sig")

    if args.inference_top_n > 0:
        save_logistic_inference(
            result=result,
            coefficients_df=coefficients_df,
            top_n=args.inference_top_n,
        )

    feature_highlights = summarize_logistic_highlights(coefficients_df)
    save_model_card(
        output_path=result.run_dir / "model_card.md",
        display_name=DISPLAY_NAME,
        model_name=MODEL_NAME,
        result=result,
        feature_highlights=feature_highlights,
        notes=[
            "Logistic regression is the main narrative model because it is easier to explain in a research report.",
            "Odds ratios here come from model coefficients and should be interpreted as predictive associations only.",
        ],
    )
    save_interpretation_summary(
        output_path=result.run_dir / "interpretation_summary.md",
        display_name=DISPLAY_NAME,
        model_name=MODEL_NAME,
        result=result,
        feature_highlights=feature_highlights,
        notes=[
            "Positive coefficients mean the model gives higher predicted probability when that encoded feature is present or larger.",
            "Negative coefficients mean the model gives lower predicted probability in the same predictive sense.",
        ],
    )
    print_run_summary(
        display_name=DISPLAY_NAME,
        result=result,
        feature_highlights=feature_highlights,
    )

    print(f"Saved Logistic Regression outputs to: {result.run_dir}")


if __name__ == "__main__":
    main()
