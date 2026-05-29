$ErrorActionPreference = "Stop"

$CommonArgs = @(
    "--feature-version", "v2",
    "--period-token", "2004_2025",
    "--cohort", "all",
    "--target-col", "is_fall",
    "--train-end-year", "2023",
    "--valid-year", "2024",
    "--test-year", "2025",
    "--cv-folds", "3",
    "--search-mode", "fast",
    "--bootstrap-iterations", "500",
    "--random-state", "42"
)

python ml_project\01_logistic_regression.py @CommonArgs --feature-set core --train-sample-n 400000
python ml_project\01_logistic_regression.py @CommonArgs --feature-set enhanced --train-sample-n 400000

python ml_project\02_random_forest.py @CommonArgs --feature-set core --train-sample-n 400000
python ml_project\02_random_forest.py @CommonArgs --feature-set enhanced --train-sample-n 400000

python ml_project\03_xgboost.py @CommonArgs --feature-set core --train-sample-n 400000
python ml_project\03_xgboost.py @CommonArgs --feature-set enhanced --train-sample-n 400000
