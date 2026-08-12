from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FAERS_ROOT = PROJECT_DIR.parent
DEFAULT_CASE_BASE = PROJECT_DIR / "outputs" / "intermediate" / "01_elderly_case_base.parquet"
DEFAULT_ML_FEATURE = (
    FAERS_ROOT
    / "runs"
    / "mainline_2026-05-20"
    / "OUTPUT_ML"
    / "features_v2"
    / "datasets"
    / "ml_feature_v2_2004_2025.parquet"
)
DEFAULT_OUT = PROJECT_DIR / "outputs" / "intermediate" / "04_covariate_matrix.parquet"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "04_covariate_matrix_qc.csv"

COVARIATE_COLUMNS = [
    "caseid",
    "drug_n",
    "distinct_drug_n",
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

BOOL_COLUMNS = [
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

COUNT_COLUMNS = ["drug_n", "distinct_drug_n"]


def read_case_base(path: Path) -> pd.DataFrame:
    base = pd.read_parquet(path, columns=["caseid"])
    base["caseid"] = base["caseid"].astype(str)
    if base["caseid"].duplicated().any():
        raise ValueError("Case base has duplicated caseid values.")
    return base


def read_covariates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ML feature table not found: {path}")
    covariates = pd.read_parquet(path, columns=COVARIATE_COLUMNS)
    covariates["caseid"] = covariates["caseid"].astype(str)
    return covariates.drop_duplicates("caseid", keep="first")


def clean_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()
    for column in COUNT_COLUMNS:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").fillna(0).astype("int16")
    for column in BOOL_COLUMNS:
        matrix[column] = matrix[column].fillna(False).astype(bool)
    return matrix


def build_qc(matrix: pd.DataFrame, covariates: pd.DataFrame, missing_after_merge: int) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "covariate_matrix", "metric": metric, "value": value, "note": note})

    add("matrix_rows", len(matrix))
    add("duplicate_caseid_final", int(matrix["caseid"].duplicated().sum()))
    add("covariate_source_rows", len(covariates))
    add("covariate_source_unique_caseid", int(covariates["caseid"].nunique()))
    add("missing_after_merge_caseid", missing_after_merge)

    for column in COUNT_COLUMNS:
        add(f"{column}__mean", float(matrix[column].mean()))
        add(f"{column}__median", float(matrix[column].median()))
        add(f"{column}__max", int(matrix[column].max()))

    for column in BOOL_COLUMNS:
        add(f"{column}__true", int(matrix[column].sum()))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-base", type=Path, default=DEFAULT_CASE_BASE)
    parser.add_argument("--ml-feature", type=Path, default=DEFAULT_ML_FEATURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    case_base = read_case_base(args.case_base)
    covariates = read_covariates(args.ml_feature)
    matrix = case_base.merge(covariates, on="caseid", how="left")
    missing_after_merge = int(matrix["drug_n"].isna().sum())
    matrix = clean_matrix(matrix)

    output_columns = ["caseid", *COUNT_COLUMNS, *BOOL_COLUMNS]
    matrix = matrix[output_columns]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.out, index=False)
    build_qc(matrix, covariates, missing_after_merge).to_csv(args.qc_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.qc_out}")
    print(f"Covariate matrix rows: {len(matrix):,}")


if __name__ == "__main__":
    main()
