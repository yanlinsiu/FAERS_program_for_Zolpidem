from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
FAERS_ROOT = PROJECT_DIR.parent
DEFAULT_CASE_BASE = PROJECT_DIR / "outputs" / "intermediate" / "01_elderly_case_base.parquet"
DEFAULT_REAC_GLOB = (
    FAERS_ROOT
    / "archive_old_outputs"
    / "results_before_20260525"
    / "OUTPUT_COUNTRY"
    / "*"
    / "reac_*_case.parquet"
)
DEFAULT_OUTC_GLOB = (
    FAERS_ROOT
    / "archive_old_outputs"
    / "results_before_20260525"
    / "OUTPUT_COUNTRY"
    / "*"
    / "outc_*_case.parquet"
)
DEFAULT_ML_FEATURE = (
    FAERS_ROOT
    / "runs"
    / "mainline_2026-05-20"
    / "OUTPUT_ML"
    / "features_v2"
    / "datasets"
    / "ml_feature_v2_2004_2025.parquet"
)
DEFAULT_OUT = PROJECT_DIR / "outputs" / "intermediate" / "03_outcome_phenotype_matrix.parquet"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "03_outcome_phenotype_qc.csv"
DEFAULT_DEFINITIONS_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s2_outcome_phenotype_definitions.csv"
DEFAULT_DEFINITIONS = PROJECT_DIR / "configs" / "outcome_definitions.csv"

REAC_COLUMNS = ["caseid", "is_fall_narrow", "fall_narrow_pt_count", "is_fall_broad", "fall_pt_list"]
OUTC_COLUMNS = [
    "caseid",
    "is_death",
    "is_life_threatening",
    "is_hospitalization",
    "is_required_intervention",
    "is_disability",
    "is_congenital_anomaly",
    "is_other_serious",
    "is_serious_any",
]
ML_PHENO_COLUMNS = [
    "caseid",
    "pheno_sedation_somnolence",
    "pheno_consciousness_cognition",
    "pheno_dizziness_vertigo_syncope",
    "pheno_gait_balance_motor",
    "pheno_hypotension",
    "pheno_visual_disturbance",
]
PHENO_RENAME = {
    "pheno_sedation_somnolence": "pheno_sedation",
    "pheno_consciousness_cognition": "pheno_neurocognitive",
    "pheno_dizziness_vertigo_syncope": "pheno_dizziness_syncope",
    "pheno_gait_balance_motor": "pheno_gait_balance",
    "pheno_hypotension": "pheno_hypotension",
    "pheno_visual_disturbance": "pheno_visual_disturbance",
}


def find_annual_files(pattern: Path, prefix: str) -> list[Path]:
    files = []
    regex = re.compile(rf"{prefix}_\d{{4}}_case\.parquet$", re.I)
    for path_text in glob.glob(str(pattern)):
        path = Path(path_text)
        if regex.fullmatch(path.name):
            files.append(path)
    return sorted(files)


def read_case_base(path: Path) -> pd.DataFrame:
    base = pd.read_parquet(path, columns=["caseid"])
    base["caseid"] = base["caseid"].astype(str)
    if base["caseid"].duplicated().any():
        raise ValueError("Case base has duplicated caseid values.")
    return base


def read_reac(files: list[Path]) -> pd.DataFrame:
    chunks = []
    for path in files:
        print(f"Reading {path}")
        chunk = pd.read_parquet(path, columns=REAC_COLUMNS)
        chunk["caseid"] = chunk["caseid"].astype(str)
        chunks.append(chunk)
    if not chunks:
        raise FileNotFoundError("No annual REAC case parquet files found.")

    reac = pd.concat(chunks, ignore_index=True)
    reac = reac.drop_duplicates("caseid", keep="first")
    return reac.rename(
        columns={
            "is_fall_narrow": "strict_fall",
            "fall_narrow_pt_count": "fall_pt_count",
            "is_fall_broad": "broad_fall",
        }
    )


def read_outc(files: list[Path]) -> pd.DataFrame:
    chunks = []
    for path in files:
        print(f"Reading {path}")
        chunk = pd.read_parquet(path, columns=OUTC_COLUMNS)
        chunk["caseid"] = chunk["caseid"].astype(str)
        chunks.append(chunk)
    if not chunks:
        raise FileNotFoundError("No annual OUTC case parquet files found.")

    outc = pd.concat(chunks, ignore_index=True)
    outc = outc.drop_duplicates("caseid", keep="first")
    return outc.rename(
        columns={
            "is_serious_any": "serious_any",
            "is_death": "serious_death",
            "is_hospitalization": "serious_hospitalization",
            "is_disability": "serious_disability",
            "is_life_threatening": "serious_life_threatening",
            "is_required_intervention": "serious_required_intervention",
            "is_congenital_anomaly": "serious_congenital_anomaly",
            "is_other_serious": "serious_other",
        }
    )


def read_phenotypes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ML feature table not found: {path}")

    pheno = pd.read_parquet(path, columns=ML_PHENO_COLUMNS)
    pheno["caseid"] = pheno["caseid"].astype(str)
    pheno = pheno.drop_duplicates("caseid", keep="first")
    return pheno.rename(columns=PHENO_RENAME)


def fill_bool_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = False
        df[column] = df[column].fillna(False).astype(bool)
    return df


def build_qc(matrix: pd.DataFrame, reac: pd.DataFrame, outc: pd.DataFrame, pheno: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "outcome_phenotype", "metric": metric, "value": value, "note": note})

    add("matrix_rows", len(matrix))
    add("duplicate_caseid_final", int(matrix["caseid"].duplicated().sum()))
    add("reac_cases_matched_elderly", int(reac["caseid"].nunique()))
    add("outc_cases_matched_elderly", int(outc["caseid"].nunique()))
    add("phenotype_cases_matched_elderly", int(pheno["caseid"].nunique()))

    for column in [
        "strict_fall",
        "broad_fall",
        "pheno_sedation",
        "pheno_neurocognitive",
        "pheno_dizziness_syncope",
        "pheno_gait_balance",
        "pheno_hypotension",
        "pheno_visual_disturbance",
        "serious_any",
        "serious_death",
        "serious_hospitalization",
        "serious_disability",
        "serious_life_threatening",
    ]:
        add(f"{column}__true", int(matrix[column].sum()))

    add("missing_fall_pt_list", int(matrix["fall_pt_list"].isna().sum()))
    add("fall_pt_count_total", int(matrix["fall_pt_count"].sum()))
    add(
        "strict_fall_with_any_phenotype",
        int(
            (
                matrix["strict_fall"]
                & matrix[
                    [
                        "pheno_sedation",
                        "pheno_neurocognitive",
                        "pheno_dizziness_syncope",
                        "pheno_gait_balance",
                        "pheno_hypotension",
                    ]
                ].any(axis=1)
            ).sum()
        ),
    )
    return pd.DataFrame(rows)


def export_definitions(input_path: Path, output_path: Path) -> None:
    definitions = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    definitions.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-base", type=Path, default=DEFAULT_CASE_BASE)
    parser.add_argument("--reac-glob", type=Path, default=DEFAULT_REAC_GLOB)
    parser.add_argument("--outc-glob", type=Path, default=DEFAULT_OUTC_GLOB)
    parser.add_argument("--ml-feature", type=Path, default=DEFAULT_ML_FEATURE)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--definitions-out", type=Path, default=DEFAULT_DEFINITIONS_OUT)
    args = parser.parse_args()

    case_base = read_case_base(args.case_base)
    reac = read_reac(find_annual_files(args.reac_glob, "reac"))
    outc = read_outc(find_annual_files(args.outc_glob, "outc"))
    pheno = read_phenotypes(args.ml_feature)

    matrix = case_base.merge(reac, on="caseid", how="left")
    matrix = matrix.merge(pheno, on="caseid", how="left")
    matrix = matrix.merge(outc, on="caseid", how="left")

    bool_columns = [
        "strict_fall",
        "broad_fall",
        "pheno_sedation",
        "pheno_neurocognitive",
        "pheno_dizziness_syncope",
        "pheno_gait_balance",
        "pheno_hypotension",
        "pheno_visual_disturbance",
        "serious_any",
        "serious_death",
        "serious_hospitalization",
        "serious_disability",
        "serious_life_threatening",
        "serious_required_intervention",
        "serious_congenital_anomaly",
        "serious_other",
    ]
    matrix = fill_bool_columns(matrix, bool_columns)
    matrix["fall_pt_count"] = matrix["fall_pt_count"].fillna(0).astype("int16")
    matrix["fall_pt_list"] = matrix["fall_pt_list"].fillna("")

    output_columns = [
        "caseid",
        "strict_fall",
        "broad_fall",
        "fall_pt_count",
        "fall_pt_list",
        "pheno_sedation",
        "pheno_neurocognitive",
        "pheno_dizziness_syncope",
        "pheno_gait_balance",
        "pheno_hypotension",
        "pheno_visual_disturbance",
        "serious_any",
        "serious_death",
        "serious_hospitalization",
        "serious_disability",
        "serious_life_threatening",
        "serious_required_intervention",
        "serious_congenital_anomaly",
        "serious_other",
    ]
    matrix = matrix[output_columns]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.out, index=False)
    build_qc(matrix, reac, outc, pheno).to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    export_definitions(args.definitions, args.definitions_out)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.definitions_out}")
    print(f"Outcome/phenotype matrix rows: {len(matrix):,}")
    print(f"Strict fall cases: {int(matrix['strict_fall'].sum()):,}")
    print(f"Serious outcome cases: {int(matrix['serious_any'].sum()):,}")


if __name__ == "__main__":
    main()
