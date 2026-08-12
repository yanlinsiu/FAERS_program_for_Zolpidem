from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASE_BASE = PROJECT_DIR / "outputs" / "intermediate" / "01_elderly_case_base.parquet"
DEFAULT_EXPOSURE = PROJECT_DIR / "outputs" / "intermediate" / "02_drug_exposure_matrix.parquet"
DEFAULT_OUTCOME = PROJECT_DIR / "outputs" / "intermediate" / "03_outcome_phenotype_matrix.parquet"
DEFAULT_COVARIATE = PROJECT_DIR / "outputs" / "intermediate" / "04_covariate_matrix.parquet"
DEFAULT_SPEC = PROJECT_DIR / "configs" / "main_analysis_dataset_spec.csv"
DEFAULT_OUT = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "05_main_analysis_dataset_qc.csv"
DEFAULT_FIELD_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "05_main_analysis_dataset_field_qc.csv"


def read_case_level_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} table not found: {path}")
    table = pd.read_parquet(path)
    if "caseid" not in table.columns:
        raise ValueError(f"{label} table has no caseid column: {path}")
    table["caseid"] = table["caseid"].astype(str)
    duplicated = int(table["caseid"].duplicated().sum())
    if duplicated:
        raise ValueError(f"{label} table has duplicated caseid values: {duplicated}")
    return table


def merge_tables(case_base: pd.DataFrame, exposure: pd.DataFrame, outcome: pd.DataFrame, covariate: pd.DataFrame) -> pd.DataFrame:
    merged = case_base.merge(exposure, on="caseid", how="left", validate="one_to_one")
    merged = merged.merge(outcome, on="caseid", how="left", validate="one_to_one")
    merged = merged.merge(covariate, on="caseid", how="left", validate="one_to_one")
    return merged


def add_eligibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["analysis_eligible_main"] = (
        df["caseid"].notna()
        & df["year"].notna()
        & df["age_group_3"].isin(["65-74", "75-84", ">=85"])
        & df["sex_clean"].isin(["M", "F"])
        & df["strict_fall"].notna()
        & df["exposure_zolpidem_ps_ss"].notna()
    )
    df["analysis_eligible_insomnia"] = df["analysis_eligible_main"] & df["indi_insomnia"]
    df["analysis_eligible_clean_ps_ss"] = (
        df["analysis_eligible_main"] & ~df["mixed_sedative_hypnotic_group_ps_ss"]
    )
    df["analysis_eligible_clean_ps_only"] = (
        df["analysis_eligible_main"] & ~df["mixed_sedative_hypnotic_group_ps_only"]
    )
    return df


def order_columns_by_spec(df: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    spec_columns = [column for column in spec["variable"].tolist() if column in df.columns]
    extra_columns = [column for column in df.columns if column not in set(spec_columns)]
    return df[spec_columns + extra_columns]


def build_field_qc(df: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = set(df.columns)
    for _, row in spec.iterrows():
        variable = row["variable"]
        present = variable in columns
        rows.append(
            {
                "variable": variable,
                "variable_group": row["variable_group"],
                "required": row["required"],
                "present_in_main_dataset": present,
                "missing_values": int(df[variable].isna().sum()) if present else pd.NA,
                "note": "" if present else "Missing from merged main dataset.",
            }
        )
    extra = sorted(columns - set(spec["variable"]))
    for variable in extra:
        rows.append(
            {
                "variable": variable,
                "variable_group": "extra_not_in_spec",
                "required": "no",
                "present_in_main_dataset": True,
                "missing_values": int(df[variable].isna().sum()),
                "note": "Present in merged table but not listed in spec.",
            }
        )
    return pd.DataFrame(rows)


def build_qc(df: pd.DataFrame, field_qc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "main_analysis_dataset", "metric": metric, "value": value, "note": note})

    add("main_dataset_rows", len(df))
    add("main_dataset_columns", len(df.columns))
    add("duplicate_caseid_final", int(df["caseid"].duplicated().sum()))
    add("required_spec_fields_missing", int(((field_qc["required"] == "yes") & (~field_qc["present_in_main_dataset"])).sum()))
    add("all_spec_fields_missing", int((~field_qc["present_in_main_dataset"]).sum()))

    for column in [
        "analysis_eligible_main",
        "analysis_eligible_insomnia",
        "analysis_eligible_clean_ps_ss",
        "analysis_eligible_clean_ps_only",
        "strict_fall",
        "exposure_zolpidem_ps_ss",
        "exposure_other_z_drug_ps_ss",
        "exposure_benzodiazepine_ps_ss",
        "exposure_orexin_antagonist_ps_ss",
        "exposure_other_insomnia_related_ps_ss",
        "mixed_sedative_hypnotic_group_ps_ss",
        "z_drug_plus_benzo_ps_ss",
        "serious_any",
    ]:
        add(f"{column}__true", int(df[column].sum()))

    add("strict_fall_and_zolpidem_ps_ss", int((df["strict_fall"] & df["exposure_zolpidem_ps_ss"]).sum()))
    add("strict_fall_and_other_z_drug_ps_ss", int((df["strict_fall"] & df["exposure_other_z_drug_ps_ss"]).sum()))
    add("strict_fall_and_benzodiazepine_ps_ss", int((df["strict_fall"] & df["exposure_benzodiazepine_ps_ss"]).sum()))
    add("strict_fall_and_orexin_antagonist_ps_ss", int((df["strict_fall"] & df["exposure_orexin_antagonist_ps_ss"]).sum()))
    add("strict_fall_and_other_insomnia_related_ps_ss", int((df["strict_fall"] & df["exposure_other_insomnia_related_ps_ss"]).sum()))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-base", type=Path, default=DEFAULT_CASE_BASE)
    parser.add_argument("--exposure", type=Path, default=DEFAULT_EXPOSURE)
    parser.add_argument("--outcome", type=Path, default=DEFAULT_OUTCOME)
    parser.add_argument("--covariate", type=Path, default=DEFAULT_COVARIATE)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--field-qc-out", type=Path, default=DEFAULT_FIELD_QC_OUT)
    args = parser.parse_args()

    case_base = read_case_level_table(args.case_base, "case base")
    exposure = read_case_level_table(args.exposure, "drug exposure")
    outcome = read_case_level_table(args.outcome, "outcome phenotype")
    covariate = read_case_level_table(args.covariate, "covariate")
    spec = pd.read_csv(args.spec)

    main_dataset = merge_tables(case_base, exposure, outcome, covariate)
    main_dataset = add_eligibility_flags(main_dataset)
    main_dataset = order_columns_by_spec(main_dataset, spec)

    field_qc = build_field_qc(main_dataset, spec)
    qc = build_qc(main_dataset, field_qc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    args.field_qc_out.parent.mkdir(parents=True, exist_ok=True)

    main_dataset.to_parquet(args.out, index=False)
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    field_qc.to_csv(args.field_qc_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.field_qc_out}")
    print(f"Main analysis dataset rows: {len(main_dataset):,}")
    print(f"Main analysis dataset columns: {len(main_dataset.columns):,}")
    print(f"Analysis eligible main: {int(main_dataset['analysis_eligible_main'].sum()):,}")


if __name__ == "__main__":
    main()
