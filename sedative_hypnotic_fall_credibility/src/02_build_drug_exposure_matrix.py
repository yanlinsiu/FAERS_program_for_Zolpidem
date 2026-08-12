from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CASE_BASE = PROJECT_DIR / "outputs" / "intermediate" / "01_elderly_case_base.parquet"
DEFAULT_MATCHES = PROJECT_DIR / "outputs" / "intermediate" / "02a_drug_dictionary_matches.parquet"
DEFAULT_MASTER = PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"
DEFAULT_OUT = PROJECT_DIR / "outputs" / "intermediate" / "02_drug_exposure_matrix.parquet"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "02_drug_exposure_qc.csv"
DEFAULT_COUNTS_OUT = PROJECT_DIR / "outputs" / "qc" / "02_drug_exposure_counts_by_drug.csv"

SUSPECT_ROLES = {"PS", "SS"}
PS_ROLE = {"PS"}


def exposure_column(drug_key: str, suffix: str) -> str:
    return f"exposure_{drug_key}_{suffix}"


def load_case_base(path: Path) -> pd.DataFrame:
    base = pd.read_parquet(path, columns=["caseid"])
    base["caseid"] = base["caseid"].astype(str)
    if base["caseid"].duplicated().any():
        raise ValueError("Case base has duplicated caseid values.")
    return base


def load_matches(path: Path, elderly_caseids: set[str]) -> pd.DataFrame:
    matches = pd.read_parquet(
        path,
        columns=["caseid", "role_cod", "matched_drug_key", "matched_drug_group", "year"],
    )
    matches["caseid"] = matches["caseid"].astype(str)
    matches["role_cod_clean"] = matches["role_cod"].fillna("").astype(str).str.upper().str.strip()
    return matches[matches["caseid"].isin(elderly_caseids)].copy()


def add_individual_exposures(matrix: pd.DataFrame, matches: pd.DataFrame, drug_keys: list[str]) -> pd.DataFrame:
    matrix = matrix.copy()
    for drug_key in drug_keys:
        matrix[exposure_column(drug_key, "ps_ss")] = False
        matrix[exposure_column(drug_key, "ps_only")] = False

    ps_ss = matches[matches["role_cod_clean"].isin(SUSPECT_ROLES)]
    ps_only = matches[matches["role_cod_clean"].isin(PS_ROLE)]

    for drug_key in drug_keys:
        ps_ss_cases = ps_ss.loc[ps_ss["matched_drug_key"].eq(drug_key), "caseid"].unique()
        ps_only_cases = ps_only.loc[ps_only["matched_drug_key"].eq(drug_key), "caseid"].unique()
        matrix.loc[matrix["caseid"].isin(ps_ss_cases), exposure_column(drug_key, "ps_ss")] = True
        matrix.loc[matrix["caseid"].isin(ps_only_cases), exposure_column(drug_key, "ps_only")] = True
    return matrix


def add_group_exposures(matrix: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()
    for suffix in ["ps_ss", "ps_only"]:
        for group in ["z_drug", "benzodiazepine", "orexin_antagonist", "other_insomnia_related"]:
            drug_keys = master.loc[master["drug_group"].eq(group), "drug_key"].tolist()
            columns = [exposure_column(drug_key, suffix) for drug_key in drug_keys]
            matrix[exposure_column(group, suffix)] = matrix[columns].any(axis=1)

        other_z_columns = [
            exposure_column(drug_key, suffix)
            for drug_key in ["eszopiclone", "zaleplon", "zopiclone"]
            if exposure_column(drug_key, suffix) in matrix.columns
        ]
        matrix[exposure_column("other_z_drug", suffix)] = matrix[other_z_columns].any(axis=1)
    return matrix


def add_mixed_exposure_flags(matrix: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()
    all_drugs = master["drug_key"].tolist()
    z_drugs = master.loc[master["drug_group"].eq("z_drug"), "drug_key"].tolist()
    groups = ["z_drug", "benzodiazepine", "orexin_antagonist", "other_insomnia_related"]

    for suffix in ["ps_ss", "ps_only"]:
        drug_columns = [exposure_column(drug_key, suffix) for drug_key in all_drugs]
        z_columns = [exposure_column(drug_key, suffix) for drug_key in z_drugs]
        group_columns = [exposure_column(group, suffix) for group in groups]

        matrix[f"n_sedative_hypnotic_drugs_{suffix}"] = matrix[drug_columns].sum(axis=1).astype("int16")
        matrix[f"n_sedative_hypnotic_groups_{suffix}"] = matrix[group_columns].sum(axis=1).astype("int8")
        matrix[f"mixed_z_drug_{suffix}"] = matrix[z_columns].sum(axis=1).gt(1)
        matrix[f"mixed_sedative_hypnotic_group_{suffix}"] = matrix[f"n_sedative_hypnotic_groups_{suffix}"].gt(1)
        matrix[f"z_drug_plus_benzo_{suffix}"] = (
            matrix[exposure_column("z_drug", suffix)]
            & matrix[exposure_column("benzodiazepine", suffix)]
        )
    return matrix


def build_counts(matches: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in master.iterrows():
        drug_key = row["drug_key"]
        subset = matches[matches["matched_drug_key"].eq(drug_key)]
        ps_ss = subset[subset["role_cod_clean"].isin(SUSPECT_ROLES)]
        ps_only = subset[subset["role_cod_clean"].isin(PS_ROLE)]
        rows.append(
            {
                "drug_key": drug_key,
                "drug_group": row["drug_group"],
                "analysis_role": row["analysis_role"],
                "main_analysis_candidate": row["main_analysis_candidate"],
                "n_matched_rows_any_role_elderly": int(len(subset)),
                "n_cases_any_role_elderly": int(subset["caseid"].nunique()),
                "n_matched_rows_ps_ss_elderly": int(len(ps_ss)),
                "n_cases_ps_ss_elderly": int(ps_ss["caseid"].nunique()),
                "n_matched_rows_ps_only_elderly": int(len(ps_only)),
                "n_cases_ps_only_elderly": int(ps_only["caseid"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_qc(matrix: pd.DataFrame, matches: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "drug_exposure", "metric": metric, "value": value, "note": note})

    add("elderly_case_rows", len(matrix))
    add("duplicate_caseid_final", int(matrix["caseid"].duplicated().sum()))
    add("matched_drug_rows_elderly_any_role", len(matches))
    add("matched_drug_cases_elderly_any_role", int(matches["caseid"].nunique()))
    add("cases_any_sedative_hypnotic_ps_ss", int(matrix["n_sedative_hypnotic_drugs_ps_ss"].gt(0).sum()))
    add("cases_any_sedative_hypnotic_ps_only", int(matrix["n_sedative_hypnotic_drugs_ps_only"].gt(0).sum()))

    for column in [
        "exposure_zolpidem_ps_ss",
        "exposure_zolpidem_ps_only",
        "exposure_other_z_drug_ps_ss",
        "exposure_other_z_drug_ps_only",
        "exposure_z_drug_ps_ss",
        "exposure_benzodiazepine_ps_ss",
        "exposure_orexin_antagonist_ps_ss",
        "exposure_other_insomnia_related_ps_ss",
        "mixed_z_drug_ps_ss",
        "mixed_sedative_hypnotic_group_ps_ss",
        "z_drug_plus_benzo_ps_ss",
    ]:
        add(f"{column}__true", int(matrix[column].sum()))

    low_count = counts[counts["n_cases_ps_ss_elderly"].lt(100)]
    add("drug_keys_with_ps_ss_cases_lt_100", int(len(low_count)), "Small-count drugs should stay exploratory.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-base", type=Path, default=DEFAULT_CASE_BASE)
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    parser.add_argument("--counts-out", type=Path, default=DEFAULT_COUNTS_OUT)
    args = parser.parse_args()

    case_base = load_case_base(args.case_base)
    master = pd.read_csv(args.master)
    drug_keys = master["drug_key"].tolist()
    matches = load_matches(args.matches, set(case_base["caseid"]))

    matrix = add_individual_exposures(case_base, matches, drug_keys)
    matrix = add_group_exposures(matrix, master)
    matrix = add_mixed_exposure_flags(matrix, master)

    counts = build_counts(matches, master)
    qc = build_qc(matrix, matches, counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.qc_out.parent.mkdir(parents=True, exist_ok=True)
    args.counts_out.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.out, index=False)
    qc.to_csv(args.qc_out, index=False, encoding="utf-8-sig")
    counts.to_csv(args.counts_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.qc_out}")
    print(f"Wrote {args.counts_out}")
    print(f"Exposure matrix rows: {len(matrix):,}")
    print(f"Any sedative-hypnotic PS+SS cases: {int(matrix['n_sedative_hypnotic_drugs_ps_ss'].gt(0).sum()):,}")


if __name__ == "__main__":
    main()
