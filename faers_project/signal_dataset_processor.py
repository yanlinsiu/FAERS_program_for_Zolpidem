from pathlib import Path

import pandas as pd


REQUIRED_CASE_BASE_COLS = [
    "caseid",
    "age_group",
    "sex_clean",
    "year",
    "quarter",
]

REQUIRED_REAC_COLS = [
    "caseid",
    "is_fall",
]

REQUIRED_DRUG_EXPOSURE_COLS = [
    "caseid",
    "suspect_role_any",
    "suspect_role_any_ps",
    "is_zolpidem_suspect",
    "is_zolpidem_suspect_ps",
    "is_other_zdrug_suspect",
    "is_other_zdrug_suspect_ps",
    "target_drug_group",
    "target_drug_group_ps",
]

BOOL_COLS = [
    "is_fall",
    "suspect_role_any",
    "suspect_role_any_ps",
    "is_zolpidem_suspect",
    "is_zolpidem_suspect_ps",
    "is_other_zdrug_suspect",
    "is_other_zdrug_suspect_ps",
]

OPTIONAL_FALL_COLS = [
    "has_fall_related_broad",
    "fall_pt_list",
]


def _normalize_caseid(df: pd.DataFrame, col: str = "caseid") -> pd.DataFrame:
    out_df = df.copy()
    out_df[col] = out_df[col].where(out_df[col].notna(), "").astype(str).str.strip()
    out_df = out_df[out_df[col] != ""].copy()
    return out_df


def _assert_has_columns(df: pd.DataFrame, cols, table_name: str):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def _assert_caseid_unique(df: pd.DataFrame, table_name: str):
    if df["caseid"].duplicated().any():
        duplicated_caseids = df.loc[df["caseid"].duplicated(), "caseid"].unique()
        raise ValueError(
            f"{table_name} has duplicated caseid and cannot be merged: "
            f"{duplicated_caseids[:10].tolist()}"
        )


def process_signal_dataset(year, quarter, output_root):
    """
    Build case-level signal-mining dataset for downstream ROR/PRR/2x2 analysis.

    Input:
        - case_base_dataset_{year}{quarter}.parquet
        - reac_{year}{quarter}_case.parquet
        - drug_exposure_{year}{quarter}_case.parquet

    Output:
        - signal_dataset_{year}{quarter}.parquet
    """
    output_root = Path(output_root)
    quarter_lower = str(quarter).lower()

    case_base_file = output_root / f"case_base_dataset_{year}{quarter_lower}.parquet"
    reac_case_file = output_root / f"reac_{year}{quarter_lower}_case.parquet"
    drug_exposure_file = output_root / f"drug_exposure_{year}{quarter_lower}_case.parquet"

    if not case_base_file.exists():
        print(f"case_base dataset not found, building automatically: {case_base_file}")
        from demo_processor import process_demo

        process_demo(year, quarter, output_root)

    if not reac_case_file.exists():
        print(f"REAC case dataset not found, building automatically: {reac_case_file}")
        from reac_processor import process_reac

        process_reac(year, quarter, output_root)

    if not drug_exposure_file.exists():
        print(
            "drug_exposure case dataset not found, building automatically: "
            f"{drug_exposure_file}"
        )
        from drug_exposure_processor import process_drug_exposure

        process_drug_exposure(year, quarter, output_root)

    if not case_base_file.exists():
        raise FileNotFoundError(f"file not found: {case_base_file}")
    if not reac_case_file.exists():
        raise FileNotFoundError(f"file not found: {reac_case_file}")
    if not drug_exposure_file.exists():
        raise FileNotFoundError(f"file not found: {drug_exposure_file}")

    case_base_df = pd.read_parquet(case_base_file)
    reac_case_df = pd.read_parquet(reac_case_file)
    drug_exposure_df = pd.read_parquet(drug_exposure_file)

    if "suspect_role_any_ps" not in drug_exposure_df.columns and "suspect_role_any" in drug_exposure_df.columns:
        drug_exposure_df["suspect_role_any_ps"] = drug_exposure_df["suspect_role_any"]
    if (
        "is_zolpidem_suspect_ps" not in drug_exposure_df.columns
        and "is_zolpidem_suspect" in drug_exposure_df.columns
    ):
        drug_exposure_df["is_zolpidem_suspect_ps"] = drug_exposure_df["is_zolpidem_suspect"]
    if (
        "is_other_zdrug_suspect_ps" not in drug_exposure_df.columns
        and "is_other_zdrug_suspect" in drug_exposure_df.columns
    ):
        drug_exposure_df["is_other_zdrug_suspect_ps"] = drug_exposure_df["is_other_zdrug_suspect"]
    if "target_drug_group_ps" not in drug_exposure_df.columns and "target_drug_group" in drug_exposure_df.columns:
        drug_exposure_df["target_drug_group_ps"] = drug_exposure_df["target_drug_group"]

    _assert_has_columns(case_base_df, REQUIRED_CASE_BASE_COLS, "case_base_dataset")
    _assert_has_columns(reac_case_df, REQUIRED_REAC_COLS, "reac_case")
    _assert_has_columns(
        drug_exposure_df, REQUIRED_DRUG_EXPOSURE_COLS, "drug_exposure_case"
    )

    case_base_keep_cols = REQUIRED_CASE_BASE_COLS.copy()
    if "serious" in case_base_df.columns:
        case_base_keep_cols.append("serious")

    reac_keep_cols = REQUIRED_REAC_COLS + [
        col for col in OPTIONAL_FALL_COLS if col in reac_case_df.columns
    ]

    case_base_df = _normalize_caseid(case_base_df[case_base_keep_cols])
    reac_case_df = _normalize_caseid(reac_case_df[reac_keep_cols])
    drug_exposure_df = _normalize_caseid(drug_exposure_df[REQUIRED_DRUG_EXPOSURE_COLS])

    _assert_caseid_unique(case_base_df, "case_base_dataset")
    _assert_caseid_unique(reac_case_df, "reac_case")
    _assert_caseid_unique(drug_exposure_df, "drug_exposure_case")

    signal_df = case_base_df.merge(reac_case_df, on="caseid", how="left")
    signal_df = signal_df.merge(drug_exposure_df, on="caseid", how="left")

    for col in BOOL_COLS:
        signal_df[col] = signal_df[col].fillna(False).astype(bool)

    if "has_fall_related_broad" in signal_df.columns:
        signal_df["has_fall_related_broad"] = (
            signal_df["has_fall_related_broad"].fillna(False).astype(bool)
        )

    signal_df["age_group"] = (
        signal_df["age_group"].where(signal_df["age_group"].notna(), "unknown")
        .astype(str)
        .str.strip()
    )
    signal_df.loc[signal_df["age_group"] == "", "age_group"] = "unknown"

    signal_df["sex_clean"] = (
        signal_df["sex_clean"].where(signal_df["sex_clean"].notna(), "unknown")
        .astype(str)
        .str.strip()
    )
    signal_df.loc[signal_df["sex_clean"] == "", "sex_clean"] = "unknown"

    signal_df["target_drug_group"] = (
        signal_df["target_drug_group"]
        .where(signal_df["target_drug_group"].notna(), "no_suspect_drug")
        .astype(str)
        .str.strip()
    )
    signal_df.loc[
        signal_df["target_drug_group"] == "", "target_drug_group"
    ] = "no_suspect_drug"
    signal_df["target_drug_group_ps"] = (
        signal_df["target_drug_group_ps"]
        .where(signal_df["target_drug_group_ps"].notna(), "no_suspect_drug")
        .astype(str)
        .str.strip()
    )
    signal_df.loc[
        signal_df["target_drug_group_ps"] == "", "target_drug_group_ps"
    ] = "no_suspect_drug"

    if "fall_pt_list" in signal_df.columns:
        signal_df["fall_pt_list"] = (
            signal_df["fall_pt_list"].where(signal_df["fall_pt_list"].notna(), "")
        )

    signal_df["year"] = pd.to_numeric(signal_df["year"], errors="coerce").astype("Int64")
    signal_df["quarter"] = (
        signal_df["quarter"].where(signal_df["quarter"].notna(), str(quarter).upper())
        .astype(str)
        .str.upper()
        .str.strip()
    )

    final_cols = [
        "caseid",
        "is_fall",
        "is_zolpidem_suspect",
        "is_zolpidem_suspect_ps",
        "is_other_zdrug_suspect",
        "is_other_zdrug_suspect_ps",
        "suspect_role_any",
        "suspect_role_any_ps",
        "target_drug_group",
        "target_drug_group_ps",
        "age_group",
        "sex_clean",
        "year",
        "quarter",
    ]

    if "serious" in signal_df.columns:
        final_cols.append("serious")
    if "has_fall_related_broad" in signal_df.columns:
        final_cols.append("has_fall_related_broad")
    if "fall_pt_list" in signal_df.columns:
        final_cols.append("fall_pt_list")

    signal_df = signal_df[final_cols]

    _assert_caseid_unique(signal_df, "signal_dataset")

    output_file = output_root / f"signal_dataset_{year}{quarter_lower}.parquet"
    signal_df.to_parquet(output_file, index=False)

    print("signal_dataset rows:", len(signal_df))
    print("fall cases (strict):", int(signal_df["is_fall"].sum()))
    print("zolpidem suspect cases:", int(signal_df["is_zolpidem_suspect"].sum()))
    print("other z-drug suspect cases:", int(signal_df["is_other_zdrug_suspect"].sum()))
    print(f"saved: {output_file}")

    return signal_df
