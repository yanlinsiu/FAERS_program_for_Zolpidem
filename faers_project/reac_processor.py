import pandas as pd
from pathlib import Path

from utils import (
    attach_caseid_from_demo,
    build_file_path,
    ensure_required_columns,
    load_retained_demo_primaryids,
    read_faers_txt,
)
from config import RAW_ROOT


FALL_TERMS = {
    "FALL",
    "DROP ATTACKS",
}


def process_reac(year, quarter, output_root):
    """Process FAERS REAC and build case-level outcomes."""
    file_path = build_file_path(RAW_ROOT, year, quarter, "REAC")
    print(f"Processing file: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = read_faers_txt(file_path, dataset_name="REAC")
    df = attach_caseid_from_demo(df, RAW_ROOT, year, quarter, output_root=output_root)
    ensure_required_columns(df, ["primaryid", "caseid"], "REAC")

    if "pt" in df.columns:
        reaction_term_col = "pt"
    elif "reac_pt" in df.columns:
        reaction_term_col = "reac_pt"
    else:
        raise ValueError("REAC missing reaction term column: need pt or reac_pt")

    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    df["pt"] = (
        df[reaction_term_col]
        .where(df[reaction_term_col].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    retained_primaryids = load_retained_demo_primaryids(
        RAW_ROOT, year, quarter, output_root=output_root
    )
    df = df[df["primaryid"].isin(retained_primaryids)]
    df = df[df["caseid"] != ""]
    print("REAC event rows after DEMO-primaryid filter:", len(df))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    event_output_file = output_root / f"reac_event_{year}{quarter.lower()}.parquet"
    df[["caseid", "primaryid", "pt"]].to_parquet(event_output_file, index=False)
    print(f"Saved event-level REAC: {event_output_file}")

    df["is_fall_row"] = df["pt"].isin(FALL_TERMS)

    case_level_df = df.groupby("caseid", as_index=False).agg(
        is_fall=("is_fall_row", "max"),
        fall_pt_count=("is_fall_row", "sum"),
        all_reac_n=("pt", "size"),
    )

    fall_pt_list_df = (
        df.loc[df["is_fall_row"], ["caseid", "pt"]]
        .drop_duplicates()
        .groupby("caseid")["pt"]
        .apply(lambda s: "|".join(sorted(s)))
        .reset_index(name="fall_pt_list")
    )

    case_level_df = case_level_df.merge(fall_pt_list_df, on="caseid", how="left")
    case_level_df["fall_pt_list"] = (
        case_level_df["fall_pt_list"].where(case_level_df["fall_pt_list"].notna(), "")
    )

    case_level_df["is_fall"] = case_level_df["is_fall"].astype(bool)
    case_level_df["fall_pt_count"] = case_level_df[
        "fall_pt_count"
    ].astype(int)
    case_level_df["all_reac_n"] = case_level_df["all_reac_n"].astype(int)

    print("Case-level REAC rows:", len(case_level_df))
    print(
        "Fall cases (definite PT definition):",
        int(case_level_df["is_fall"].sum()),
    )

    output_file = output_root / f"reac_{year}{quarter.lower()}_case.parquet"
    case_level_df.to_parquet(output_file, index=False)

    print(f"Saved: {output_file}")

