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


NARROW_FALL_TERMS = {
    "FALL",
    "FALLING",
    "FALLING DOWN",
}

# Broad fall-related PT definition adapted from the reference paper and
# reconciled against FAERS terms that actually appear in the local data.
# We keep the paper-origin candidates for reproducibility and add a small
# number of close FAERS PT variants (e.g. GAIT INABILITY, VERTIGO POSITIONAL).
BROAD_FALL_TERMS = {
    # Narrow fall events
    "FALL",
    "FALLING",
    "FALLING DOWN",
    # Balance / gait
    "DISEQUILIBRIUM",
    "DISEQUILIBRIUM SYNDROME",
    "GAIT ABNORMAL",
    "GAIT ABNORMAL NOS",
    "GAIT DISORDER",
    "GAIT DISTURBANCE",
    "GAIT INABILITY",
    "GAIT INSTABILITY",
    "BALANCE DISORDER",
    # Vertigo / vestibular
    "VERTIGO",
    "VERTIGO (EXCL. DIZZINESS)",
    "VERTIGO AGGRAVATED",
    "VERTIGO CNS ORIGIN",
    "VERTIGO LABYRINTHINE",
    "VERTIGO POSITIONAL",
    "VESTIBULAR ABNORMALITIES",
    "VESTIBULAR DISORDER",
    "VESTIBULAR VERTIGO",
    # Visual impairment
    "VISUAL ACUITY DECREASED",
    "VISUAL ACUITY LOST",
    "VISUAL ACUITY REDUCED",
    "VISUAL DISTURBANCE",
    "VISUAL DISTURBANCE NOS",
    "VISUAL DISTURBANCES",
    "VISUAL IMPAIRMENT",
    # Hypotension / orthostatic hypotension
    "HYPOTENSION",
    "HYPOTENSION AGGRAVATED",
    "HYPOTENSION ORTHOSTATIC",
    "HYPOTENSION ORTHOSTATIC ASYMPTOMATIC",
    "HYPOTENSION ORTHOSTATIC SYMPTOMATIC",
    "HYPOTENSION POSTURAL",
    "HYPOTENSION POSTURAL AGGRAVATED",
    "ORTHOSTATIC HYPOTENSION",
    "POSTURAL HYPOTENSION",
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

    df["is_fall_narrow_row"] = df["pt"].isin(NARROW_FALL_TERMS)
    df["is_fall_broad_row"] = df["pt"].isin(BROAD_FALL_TERMS)

    case_level_df = df.groupby("caseid", as_index=False).agg(
        is_fall_narrow=("is_fall_narrow_row", "max"),
        fall_narrow_pt_count=("is_fall_narrow_row", "sum"),
        all_reac_n=("pt", "size"),
        is_fall_broad=("is_fall_broad_row", "max"),
    )

    fall_pt_list_df = (
        df.loc[df["is_fall_broad_row"], ["caseid", "pt"]]
        .drop_duplicates()
        .groupby("caseid")["pt"]
        .apply(lambda s: "|".join(sorted(s)))
        .reset_index(name="fall_pt_list")
    )

    case_level_df = case_level_df.merge(fall_pt_list_df, on="caseid", how="left")
    case_level_df["fall_pt_list"] = (
        case_level_df["fall_pt_list"].where(case_level_df["fall_pt_list"].notna(), "")
    )

    case_level_df["is_fall_narrow"] = case_level_df["is_fall_narrow"].astype(bool)
    case_level_df["fall_narrow_pt_count"] = case_level_df[
        "fall_narrow_pt_count"
    ].astype(int)
    case_level_df["all_reac_n"] = case_level_df["all_reac_n"].astype(int)
    case_level_df["is_fall_broad"] = case_level_df["is_fall_broad"].astype(bool)

    print("Case-level REAC rows:", len(case_level_df))
    print(
        "Fall cases (narrow PT definition):",
        int(case_level_df["is_fall_narrow"].sum()),
    )
    print(
        "Fall-related broad cases:",
        int(case_level_df["is_fall_broad"].sum()),
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"reac_{year}{quarter.lower()}_case.parquet"
    case_level_df.to_parquet(output_file, index=False)

    print(f"Saved: {output_file}")
