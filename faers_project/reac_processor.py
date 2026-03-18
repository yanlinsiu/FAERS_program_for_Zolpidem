import pandas as pd
from pathlib import Path

from utils import build_file_path, load_retained_demo_primaryids, read_faers_txt
from config import RAW_ROOT


def process_reac(year, quarter, output_root):
    """Process FAERS REAC and build case-level outcomes."""
    file_path = build_file_path(RAW_ROOT, year, quarter, "REAC")
    print(f"Processing file: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = read_faers_txt(file_path)

    required_cols = ["primaryid", "caseid"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"REAC missing required columns: {missing_cols}")

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

    fall_terms = {"FALL", "FALLS"}
    broad_fall_pattern = (
        r"(?<![A-Z0-9])FALLS?(?![A-Z0-9])|"
        r"(?<![A-Z0-9])FALLEN(?![A-Z0-9])|"
        r"(?<![A-Z0-9])FALLING(?![A-Z0-9])"
    )

    df["is_fall_exact_row"] = df["pt"].isin(fall_terms)
    df["is_fall_related_broad_row"] = df["pt"].str.contains(
        broad_fall_pattern, regex=True, na=False
    )

    case_level_df = df.groupby("caseid", as_index=False).agg(
        is_fall=("is_fall_exact_row", "max"),
        fall_pt_count=("is_fall_exact_row", "sum"),
        all_reac_n=("pt", "size"),
        has_fall_related_broad=("is_fall_related_broad_row", "max"),
    )

    fall_pt_list_df = (
        df.loc[df["is_fall_related_broad_row"], ["caseid", "pt"]]
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
    case_level_df["fall_pt_count"] = case_level_df["fall_pt_count"].astype(int)
    case_level_df["all_reac_n"] = case_level_df["all_reac_n"].astype(int)
    case_level_df["has_fall_related_broad"] = case_level_df[
        "has_fall_related_broad"
    ].astype(bool)

    print("Case-level REAC rows:", len(case_level_df))
    print("Fall cases (strict PT=FALL/FALLS):", int(case_level_df["is_fall"].sum()))
    print(
        "Fall-related broad cases:",
        int(case_level_df["has_fall_related_broad"].sum()),
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"reac_{year}{quarter.lower()}_case.parquet"
    case_level_df.to_parquet(output_file, index=False)

    print(f"Saved: {output_file}")
