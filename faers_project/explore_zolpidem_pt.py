import argparse
from pathlib import Path

import pandas as pd

from config import DEFAULT_OUTPUT_ROOT, RAW_ROOT
from drug_exposure_processor import process_drug_exposure
from utils import (
    attach_caseid_from_demo,
    build_file_path,
    ensure_required_columns,
    load_retained_demo_primaryids,
    read_faers_txt,
)


EXPOSURE_SCOPE_TO_COLUMN = {
    "suspect": "is_zolpidem_suspect",
    "ps": "is_zolpidem_suspect_ps",
}


def _load_zolpidem_caseids(year, quarter, output_root: Path, scope: str) -> set[str]:
    exposure_file = output_root / f"drug_exposure_{year}{str(quarter).lower()}_case.parquet"
    if not exposure_file.exists():
        print(f"Drug exposure output not found, generating: {exposure_file}")
        process_drug_exposure(year, quarter, output_root)

    if not exposure_file.exists():
        raise FileNotFoundError(f"drug exposure output not found: {exposure_file}")

    exposure_df = pd.read_parquet(exposure_file)
    ensure_required_columns(exposure_df, ["caseid"], "drug_exposure output")

    exposure_col = EXPOSURE_SCOPE_TO_COLUMN[scope]
    ensure_required_columns(exposure_df, [exposure_col], "drug_exposure output")

    exposure_df["caseid"] = (
        exposure_df["caseid"].where(exposure_df["caseid"].notna(), "").astype(str).str.strip()
    )
    exposure_df[exposure_col] = exposure_df[exposure_col].fillna(False).astype(bool)

    caseids = set(exposure_df.loc[exposure_df[exposure_col], "caseid"])
    caseids.discard("")
    return caseids


def _load_reac_rows(year, quarter, output_root: Path) -> pd.DataFrame:
    file_path = build_file_path(RAW_ROOT, year, quarter, "REAC")
    print(f"Reading REAC file: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"REAC file not found: {file_path}")

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
    df = df[(df["caseid"] != "") & (df["pt"] != "")]
    print("REAC rows after DEMO filter:", len(df))
    return df


def explore_zolpidem_pt(year, quarter, output_root, scope="suspect", top_n=50):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    zolpidem_caseids = _load_zolpidem_caseids(year, quarter, output_root, scope)
    if not zolpidem_caseids:
        raise ValueError(f"no zolpidem cases found under scope={scope}")

    reac_df = _load_reac_rows(year, quarter, output_root)
    zolpidem_reac_df = reac_df[reac_df["caseid"].isin(zolpidem_caseids)].copy()

    if zolpidem_reac_df.empty:
        raise ValueError("no REAC rows found after filtering to zolpidem cases")

    pt_summary = (
        zolpidem_reac_df.groupby("pt", as_index=False)
        .agg(
            reaction_row_count=("pt", "size"),
            case_count=("caseid", "nunique"),
        )
        .sort_values(
            by=["case_count", "reaction_row_count", "pt"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    pt_summary["case_pct"] = (
        pt_summary["case_count"] / len(zolpidem_caseids) * 100
    ).round(2)
    pt_summary["row_pct"] = (
        pt_summary["reaction_row_count"] / len(zolpidem_reac_df) * 100
    ).round(2)
    pt_summary.insert(0, "rank", range(1, len(pt_summary) + 1))

    output_file = output_root / f"zolpidem_pt_frequency_{year}{str(quarter).lower()}_{scope}.csv"
    pt_summary.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("Zolpidem case scope:", scope)
    print("Zolpidem case count:", len(zolpidem_caseids))
    print("Zolpidem REAC row count:", len(zolpidem_reac_df))
    print("Distinct PT count:", len(pt_summary))
    print(f"Saved: {output_file}")
    print()
    print(f"Top {min(top_n, len(pt_summary))} PT by case_count:")
    print(pt_summary.head(top_n).to_string(index=False))

    return pt_summary


def main():
    parser = argparse.ArgumentParser(
        description="Explore PT frequencies among zolpidem cases in FAERS."
    )
    parser.add_argument("--year", required=True, type=int, help="Year, e.g. 2024")
    parser.add_argument(
        "--quarter",
        required=True,
        type=str,
        choices=["Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"],
        help="Quarter, e.g. Q1",
    )
    parser.add_argument(
        "--scope",
        default="suspect",
        choices=["suspect", "ps"],
        help="suspect=PS+SS, ps=PS-only",
    )
    parser.add_argument(
        "--top-n",
        default=50,
        type=int,
        help="How many top PT rows to print in the console preview",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=str,
        help="Output directory (optional, defaults to config value)",
    )
    args = parser.parse_args()

    explore_zolpidem_pt(
        year=args.year,
        quarter=args.quarter.upper(),
        output_root=args.output,
        scope=args.scope,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
