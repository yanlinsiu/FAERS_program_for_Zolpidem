import re
from pathlib import Path

import pandas as pd

from config import RAW_ROOT
from utils import build_file_path, load_retained_demo_primaryids, read_faers_txt


OUTC_CODE_TO_FLAG = {
    "DE": "is_death",
    "LT": "is_life_threatening",
    "HO": "is_hospitalization",
    "DS": "is_disability",
    "CA": "is_congenital_anomaly",
    "OT": "is_other_serious",
}


def _extract_outc_tokens(value) -> list[str]:
    text = "" if pd.isna(value) else str(value).upper().strip()
    if not text:
        return []
    # Supports values like "DE,HO", "DE/HO", "DE HO", "DE;HO", "DE|HO".
    tokens = re.findall(r"[A-Z]{2}", text)
    return tokens


def process_outc(year, quarter, output_root):
    file_path = build_file_path(RAW_ROOT, year, quarter, "OUTC")
    print(f"processing file: {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    df = read_faers_txt(file_path)
    required_cols = ["primaryid", "caseid", "outc_cod"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"OUTC missing required columns: {missing_cols}")

    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()

    retained_primaryids = load_retained_demo_primaryids(
        RAW_ROOT, year, quarter, output_root=output_root
    )
    df = df[df["primaryid"].isin(retained_primaryids)]
    df = df[df["caseid"] != ""].copy()

    df["outc_tokens"] = df["outc_cod"].apply(_extract_outc_tokens)
    df = df[df["outc_tokens"].map(len) > 0].copy()
    df = df.explode("outc_tokens", ignore_index=True)
    df["outc_code"] = df["outc_tokens"].astype(str).str.strip().str.upper()
    df = df[df["outc_code"] != ""].copy()

    known_codes = set(OUTC_CODE_TO_FLAG.keys())
    unknown_codes = df.loc[~df["outc_code"].isin(known_codes), "outc_code"]
    if not unknown_codes.empty:
        unknown_counts = unknown_codes.value_counts()
        print("unknown outc_cod found (top 20):", unknown_counts.head(20).to_dict())
        print("unknown outc_cod total rows:", int(unknown_counts.sum()))

    df = df[df["outc_code"].isin(known_codes)].copy()
    if df.empty:
        case_level_df = pd.DataFrame(columns=["caseid", *OUTC_CODE_TO_FLAG.values()])
    else:
        # Remove duplicate code hits within the same case before pivoting to flags.
        df = df.drop_duplicates(subset=["caseid", "outc_code"])
        for outc_code, flag_col in OUTC_CODE_TO_FLAG.items():
            df[flag_col] = df["outc_code"].eq(outc_code)

        flag_cols = list(OUTC_CODE_TO_FLAG.values())
        case_level_df = (
            df[["caseid", *flag_cols]].groupby("caseid", as_index=False)[flag_cols].max()
        )

    flag_cols = list(OUTC_CODE_TO_FLAG.values())
    for col in flag_cols:
        if col not in case_level_df.columns:
            case_level_df[col] = False

    case_level_df["is_serious_any"] = case_level_df[flag_cols].any(axis=1)
    for col in [*flag_cols, "is_serious_any"]:
        case_level_df[col] = case_level_df[col].fillna(False).astype(bool)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"outcome_dataset_{year}{str(quarter).lower()}.parquet"
    case_level_df.to_parquet(output_file, index=False)

    # Keep legacy filename for downstream continuity.
    legacy_output_file = output_root / f"outc_{year}{str(quarter).lower()}_case.parquet"
    case_level_df.to_parquet(legacy_output_file, index=False)

    print("outcome_dataset rows:", len(case_level_df))
    print("serious cases:", int(case_level_df["is_serious_any"].sum()))
    print(f"saved: {output_file}")
    print(f"saved (legacy): {legacy_output_file}")
    return case_level_df
