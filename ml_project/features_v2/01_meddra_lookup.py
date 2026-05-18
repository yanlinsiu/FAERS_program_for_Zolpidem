from __future__ import annotations

import argparse

import pandas as pd

from feature_v2_common import (
    LOOKUP_DIR,
    QC_DIR,
    ensure_feature_v2_dirs,
    latest_meddra_excel,
    normalize_meddra_term,
    write_csv,
    write_parquet,
)


KEEP_COLUMNS = [
    "llt_code",
    "llt_chinese",
    "llt_english",
    "pt_code",
    "pt_chinese",
    "pt_english",
    "hlt_code",
    "hlt_chinese",
    "hlt_english",
    "hlgt_code",
    "hlgt_chinese",
    "hlgt_english",
    "soc_code",
    "soc_chinese",
    "soc_english",
    "system",
    "主soc",
    "llt_currency",
]


def build_lookup(excel_path) -> pd.DataFrame:
    xl = pd.ExcelFile(excel_path)
    sheet_name = xl.sheet_names[0]
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    missing = [column for column in KEEP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"MedDRA sheet missing columns: {missing}")

    lookup = df[KEEP_COLUMNS].copy()
    lookup["llt_term_norm"] = normalize_meddra_term(lookup["llt_english"])
    lookup["pt_term_norm"] = normalize_meddra_term(lookup["pt_english"])
    lookup = lookup[(lookup["llt_term_norm"] != "") | (lookup["pt_term_norm"] != "")].copy()
    lookup = lookup.drop_duplicates(subset=["llt_term_norm", "pt_term_norm", "pt_code"])
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized MedDRA lookup for ML-v2.")
    parser.add_argument("--excel-path", default=None)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    excel_path = latest_meddra_excel() if args.excel_path is None else args.excel_path
    lookup = build_lookup(excel_path)

    output_path = LOOKUP_DIR / "meddra_lookup.parquet"
    qc_path = QC_DIR / "meddra_lookup_qc.csv"
    write_parquet(lookup, output_path)
    qc_df = pd.DataFrame(
        [
            {"metric": "n_rows", "value": int(len(lookup))},
            {"metric": "n_unique_llt_norm", "value": int(lookup["llt_term_norm"].nunique())},
            {"metric": "n_unique_pt_norm", "value": int(lookup["pt_term_norm"].nunique())},
            {"metric": "n_unique_soc", "value": int(lookup["soc_english"].nunique())},
        ]
    )
    write_csv(qc_df, qc_path)
    print(f"Saved MedDRA lookup to: {output_path}")
    print(f"Saved MedDRA QC to: {qc_path}")


if __name__ == "__main__":
    main()
