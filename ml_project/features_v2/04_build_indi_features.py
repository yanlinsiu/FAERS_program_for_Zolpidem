from __future__ import annotations

import argparse
import re

import pandas as pd

from feature_v2_common import (
    LOOKUP_DIR,
    QC_DIR,
    QUARTERLY_DIR,
    clean_caseid,
    ensure_feature_v2_dirs,
    iter_quarters,
    load_clean_case_table,
    missingness_table,
    normalize_meddra_term,
    quarter_token,
    summarize_frame,
    write_csv,
    write_parquet,
)


IMPORTANT_GROUPS = {
    "indi_insomnia": ["INSOMNIA", "SLEEP"],
    "indi_anxiety": ["ANXI", "PHOBIA", "STRESS DISORDER"],
    "indi_depression": ["DEPRESS"],
    "indi_pain": ["PAIN", "NEURALGIA", "ARTHRALGIA", "MYALGIA"],
    "indi_epilepsy": ["EPILEP", "SEIZURE", "CONVULSION"],
    "indi_dizziness_vertigo": ["DIZZINESS", "VERTIGO", "SYNCOPE", "PRESYNCOPE"],
}


def _sanitize_soc(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _load_lookup() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = LOOKUP_DIR / "meddra_lookup.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"MedDRA lookup not found: {path}. Run 01_meddra_lookup.py first."
        )
    lookup = pd.read_parquet(path)
    llt_lookup = lookup.drop_duplicates(subset=["llt_term_norm"]).copy()
    pt_lookup = lookup.drop_duplicates(subset=["pt_term_norm"]).copy()
    return llt_lookup, pt_lookup


def _add_group_flags(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = ["pt_english", "hlt_english", "hlgt_english", "soc_english"]
    combined = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.upper()
    )
    for group_name, terms in IMPORTANT_GROUPS.items():
        pattern = "|".join(re.escape(term) for term in terms)
        df[group_name] = combined.str.contains(pattern, regex=True, na=False)
    return df


def build_indi_features(year: int, quarter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    llt_lookup, pt_lookup = _load_lookup()
    df = load_clean_case_table(year, quarter, "INDI")
    if "indi_pt" not in df.columns:
        raise ValueError("INDI missing required column: indi_pt")

    df["caseid"] = clean_caseid(df["caseid"])
    df["indi_pt_norm"] = normalize_meddra_term(df["indi_pt"])
    df = df[(df["caseid"] != "") & (df["indi_pt_norm"] != "")].copy()

    row_df = df[["caseid", "indi_pt_norm"]].drop_duplicates().copy()
    llt_match = row_df.merge(
        llt_lookup,
        left_on="indi_pt_norm",
        right_on="llt_term_norm",
        how="left",
        suffixes=("", "_meddra"),
    )
    unmatched = llt_match["pt_code"].isna()
    pt_match = row_df.loc[unmatched.to_numpy(), ["caseid", "indi_pt_norm"]].merge(
        pt_lookup,
        left_on="indi_pt_norm",
        right_on="pt_term_norm",
        how="left",
    )
    matched = pd.concat([llt_match.loc[~unmatched], pt_match], ignore_index=True, sort=False)
    matched["is_meddra_mapped"] = matched["pt_code"].notna()
    matched = _add_group_flags(matched)

    soc_flags = []
    for soc in matched.loc[matched["soc_english"].notna(), "soc_english"].drop_duplicates():
        col = f"indi_soc_{_sanitize_soc(soc)}"
        if col == "indi_soc_":
            continue
        matched[col] = matched["soc_english"].eq(soc)
        soc_flags.append(col)

    grouped = matched.groupby("caseid", as_index=False).agg(
        indi_n=("indi_pt_norm", "size"),
        distinct_indi_n=("indi_pt_norm", "nunique"),
        indi_mapped_n=("is_meddra_mapped", "sum"),
    )
    grouped["indi_unmapped_n"] = grouped["indi_n"] - grouped["indi_mapped_n"]

    bool_cols = list(IMPORTANT_GROUPS.keys()) + sorted(set(soc_flags))
    if bool_cols:
        bool_grouped = matched.groupby("caseid", as_index=False)[bool_cols].max()
        grouped = grouped.merge(bool_grouped, on="caseid", how="left")
    for column in bool_cols:
        grouped[column] = grouped[column].fillna(False).astype(bool)

    unmapped = (
        matched.loc[~matched["is_meddra_mapped"], ["indi_pt_norm"]]
        .value_counts()
        .reset_index(name="n_rows")
        .head(200)
    )
    return grouped, unmapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build INDI-MedDRA ML-v2 quarterly features.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    ensure_feature_v2_dirs()
    qc_frames = []
    unmapped_frames = []
    for year, quarter in iter_quarters(args.start_year, args.end_year):
        token = quarter_token(year, quarter)
        print(f"[indi-v2] {token}", flush=True)
        features, unmapped = build_indi_features(year, quarter)
        write_parquet(features, QUARTERLY_DIR / f"indi_v2_{token}.parquet")
        if not unmapped.empty:
            unmapped.insert(0, "period", token)
            unmapped_frames.append(unmapped)
        qc_frames.append(summarize_frame(features, f"indi_v2_{token}"))
        qc_frames.append(missingness_table(features, f"indi_v2_{token}"))

    if qc_frames:
        write_csv(pd.concat(qc_frames, ignore_index=True), QC_DIR / "indi_v2_qc.csv")
    if unmapped_frames:
        write_csv(
            pd.concat(unmapped_frames, ignore_index=True),
            QC_DIR / "indi_v2_unmapped_top_terms.csv",
        )


if __name__ == "__main__":
    main()
