import re
from pathlib import Path

import pandas as pd

from drug_processor import process_drug


DRUG_FEATURE_TERMS = {
    "is_zolpidem": [
        "ZOLPIDEM",
        "AMBIEN",
        "STILNOX",
        "EDLUAR",
        "INTERMEZZO",
        "ZOLPIMIST",
    ],
    "is_zaleplon": ["ZALEPLON", "SONATA"],
    "is_zopiclone": ["ZOPICLONE", "IMOVANE", "ZIMOVANE"],
    "is_eszopiclone": ["ESZOPICLONE", "LUNESTA"],
    "is_benzo": [
        "ALPRAZOLAM",
        "DIAZEPAM",
        "LORAZEPAM",
        "CLONAZEPAM",
        "TEMAZEPAM",
    ],
    "is_antidepressant": [
        "SERTRALINE",
        "ESCITALOPRAM",
        "FLUOXETINE",
        "CITALOPRAM",
        "PAROXETINE",
        "VENLAFAXINE",
        "DULOXETINE",
        "AMITRIPTYLINE",
        "MIRTAZAPINE",
    ],
    "is_antipsychotic": [
        "QUETIAPINE",
        "OLANZAPINE",
        "RISPERIDONE",
        "ARIPIPRAZOLE",
        "HALOPERIDOL",
    ],
    "is_opioid": [
        "OXYCODONE",
        "HYDROCODONE",
        "MORPHINE",
        "FENTANYL",
        "TRAMADOL",
        "CODEINE",
    ],
    "is_antiepileptic": [
        "GABAPENTIN",
        "PREGABALIN",
        "VALPROATE",
        "CARBAMAZEPINE",
        "LAMOTRIGINE",
    ],
}


def _normalize_drug_text(series: pd.Series) -> pd.Series:
    return (
        series.where(series.notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def _build_boundary_pattern(terms) -> str:
    escaped_terms = sorted({re.escape(term) for term in terms}, key=len, reverse=True)
    alternation = "|".join(escaped_terms)
    return rf"(?<![A-Z0-9])(?:{alternation})(?![A-Z0-9])"


def process_drug_feature(year, quarter, output_root):
    output_root = Path(output_root)
    drug_file = output_root / f"drug_{year}{str(quarter).lower()}.parquet"

    if not drug_file.exists():
        print(f"missing DRUG parquet, generating: {drug_file}")
        process_drug(year, quarter, output_root)

    if not drug_file.exists():
        raise FileNotFoundError(f"DRUG file not found: {drug_file}")

    df = pd.read_parquet(drug_file)
    required_cols = ["caseid", "drugname", "prod_ai"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DRUG missing required columns: {missing_cols}")

    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    df["drugname"] = _normalize_drug_text(df["drugname"])
    df["prod_ai"] = _normalize_drug_text(df["prod_ai"])

    df = df[df["caseid"] != ""]
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))]

    # Distinct-drug count uses standardized active ingredient first, then drugname.
    df["resolved_drug_name"] = df["prod_ai"].where(df["prod_ai"] != "", df["drugname"])

    for feature_name, terms in DRUG_FEATURE_TERMS.items():
        pattern = _build_boundary_pattern(terms)
        df[feature_name] = (
            df["drugname"].str.contains(pattern, regex=True, na=False)
            | df["prod_ai"].str.contains(pattern, regex=True, na=False)
        )

    feature_cols = list(DRUG_FEATURE_TERMS.keys())
    case_feature_df = (
        df[["caseid", *feature_cols]]
        .groupby("caseid", as_index=False)[feature_cols]
        .max()
    )

    # drug_n: row count after DRUG cleaning/filtering (record count).
    raw_count_df = (
        df.groupby("caseid", as_index=False)
        .size()
        .rename(columns={"size": "drug_n"})
    )

    # distinct_drug_n: count of distinct resolved drug names.
    distinct_count_df = (
        df[df["resolved_drug_name"] != ""][["caseid", "resolved_drug_name"]]
        .drop_duplicates()
        .groupby("caseid", as_index=False)
        .size()
        .rename(columns={"size": "distinct_drug_n"})
    )

    case_feature_df = case_feature_df.merge(raw_count_df, on="caseid", how="left")
    case_feature_df = case_feature_df.merge(distinct_count_df, on="caseid", how="left")
    case_feature_df["drug_n"] = case_feature_df["drug_n"].fillna(0).astype(int)
    case_feature_df["distinct_drug_n"] = (
        case_feature_df["distinct_drug_n"].fillna(0).astype(int)
    )
    case_feature_df["polypharmacy"] = case_feature_df["distinct_drug_n"] >= 5

    output_root.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_root / f"drug_feature_dataset_{year}{str(quarter).lower()}.parquet"
    )
    case_feature_df.to_parquet(output_file, index=False)

    # Keep legacy filename for downstream continuity.
    legacy_output_file = output_root / f"drug_feature_{year}{str(quarter).lower()}_case.parquet"
    case_feature_df.to_parquet(legacy_output_file, index=False)

    print("drug_feature_dataset rows:", len(case_feature_df))
    print("polypharmacy cases:", int(case_feature_df["polypharmacy"].sum()))
    print(f"saved: {output_file}")
    print(f"saved (legacy): {legacy_output_file}")
    return case_feature_df
