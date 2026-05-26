from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT


DEFAULT_DICTIONARY_PATH = (
    PROJECT_ROOT / "references" / "drug_dictionary" / "zdrug_dictionary.csv"
)

STANDARD_TO_FEATURE = {
    "zolpidem": "is_zolpidem",
    "zaleplon": "is_zaleplon",
    "zopiclone": "is_zopiclone",
    "eszopiclone": "is_eszopiclone",
}


def normalize_dictionary_term(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().upper()
    text = re.sub(r"[\(\)\[\]\{\},.;:]+", " ", text)
    text = re.sub(r"[/_-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_zdrug_dictionary(path: str | Path | None = None) -> pd.DataFrame:
    dictionary_path = Path(path) if path is not None else DEFAULT_DICTIONARY_PATH
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Drug dictionary not found: {dictionary_path}")

    dictionary = pd.read_csv(dictionary_path, dtype=str).fillna("")
    required_cols = {"raw_name", "clean_name", "standard_name", "include"}
    missing_cols = required_cols - set(dictionary.columns)
    if missing_cols:
        raise ValueError(f"Drug dictionary missing columns: {sorted(missing_cols)}")

    dictionary["include_bool"] = dictionary["include"].str.lower().eq("true")
    dictionary["match_term"] = dictionary["clean_name"].where(
        dictionary["clean_name"].str.strip() != "",
        dictionary["raw_name"],
    )
    dictionary["match_term"] = dictionary["match_term"].map(normalize_dictionary_term)
    dictionary["standard_name"] = dictionary["standard_name"].str.strip().str.lower()
    return dictionary


def build_zdrug_feature_terms(path: str | Path | None = None) -> dict[str, list[str]]:
    dictionary = load_zdrug_dictionary(path)
    dictionary = dictionary[
        dictionary["include_bool"]
        & dictionary["standard_name"].isin(STANDARD_TO_FEATURE)
        & (dictionary["match_term"] != "")
    ]

    feature_terms: dict[str, set[str]] = {
        feature_name: set() for feature_name in STANDARD_TO_FEATURE.values()
    }
    for row in dictionary.itertuples(index=False):
        feature_terms[STANDARD_TO_FEATURE[row.standard_name]].add(row.match_term)

    return {
        feature_name: sorted(terms, key=len, reverse=True)
        for feature_name, terms in feature_terms.items()
    }


def build_zdrug_exposure_terms(path: str | Path | None = None) -> tuple[list[str], list[str]]:
    feature_terms = build_zdrug_feature_terms(path)
    zolpidem_terms = feature_terms["is_zolpidem"]
    other_zdrug_terms = sorted(
        set(feature_terms["is_zaleplon"])
        | set(feature_terms["is_zopiclone"])
        | set(feature_terms["is_eszopiclone"]),
        key=len,
        reverse=True,
    )
    return zolpidem_terms, other_zdrug_terms
