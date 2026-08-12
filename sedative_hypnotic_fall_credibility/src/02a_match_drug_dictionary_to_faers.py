from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DICT = PROJECT_DIR / "configs" / "drug_name_dictionary.csv"
DEFAULT_MASTER = PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv"
DEFAULT_INPUT_GLOB = (
    r"D:\program_FAERS\archive_old_outputs\results_before_20260525"
    r"\OUTPUT_COUNTRY\*\drug_*.parquet"
)
DEFAULT_MATCHES = PROJECT_DIR / "outputs" / "intermediate" / "02a_drug_dictionary_matches.parquet"
DEFAULT_COUNTS = PROJECT_DIR / "outputs" / "qc" / "02a_drug_dictionary_match_counts.csv"
DEFAULT_TERMS = PROJECT_DIR / "outputs" / "qc" / "02a_drug_dictionary_term_counts.csv"

MATCH_COLUMNS = ["drugname", "prod_ai"]
ROLE_PS_SS = {"PS", "SS"}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def term_pattern(term: str) -> str:
    normalized = normalize_text(term)
    escaped_parts = [re.escape(part) for part in normalized.split()]
    return r"\b" + r"\s+".join(escaped_parts) + r"\b"


def year_from_path(path: Path) -> str:
    match = re.search(r"drug_(\d{4})(?:q[1-4])?\.parquet$", path.name, flags=re.I)
    return match.group(1) if match else ""


def load_dictionary(path: Path) -> pd.DataFrame:
    terms = pd.read_csv(path)
    required = {"drug_key", "drug_group", "term", "term_type", "match_action"}
    missing = required - set(terms.columns)
    if missing:
        raise ValueError(f"Dictionary missing columns: {sorted(missing)}")
    terms["normalized_term"] = terms["term"].map(normalize_text)
    return terms


def build_patterns(terms: pd.DataFrame, action: str) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    selected = terms[terms["match_action"].str.lower().eq(action)].copy()
    for drug_key, group in selected.groupby("drug_key"):
        pieces = [term_pattern(term) for term in group["term"].dropna().unique()]
        if pieces:
            patterns[drug_key] = re.compile("|".join(pieces), flags=re.I)
    return patterns


def matching_terms_series(text: pd.Series, terms: list[str]) -> pd.Series:
    result = pd.Series("", index=text.index, dtype="object")
    remaining = pd.Series(True, index=text.index)
    for term in sorted(terms, key=len, reverse=True):
        current = remaining & text.str.contains(term_pattern(term), regex=True, na=False)
        if current.any():
            result.loc[current] = term
            remaining.loc[current] = False
        if not remaining.any():
            break
    return result


def find_input_files(pattern: str) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(pattern))
    return [p for p in files if re.fullmatch(r"drug_\d{4}\.parquet", p.name, flags=re.I)]


def match_file(
    path: Path,
    include_patterns: dict[str, re.Pattern[str]],
    exclude_patterns: dict[str, re.Pattern[str]],
    include_terms: dict[str, list[str]],
    master: pd.DataFrame,
) -> pd.DataFrame:
    use_columns = [
        "primaryid",
        "caseid",
        "role_cod",
        "drugname",
        "prod_ai",
        "route",
        "dose_form",
    ]
    available_columns = set(pq.read_schema(path).names)
    read_columns = [column for column in use_columns if column in available_columns]
    df = pd.read_parquet(path, columns=read_columns)
    for column in use_columns:
        if column not in df.columns:
            df[column] = pd.NA

    normalized = normalize_series(df["drugname"]) + " " + normalize_series(df["prod_ai"])
    route_form = normalize_series(df["route"]) + " " + normalize_series(df["dose_form"])
    topical_mask = route_form.str.contains(r"\b(?:topical|cream|ointment)\b", regex=True, na=False)

    chunks = []
    for drug_key, pattern in include_patterns.items():
        mask = normalized.str.contains(pattern, na=False)
        if drug_key in exclude_patterns:
            mask &= ~normalized.str.contains(exclude_patterns[drug_key], na=False)
        if drug_key in {"doxepin", "mirtazapine"}:
            mask &= ~topical_mask
        if not mask.any():
            continue

        matched = df.loc[mask].copy()
        matched["year"] = year_from_path(path)
        matched["matched_drug_key"] = drug_key
        matched["matched_drug_group"] = master.loc[drug_key, "drug_group"]
        matched["matched_term"] = matching_terms_series(normalized.loc[mask], include_terms[drug_key]).values
        matched["source_file"] = str(path)
        chunks.append(matched)

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def summarize_matches(matches: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    term_rows = []
    if matches.empty:
        for drug_key, row in master.iterrows():
            all_rows.append(
                {
                    "drug_key": drug_key,
                    "drug_group": row["drug_group"],
                    "n_drug_rows_any_role": 0,
                    "n_cases_any_role": 0,
                    "n_drug_rows_ps_ss": 0,
                    "n_cases_ps_ss": 0,
                    "n_cases_ps_only": 0,
                    "first_year": "",
                    "last_year": "",
                    "n_years_observed": 0,
                }
            )
        return pd.DataFrame(all_rows), pd.DataFrame(term_rows)

    matches["role_cod_clean"] = matches["role_cod"].fillna("").astype(str).str.upper()
    for drug_key, row in master.iterrows():
        subset = matches[matches["matched_drug_key"].eq(drug_key)]
        ps_ss = subset[subset["role_cod_clean"].isin(ROLE_PS_SS)]
        ps_only = subset[subset["role_cod_clean"].eq("PS")]
        years = sorted(y for y in subset["year"].dropna().astype(str).unique() if y)
        all_rows.append(
            {
                "drug_key": drug_key,
                "drug_group": row["drug_group"],
                "n_drug_rows_any_role": int(len(subset)),
                "n_cases_any_role": int(subset["caseid"].nunique()),
                "n_drug_rows_ps_ss": int(len(ps_ss)),
                "n_cases_ps_ss": int(ps_ss["caseid"].nunique()),
                "n_cases_ps_only": int(ps_only["caseid"].nunique()),
                "first_year": years[0] if years else "",
                "last_year": years[-1] if years else "",
                "n_years_observed": int(len(years)),
            }
        )

    term_summary = (
        matches.groupby(["matched_drug_key", "matched_drug_group", "matched_term"], dropna=False)
        .agg(n_drug_rows=("caseid", "size"), n_cases=("caseid", "nunique"))
        .reset_index()
        .sort_values(["matched_drug_key", "n_cases"], ascending=[True, False])
    )
    return pd.DataFrame(all_rows), term_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--matches-out", type=Path, default=DEFAULT_MATCHES)
    parser.add_argument("--counts-out", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--terms-out", type=Path, default=DEFAULT_TERMS)
    args = parser.parse_args()

    terms = load_dictionary(args.dictionary)
    master = pd.read_csv(args.master).set_index("drug_key", drop=False)
    include_patterns = build_patterns(terms, "include")
    exclude_patterns = build_patterns(terms, "exclude")
    include_terms = {
        key: group["term"].dropna().astype(str).tolist()
        for key, group in terms[terms["match_action"].str.lower().eq("include")].groupby("drug_key")
    }
    files = find_input_files(args.input_glob)
    if not files:
        raise FileNotFoundError(f"No annual drug parquet files matched: {args.input_glob}")

    chunks = []
    for path in files:
        print(f"Matching {path}")
        matched = match_file(path, include_patterns, exclude_patterns, include_terms, master)
        if not matched.empty:
            chunks.append(matched)

    matches = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    args.matches_out.parent.mkdir(parents=True, exist_ok=True)
    args.counts_out.parent.mkdir(parents=True, exist_ok=True)
    args.terms_out.parent.mkdir(parents=True, exist_ok=True)

    if matches.empty:
        matches.to_csv(args.matches_out.with_suffix(".csv"), index=False)
    else:
        matches.to_parquet(args.matches_out, index=False)

    counts, term_counts = summarize_matches(matches, master)
    counts.to_csv(args.counts_out, index=False, encoding="utf-8-sig")
    term_counts.to_csv(args.terms_out, index=False, encoding="utf-8-sig")
    print(f"Wrote {args.counts_out}")
    print(f"Wrote {args.terms_out}")
    print(f"Wrote {args.matches_out if not matches.empty else args.matches_out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
