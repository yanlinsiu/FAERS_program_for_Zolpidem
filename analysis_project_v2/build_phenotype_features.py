from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
PHENOTYPE_OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "phenotypes"


@dataclass(frozen=True)
class PhenotypeSpec:
    column: str
    layer: str
    label: str
    pt_terms: tuple[str, ...]


PHENOTYPE_SPECS: tuple[PhenotypeSpec, ...] = (
    PhenotypeSpec(
        column="pheno_sedation_somnolence",
        layer="prodrome",
        label="Sedation / somnolence",
        pt_terms=("SOMNOLENCE", "SEDATION", "HYPERSOMNIA", "LETHARGY"),
    ),
    PhenotypeSpec(
        column="pheno_consciousness_cognition",
        layer="prodrome",
        label="Consciousness / cognition change",
        pt_terms=(
            "ALTERED STATE OF CONSCIOUSNESS",
            "DEPRESSED LEVEL OF CONSCIOUSNESS",
            "LOSS OF CONSCIOUSNESS",
            "CONFUSIONAL STATE",
            "DISORIENTATION",
            "DELIRIUM",
            "COGNITIVE DISORDER",
            "DISTURBANCE IN ATTENTION",
            "MEMORY IMPAIRMENT",
            "MENTAL IMPAIRMENT",
            "MENTAL STATUS CHANGES",
        ),
    ),
    PhenotypeSpec(
        column="pheno_dizziness_vertigo_syncope",
        layer="prodrome",
        label="Dizziness / vertigo / syncope",
        pt_terms=(
            "DIZZINESS",
            "VERTIGO",
            "VERTIGO POSITIONAL",
            "VERTIGO CNS ORIGIN",
            "VESTIBULAR DISORDER",
            "SYNCOPE",
            "PRESYNCOPE",
        ),
    ),
    PhenotypeSpec(
        column="pheno_gait_balance_motor",
        layer="prodrome",
        label="Gait / balance / motor control abnormality",
        pt_terms=(
            "GAIT DISTURBANCE",
            "GAIT INABILITY",
            "BALANCE DISORDER",
            "ATAXIA",
            "COORDINATION ABNORMAL",
            "MOBILITY DECREASED",
            "MOVEMENT DISORDER",
        ),
    ),
    PhenotypeSpec(
        column="pheno_hypotension",
        layer="prodrome",
        label="Hypotension / orthostatic hypotension",
        pt_terms=("HYPOTENSION", "ORTHOSTATIC HYPOTENSION", "BLOOD PRESSURE DECREASED"),
    ),
    PhenotypeSpec(
        column="pheno_visual_disturbance",
        layer="secondary_prodrome",
        label="Visual disturbance",
        pt_terms=("VISUAL IMPAIRMENT", "VISUAL ACUITY REDUCED", "VISION BLURRED"),
    ),
    PhenotypeSpec(
        column="pheno_fall_event",
        layer="fall_event",
        label="Fall event",
        pt_terms=("FALL",),
    ),
    PhenotypeSpec(
        column="pheno_fracture_injury",
        layer="consequence",
        label="Fracture / injury consequence",
        pt_terms=(
            "FRACTURE",
            "HIP FRACTURE",
            "FEMUR FRACTURE",
            "INJURY",
            "HEAD INJURY",
            "CRANIOCEREBRAL INJURY",
            "CONTUSION",
            "WOUND",
            "SKIN LACERATION",
        ),
    ),
    PhenotypeSpec(
        column="pheno_hospitalisation_pt",
        layer="consequence",
        label="Hospitalisation PT",
        pt_terms=("HOSPITALISATION",),
    ),
)


def _norm_term(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _find_meddra_file() -> Path:
    candidates = sorted(PROJECT_ROOT.glob("MedDRA*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No MedDRA*.xlsx file found in project root.")
    return candidates[0]


def build_meddra_llt_to_pt_map(meddra_file: Path) -> dict[str, str]:
    meddra = pd.read_excel(meddra_file, sheet_name=0)
    required = {"llt_english", "pt_english"}
    missing = required - set(meddra.columns)
    if missing:
        raise ValueError(f"MedDRA sheet missing columns: {sorted(missing)}")

    llt_to_pt = {}
    for row in meddra[["llt_english", "pt_english"]].dropna(subset=["pt_english"]).itertuples(index=False):
        llt = _norm_term(row.llt_english)
        pt = _norm_term(row.pt_english)
        if llt and pt:
            llt_to_pt[llt] = pt
            llt_to_pt[pt] = pt
    return llt_to_pt


def build_dictionary_frame(llt_to_pt: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    term_to_category: dict[str, list[str]] = {}
    for spec in PHENOTYPE_SPECS:
        for pt in spec.pt_terms:
            term_to_category.setdefault(pt, []).append(spec.column)
            rows.append(
                {
                    "phenotype_column": spec.column,
                    "layer": spec.layer,
                    "label": spec.label,
                    "pt_term": pt,
                    "pt_term_in_meddra_map": pt in llt_to_pt,
                    "mapped_pt": llt_to_pt.get(pt, pt),
                }
            )
    dictionary = pd.DataFrame(rows)
    duplicate_terms = {
        term: columns for term, columns in term_to_category.items() if len(set(columns)) > 1
    }
    if duplicate_terms:
        raise ValueError(f"PT terms assigned to multiple phenotype categories: {duplicate_terms}")
    return dictionary


def load_case_index(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    case_index = pd.read_parquet(path)
    required = {"caseid", "primaryid", "year", "quarter"}
    missing = required - set(case_index.columns)
    if missing:
        raise ValueError(f"case index missing columns: {sorted(missing)}")
    out = case_index[["caseid", "primaryid", "year", "quarter"]].copy()
    out["caseid"] = out["caseid"].where(out["caseid"].notna(), "").astype(str).str.strip()
    out["primaryid"] = pd.to_numeric(out["primaryid"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["quarter"] = out["quarter"].where(out["quarter"].notna(), "").astype(str).str.upper().str.strip()
    out = out[out["year"].between(start_year, end_year, inclusive="both")].copy()
    out = out[(out["caseid"] != "") & out["primaryid"].notna()].copy()
    return out


def _quarter_sort_key(value: str) -> int:
    value = str(value).upper().strip()
    if value.startswith("Q") and value[1:].isdigit():
        return int(value[1:])
    return 0


def process_quarter(
    case_index_quarter: pd.DataFrame,
    year: int,
    quarter: str,
    llt_to_pt: dict[str, str],
    pt_to_column: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    file_path = OUTPUT_ROOT / str(year) / "quarterly" / f"reac_event_{year}{quarter.lower()}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned REAC event file not found: {file_path}. "
            "Run faers_project/year_batch_runner.py first."
        )

    reac = pd.read_parquet(file_path, columns=["caseid", "primaryid", "pt"])
    reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
    reac = reac[reac["primaryid"].notna()].copy()
    reac["reported_term"] = reac["pt"].map(_norm_term)
    reac = reac[reac["reported_term"] != ""].copy()

    reac["caseid"] = reac["caseid"].where(reac["caseid"].notna(), "").astype(str).str.strip()
    merged = reac.merge(case_index_quarter[["caseid", "primaryid"]], on=["caseid", "primaryid"], how="inner")
    if merged.empty:
        empty = case_index_quarter[["caseid"]].copy()
        for spec in PHENOTYPE_SPECS:
            empty[spec.column] = False
        empty["phenotype_pt_list"] = ""
        return empty, {
            "year": year,
            "quarter": quarter,
            "reac_rows": int(len(reac)),
            "matched_reac_rows": 0,
            "matched_cases_with_reac": 0,
            "matched_phenotype_rows": 0,
            "unmapped_term_rows": int((~reac["reported_term"].isin(llt_to_pt)).sum()),
        }

    merged["pt_term"] = merged["reported_term"].map(llt_to_pt).fillna(merged["reported_term"])
    merged["phenotype_column"] = merged["pt_term"].map(pt_to_column)
    phenotype_rows = merged[merged["phenotype_column"].notna()].copy()

    base = case_index_quarter[["caseid"]].drop_duplicates().copy()
    for spec in PHENOTYPE_SPECS:
        base[spec.column] = False

    if not phenotype_rows.empty:
        flags = (
            phenotype_rows[["caseid", "phenotype_column"]]
            .drop_duplicates()
            .assign(value=True)
            .pivot(index="caseid", columns="phenotype_column", values="value")
            .fillna(False)
            .reset_index()
        )
        base = base.merge(flags, on="caseid", how="left", suffixes=("", "_hit"))
        for spec in PHENOTYPE_SPECS:
            hit_col = f"{spec.column}_hit"
            if hit_col in base.columns:
                base[spec.column] = base[hit_col].fillna(False).astype(bool)
                base = base.drop(columns=[hit_col])
            else:
                base[spec.column] = base[spec.column].fillna(False).astype(bool)

        pt_list = (
            phenotype_rows[["caseid", "pt_term"]]
            .drop_duplicates()
            .sort_values(["caseid", "pt_term"])
            .groupby("caseid")["pt_term"]
            .apply(lambda values: "|".join(values))
            .reset_index(name="phenotype_pt_list")
        )
        base = base.merge(pt_list, on="caseid", how="left")
    else:
        base["phenotype_pt_list"] = ""

    base["phenotype_pt_list"] = base["phenotype_pt_list"].fillna("")
    unmapped_rows = int((~merged["reported_term"].isin(llt_to_pt)).sum())
    qc = {
        "year": year,
        "quarter": quarter,
        "reac_rows": int(len(reac)),
        "matched_reac_rows": int(len(merged)),
        "matched_cases_with_reac": int(merged["caseid"].nunique()),
        "matched_phenotype_rows": int(len(phenotype_rows)),
        "unmapped_term_rows": unmapped_rows,
    }
    return base, qc


def build_phenotype_features(
    start_year: int,
    end_year: int,
    case_index_file: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    period_token = f"{start_year}_{end_year}"

    llt_to_pt = build_meddra_llt_to_pt_map(_find_meddra_file())
    dictionary = build_dictionary_frame(llt_to_pt)
    pt_to_column = dict(zip(dictionary["pt_term"], dictionary["phenotype_column"], strict=True))

    case_index = load_case_index(case_index_file, start_year, end_year)
    phenotype_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    for (year, quarter), quarter_cases in case_index.groupby(["year", "quarter"], sort=True):
        year_int = int(year)
        quarter_str = str(quarter).upper()
        print(f"Processing REAC phenotype features: {year_int} {quarter_str}")
        part, qc = process_quarter(
            case_index_quarter=quarter_cases,
            year=year_int,
            quarter=quarter_str,
            llt_to_pt=llt_to_pt,
            pt_to_column=pt_to_column,
        )
        phenotype_parts.append(part)
        qc_rows.append(qc)

    phenotype = pd.concat(phenotype_parts, ignore_index=True)
    phenotype = case_index[["caseid"]].drop_duplicates().merge(phenotype, on="caseid", how="left")
    for spec in PHENOTYPE_SPECS:
        phenotype[spec.column] = phenotype[spec.column].fillna(False).astype(bool)
    phenotype["phenotype_pt_list"] = phenotype["phenotype_pt_list"].fillna("")

    summary_rows = []
    total_cases = len(phenotype)
    for spec in PHENOTYPE_SPECS:
        n = int(phenotype[spec.column].sum())
        summary_rows.append(
            {
                "phenotype_column": spec.column,
                "layer": spec.layer,
                "label": spec.label,
                "case_count": n,
                "case_percent": round(n / total_cases * 100, 4) if total_cases else 0.0,
            }
        )

    feature_file = output_dir / f"phenotype_features_{period_token}_case.parquet"
    dictionary_file = output_dir / f"phenotype_dictionary_{period_token}.csv"
    qc_file = output_dir / f"phenotype_build_qc_{period_token}.csv"
    summary_file = output_dir / f"phenotype_summary_{period_token}.csv"

    phenotype.to_parquet(feature_file, index=False)
    dictionary.to_csv(dictionary_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(qc_rows).sort_values(
        ["year", "quarter"], key=lambda col: col.map(_quarter_sort_key) if col.name == "quarter" else col
    ).to_csv(qc_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False, encoding="utf-8-sig")

    return {
        "features": feature_file,
        "dictionary": dictionary_file,
        "qc": qc_file,
        "summary": summary_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build case-level fall phenotype features from FAERS REAC.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--case-index-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHENOTYPE_OUTPUT_DIR)
    args = parser.parse_args()

    period_token = f"{args.start_year}_{args.end_year}"
    case_index_file = args.case_index_file or GLOBAL_DATASET_DIR / f"global_case_index_{period_token}.parquet"
    outputs = build_phenotype_features(
        start_year=args.start_year,
        end_year=args.end_year,
        case_index_file=case_index_file,
        output_dir=args.output_dir,
    )
    print("phenotype feature build completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
