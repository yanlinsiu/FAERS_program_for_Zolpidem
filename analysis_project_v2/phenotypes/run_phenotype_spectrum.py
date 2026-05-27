from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from ..signal_metrics import signal_metrics, two_by_two_counts
except ImportError:
    from signal_metrics import signal_metrics, two_by_two_counts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
PHENOTYPE_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "phenotypes"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "phenotype_analysis"


PHENOTYPE_COLUMNS = (
    "pheno_sedation_somnolence",
    "pheno_consciousness_cognition",
    "pheno_dizziness_vertigo_syncope",
    "pheno_gait_balance_motor",
    "pheno_hypotension",
    "pheno_visual_disturbance",
    "pheno_fall_event",
    "pheno_fracture_injury",
    "pheno_hospitalisation_pt",
)


def _rate(n: int, denominator: int) -> float:
    return round(n / denominator * 100, 4) if denominator else 0.0


def _load_dictionary(path: Path) -> pd.DataFrame:
    dictionary = pd.read_csv(path)
    return dictionary[["phenotype_column", "layer", "label"]].drop_duplicates()


def _load_analysis_frame(period_token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.read_parquet(GLOBAL_DATASET_DIR / f"signal_dataset_{period_token}.parquet")
    phenotype = pd.read_parquet(PHENOTYPE_DIR / f"phenotype_features_{period_token}_case.parquet")
    dictionary = _load_dictionary(PHENOTYPE_DIR / f"phenotype_dictionary_{period_token}.csv")
    df = signal.merge(phenotype, on="caseid", how="left")
    for col in PHENOTYPE_COLUMNS:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["is_fall", "is_zolpidem_suspect", "suspect_role_any"]:
        df[col] = df[col].fillna(False).astype(bool)
    return df, dictionary


def build_descriptive_spectrum(df: pd.DataFrame, dictionary: pd.DataFrame) -> pd.DataFrame:
    zolpidem = df[
        df["is_zolpidem_suspect"]
        & df["suspect_role_any"]
        & df["target_drug_group"].ne("both_zolpidem_and_other_zdrug")
    ].copy()
    zolpidem_fall = zolpidem[zolpidem["is_fall"]].copy()
    cohorts = (
        ("zolpidem_ps_ss_all", zolpidem),
        ("zolpidem_ps_ss_strict_fall", zolpidem_fall),
    )

    rows: list[dict[str, Any]] = []
    for cohort_name, subset in cohorts:
        denominator = len(subset)
        for col in PHENOTYPE_COLUMNS:
            n = int(subset[col].sum())
            rows.append(
                {
                    "cohort": cohort_name,
                    "denominator": denominator,
                    "phenotype_column": col,
                    "case_count": n,
                    "case_percent": _rate(n, denominator),
                }
            )
    return pd.DataFrame(rows).merge(dictionary, on="phenotype_column", how="left")


def build_within_zolpidem_fall_comparison(
    df: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    zolpidem = df[
        df["is_zolpidem_suspect"]
        & df["suspect_role_any"]
        & df["target_drug_group"].ne("both_zolpidem_and_other_zdrug")
    ].copy()
    rows: list[dict[str, Any]] = []
    outcome = zolpidem["is_fall"]
    for col in PHENOTYPE_COLUMNS:
        if col == "pheno_fall_event":
            continue
        counts = two_by_two_counts(zolpidem[col], outcome)
        rows.append(
            {
                "comparison": "phenotype_positive_vs_negative_among_zolpidem_ps_ss",
                "phenotype_column": col,
                **signal_metrics(**counts),
            }
        )
    return pd.DataFrame(rows).merge(dictionary, on="phenotype_column", how="left")


def build_zolpidem_vs_other_suspect_signal(
    df: pd.DataFrame,
    dictionary: pd.DataFrame,
) -> pd.DataFrame:
    subset = df[df["suspect_role_any"] & df["target_drug_group"].ne("both_zolpidem_and_other_zdrug")].copy()
    exposed = subset["is_zolpidem_suspect"]
    rows: list[dict[str, Any]] = []
    for col in PHENOTYPE_COLUMNS:
        counts = two_by_two_counts(exposed, subset[col])
        rows.append(
            {
                "comparison": "zolpidem_ps_ss_vs_other_suspect_drugs",
                "phenotype_column": col,
                **signal_metrics(**counts),
            }
        )
    return pd.DataFrame(rows).merge(dictionary, on="phenotype_column", how="left")


def run(period_token: str, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df, dictionary = _load_analysis_frame(period_token)
    descriptive = build_descriptive_spectrum(df, dictionary)
    within = build_within_zolpidem_fall_comparison(df, dictionary)
    signal = build_zolpidem_vs_other_suspect_signal(df, dictionary)

    outputs = {
        "descriptive_spectrum": output_dir / f"phenotype_descriptive_spectrum_{period_token}.csv",
        "within_zolpidem_fall_comparison": output_dir
        / f"phenotype_within_zolpidem_fall_comparison_{period_token}.csv",
        "zolpidem_vs_other_signal": output_dir / f"phenotype_zolpidem_vs_other_signal_{period_token}.csv",
    }
    descriptive.to_csv(outputs["descriptive_spectrum"], index=False, encoding="utf-8-sig")
    within.to_csv(outputs["within_zolpidem_fall_comparison"], index=False, encoding="utf-8-sig")
    signal.to_csv(outputs["zolpidem_vs_other_signal"], index=False, encoding="utf-8-sig")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fall phenotype spectrum analysis.")
    parser.add_argument("--period-token", default="2004_2025")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    outputs = run(args.period_token, args.output_dir)
    print("phenotype spectrum analysis completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

