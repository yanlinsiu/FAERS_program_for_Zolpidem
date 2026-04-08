from __future__ import annotations

import argparse
import contextlib
import io
import importlib.util
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_project"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from config import DEFAULT_OUTPUT_ROOT, RAW_ROOT
from case_dataset_processor import process_case_dataset
from demo_processor import process_demo
from drug_exposure_processor import process_drug_exposure
from drug_feature_processor import process_drug_feature
from drug_processor import process_drug
from outc_processor import process_outc
from reac_processor import process_reac
from signal_dataset_processor import process_signal_dataset
from utils import build_file_path


def _load_analysis_function(file_name: str, function_name: str):
    module_path = ANALYSIS_ROOT / file_name
    spec = importlib.util.spec_from_file_location(function_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load analysis module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


build_signal_analysis = _load_analysis_function(
    "01_signal_analysis.py", "build_signal_analysis"
)
build_comparative_analysis = _load_analysis_function(
    "02_comparative_analysis.py", "build_comparative_analysis"
)
build_feature_analysis = _load_analysis_function(
    "03_feature_analysis.py", "build_feature_analysis"
)


QUARTERS = ("Q1", "Q2", "Q3", "Q4")

ANNUAL_DATASETS = {
    "case_base_dataset": "case_base_dataset_{year}.parquet",
    "demo": "demo_{year}.parquet",
    "drug": "drug_{year}.parquet",
    "drug_feature_dataset": "drug_feature_dataset_{year}.parquet",
    "drug_feature_case": "drug_feature_{year}_case.parquet",
    "drug_exposure_case": "drug_exposure_{year}_case.parquet",
    "reac_case": "reac_{year}_case.parquet",
    "outcome_dataset": "outcome_dataset_{year}.parquet",
    "outc_case": "outc_{year}_case.parquet",
    "case_dataset": "case_dataset_{year}.parquet",
    "signal_dataset": "signal_dataset_{year}.parquet",
}


def _available_quarters(year: int) -> list[str]:
    available: list[str] = []
    for quarter in QUARTERS:
        demo_file = build_file_path(RAW_ROOT, year, quarter, "DEMO")
        if demo_file.exists():
            available.append(quarter)
    return available


def _run_step(func, year: int, quarter: str, output_root: Path, quiet: bool):
    if not quiet:
        return func(year, quarter, output_root)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return func(year, quarter, output_root)


def _is_year_completed(year_root: Path) -> bool:
    summary_file = year_root / "analysis" / f"summary_{year_root.name}.txt"
    signal_file = year_root / "analysis" / "01_signal_analysis_results.csv"
    comparative_file = year_root / "analysis" / "02_comparative_analysis_results.csv"
    feature_file = year_root / "analysis" / "03_feature_analysis_results.csv"
    return all(path.exists() for path in [summary_file, signal_file, comparative_file, feature_file])


def _looks_like_parquet(file_path: Path) -> bool:
    if not file_path.exists() or not file_path.is_file():
        return False

    try:
        with file_path.open("rb") as handle:
            header = handle.read(4)
            if len(header) < 4:
                return False
            handle.seek(-4, 2)
            footer = handle.read(4)
    except OSError:
        return False

    return header == b"PAR1" and footer == b"PAR1"


def _concat_parquet_files(files: list[Path], output_file: Path) -> int:
    frames: list[pd.DataFrame] = []
    for file_path in files:
        if not _looks_like_parquet(file_path):
            size = file_path.stat().st_size if file_path.exists() else "missing"
            raise ValueError(
                f"Invalid parquet file detected before annual combine: {file_path} "
                f"(size={size})"
            )
        try:
            df = pd.read_parquet(file_path)
        except Exception as exc:
            size = file_path.stat().st_size if file_path.exists() else "missing"
            raise RuntimeError(
                f"Failed to read parquet during annual combine: {file_path} "
                f"(size={size})"
            ) from exc
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_file, index=False)
    return len(combined)


def _combine_year_outputs(year: int, quarterly_root: Path, year_root: Path) -> dict[str, int]:
    summary: dict[str, int] = {}
    year_token = str(year)

    patterns = {
        "case_base_dataset": f"case_base_dataset_{year_token}q*.parquet",
        "demo": f"demo_{year_token}q*.parquet",
        "drug": f"drug_{year_token}q*.parquet",
        "drug_feature_dataset": f"drug_feature_dataset_{year_token}q*.parquet",
        "drug_feature_case": f"drug_feature_{year_token}q*_case.parquet",
        "drug_exposure_case": f"drug_exposure_{year_token}q*_case.parquet",
        "reac_case": f"reac_{year_token}q*_case.parquet",
        "outcome_dataset": f"outcome_dataset_{year_token}q*.parquet",
        "outc_case": f"outc_{year_token}q*_case.parquet",
        "case_dataset": f"case_dataset_{year_token}q*.parquet",
        "signal_dataset": f"signal_dataset_{year_token}q*.parquet",
    }

    for key, pattern in patterns.items():
        files = sorted(quarterly_root.glob(pattern))
        if not files:
            continue
        output_file = year_root / ANNUAL_DATASETS[key].format(year=year_token)
        summary[key] = _concat_parquet_files(files, output_file)

    return summary


def _write_year_summary(
    year: int,
    year_root: Path,
    quarters: list[str],
    combined_summary: dict[str, int],
    signal_results: pd.DataFrame,
    comparative_results: pd.DataFrame,
    feature_results: pd.DataFrame,
) -> Path:
    summary_file = year_root / "analysis" / f"summary_{year}.txt"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"FAERS yearly summary: {year}",
        f"quarters processed: {', '.join(quarters)}",
        "",
        "annual combined outputs:",
    ]
    for key, row_count in sorted(combined_summary.items()):
        lines.append(f"- {key}: {row_count} rows")

    lines.extend(
        [
            "",
            "01 signal analysis:",
        ]
    )
    for row in signal_results.to_dict(orient="records"):
        ror_text = "NA" if pd.isna(row["ror"]) else f"{row['ror']:.4f}"
        prr_text = "NA" if pd.isna(row["prr"]) else f"{row['prr']:.4f}"
        ic_text = "NA" if pd.isna(row.get("ic")) else f"{row['ic']:.4f}"
        ebgm_text = "NA" if pd.isna(row.get("ebgm")) else f"{row['ebgm']:.4f}"
        lines.append(
            f"- {row['analysis']} | {row['outcome_name']} | {row['conclusion']}: "
            f"a={row['a']}, b={row['b']}, c={row['c']}, d={row['d']}, "
            f"ROR={ror_text}, PRR={prr_text}, IC={ic_text}, EBGM={ebgm_text}"
        )

    lines.extend(
        [
            "",
            "02 comparative analysis:",
        ]
    )
    for row in comparative_results.to_dict(orient="records"):
        ror_text = "NA" if pd.isna(row["ror"]) else f"{row['ror']:.4f}"
        prr_text = "NA" if pd.isna(row["prr"]) else f"{row['prr']:.4f}"
        ic_text = "NA" if pd.isna(row.get("ic")) else f"{row['ic']:.4f}"
        ebgm_text = "NA" if pd.isna(row.get("ebgm")) else f"{row['ebgm']:.4f}"
        lines.append(
            f"- {row['analysis']} | {row['outcome_name']} | {row['conclusion']}: "
            f"a={row['a']}, b={row['b']}, c={row['c']}, d={row['d']}, "
            f"ROR={ror_text}, PRR={prr_text}, IC={ic_text}, EBGM={ebgm_text}"
        )

    lines.extend(
        [
            "",
            "03 stratified feature analysis top rows by ROR:",
        ]
    )
    for (analysis_name, outcome_name), subset in feature_results.groupby(["analysis", "outcome_name"], sort=False):
        top_rows = subset.head(5)
        lines.append(f"- {analysis_name} | {outcome_name}:")
        for row in top_rows.to_dict(orient="records"):
            ror_text = "NA" if pd.isna(row["ror"]) else f"{row['ror']:.4f}"
            lines.append(
                f"  {row['feature_name']} | positive={row['n_feature_positive']} | "
                f"outcome={row['n_feature_positive_outcome']} | {row['conclusion']} | ROR={ror_text}"
            )

    summary_file.write_text("\n".join(lines), encoding="utf-8")
    return summary_file


def _build_year_trend_row(year: int, analysis_root: Path) -> pd.DataFrame:
    signal_results_file = analysis_root / "01_signal_analysis_results.csv"
    if not signal_results_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(signal_results_file)
    df["year"] = int(year)
    if "outcome_name" not in df.columns:
        df["outcome_name"] = "strict_fall"
    if "outcome_definition" not in df.columns:
        df["outcome_definition"] = "Narrow fall definition"
    if "conclusion" not in df.columns:
        signal_flag_ror = df["signal_flag_ror"].fillna(False) if "signal_flag_ror" in df.columns else pd.Series(False, index=df.index)
        signal_flag_mhra = df["signal_flag_mhra"].fillna(False) if "signal_flag_mhra" in df.columns else pd.Series(False, index=df.index)
        has_signal = signal_flag_ror | signal_flag_mhra
        df["conclusion"] = has_signal.map(
            {
                True: "存在不成比例性信号",
                False: "未见明确不成比例性信号",
            }
        )
    required_defaults = {
        "ic": None,
        "ic025": None,
        "ic975": None,
        "ebgm": None,
        "eb05": None,
        "eb95": None,
        "signal_flag_ic": False,
        "signal_flag_ebgm": False,
    }
    for column, default_value in required_defaults.items():
        if column not in df.columns:
            df[column] = default_value
    return df[
        [
            "year",
            "analysis",
            "outcome_name",
            "outcome_definition",
            "conclusion",
            "a",
            "b",
            "c",
            "d",
            "n",
            "ror",
            "ror_ci_low",
            "ror_ci_high",
            "prr",
            "prr_ci_low",
            "prr_ci_high",
            "ic",
            "ic025",
            "ic975",
            "ebgm",
            "eb05",
            "eb95",
            "chi_square_yates",
            "expected_a",
            "signal_flag_mhra",
            "signal_flag_ror",
            "signal_flag_ic",
            "signal_flag_ebgm",
        ]
    ].copy()


def _write_range_trend_summary(
    results: list[dict[str, object]],
    output_root: Path,
    start_year: int,
    end_year: int,
) -> Path | None:
    trend_frames = []
    for result in results:
        trend_row_df = _build_year_trend_row(result["year"], result["analysis_root"])
        if not trend_row_df.empty:
            trend_frames.append(trend_row_df)

    if not trend_frames:
        return None

    trend_df = pd.concat(trend_frames, ignore_index=True)
    output_file = output_root / f"trend_{start_year}_{end_year}.csv"
    trend_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_file


def process_year(
    year: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    quiet: bool = True,
    skip_completed: bool = True,
) -> dict[str, object]:
    year = int(year)
    base_output_root = Path(output_root)
    year_root = base_output_root / str(year)
    quarterly_root = year_root / "quarterly"
    analysis_root = year_root / "analysis"
    quarterly_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)

    if skip_completed and _is_year_completed(year_root):
        print(f"{year}: skipped (existing annual outputs found)")
        return {
            "year": year,
            "quarters": _available_quarters(year),
            "year_root": year_root,
            "quarterly_root": quarterly_root,
            "analysis_root": analysis_root,
            "quarter_summary_path": analysis_root / f"quarter_summary_{year}.csv",
            "summary_file": analysis_root / f"summary_{year}.txt",
            "signal_results": pd.read_csv(analysis_root / "01_signal_analysis_results.csv"),
            "signal_qc": pd.read_csv(analysis_root / "01_signal_analysis_qc.csv"),
            "comparative_results": pd.read_csv(analysis_root / "02_comparative_analysis_results.csv"),
            "comparative_qc": pd.read_csv(analysis_root / "02_comparative_analysis_qc.csv"),
            "feature_results": pd.read_csv(analysis_root / "03_feature_analysis_results.csv"),
            "feature_qc": pd.read_csv(analysis_root / "03_feature_analysis_qc.csv"),
        }

    quarters = _available_quarters(year)
    if not quarters:
        raise FileNotFoundError(f"No raw FAERS quarters found for year {year}.")

    steps = [
        process_demo,
        process_drug,
        process_drug_feature,
        process_drug_exposure,
        process_reac,
        process_outc,
        process_case_dataset,
        process_signal_dataset,
    ]

    quarter_summary_rows: list[dict[str, object]] = []
    for quarter in quarters:
        for step in steps:
            _run_step(step, year, quarter, quarterly_root, quiet=quiet)

        signal_file = quarterly_root / f"signal_dataset_{year}{quarter.lower()}.parquet"
        signal_df = pd.read_parquet(signal_file)
        quarter_summary_rows.append(
            {
                "year": year,
                "quarter": quarter,
                "n_cases": int(len(signal_df)),
                "n_fall_narrow": int(
                    signal_df["is_fall_narrow"].fillna(False).astype(bool).sum()
                ),
                "n_fall_broad": int(
                    signal_df["is_fall_broad"].fillna(False).astype(bool).sum()
                ) if "is_fall_broad" in signal_df.columns else None,
                "n_zolpidem_any": int(
                    signal_df["is_zolpidem_any"].fillna(False).astype(bool).sum()
                ) if "is_zolpidem_any" in signal_df.columns else None,
                "n_zolpidem_suspect": int(
                    signal_df["is_zolpidem_suspect"].fillna(False).astype(bool).sum()
                ),
                "n_other_zdrug_suspect": int(
                    signal_df["is_other_zdrug_suspect"].fillna(False).astype(bool).sum()
                ),
            }
        )
        print(
            f"{year} {quarter}: cases={quarter_summary_rows[-1]['n_cases']}, "
            f"fall_narrow={quarter_summary_rows[-1]['n_fall_narrow']}, "
            f"fall_broad={quarter_summary_rows[-1]['n_fall_broad']}, "
            f"zolpidem_any={quarter_summary_rows[-1]['n_zolpidem_any']}, "
            f"zolpidem_suspect={quarter_summary_rows[-1]['n_zolpidem_suspect']}"
        )

    combined_summary = _combine_year_outputs(year, quarterly_root, year_root)

    signal_results, signal_qc = build_signal_analysis(
        signal_root=quarterly_root,
        output_dir=analysis_root,
    )
    comparative_results, comparative_qc = build_comparative_analysis(
        signal_root=quarterly_root,
        output_dir=analysis_root,
    )
    feature_results, feature_qc = build_feature_analysis(
        signal_root=quarterly_root,
        output_dir=analysis_root,
    )

    quarter_summary_df = pd.DataFrame(quarter_summary_rows)
    quarter_summary_path = analysis_root / f"quarter_summary_{year}.csv"
    quarter_summary_df.to_csv(quarter_summary_path, index=False, encoding="utf-8-sig")

    summary_file = _write_year_summary(
        year,
        year_root,
        quarters,
        combined_summary,
        signal_results,
        comparative_results,
        feature_results,
    )

    return {
        "year": year,
        "quarters": quarters,
        "year_root": year_root,
        "quarterly_root": quarterly_root,
        "analysis_root": analysis_root,
        "quarter_summary_path": quarter_summary_path,
        "summary_file": summary_file,
        "signal_results": signal_results,
        "signal_qc": signal_qc,
        "comparative_results": comparative_results,
        "comparative_qc": comparative_qc,
        "feature_results": feature_results,
        "feature_qc": feature_qc,
    }


def process_year_range(
    start_year: int,
    end_year: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    quiet: bool = True,
    skip_completed: bool = True,
) -> list[dict[str, object]]:
    if int(start_year) > int(end_year):
        raise ValueError("start_year must be earlier than or equal to end_year")

    results: list[dict[str, object]] = []
    for year in range(int(start_year), int(end_year) + 1):
        print(f"Processing year {year}...")
        results.append(
            process_year(
                year,
                output_root=output_root,
                quiet=quiet,
                skip_completed=skip_completed,
            )
        )
    trend_file = _write_range_trend_summary(results, Path(output_root), int(start_year), int(end_year))
    if trend_file is not None:
        print(f"Trend summary: {trend_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run FAERS processing in yearly batches and save annual outputs."
    )
    parser.add_argument("--year", type=int, help="Single year to process, e.g. 2024")
    parser.add_argument("--start-year", type=int, help="First year in a year range")
    parser.add_argument("--end-year", type=int, help="Last year in a year range")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for annual outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-step processor output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild year outputs even if annual files already exist",
    )
    args = parser.parse_args()

    quiet = not args.verbose
    skip_completed = not args.force

    if args.year is not None:
        result = process_year(
            args.year,
            output_root=args.output_root,
            quiet=quiet,
            skip_completed=skip_completed,
        )
        print(f"Saved yearly outputs to: {result['year_root']}")
        print(f"Summary file: {result['summary_file']}")
        return

    if args.start_year is None or args.end_year is None:
        raise ValueError("Provide either --year or both --start-year and --end-year.")

    results = process_year_range(
        args.start_year,
        args.end_year,
        output_root=args.output_root,
        quiet=quiet,
        skip_completed=skip_completed,
    )
    print(f"Completed {len(results)} year(s).")
    for result in results:
        print(f"- {result['year_root']}")


if __name__ == "__main__":
    main()
