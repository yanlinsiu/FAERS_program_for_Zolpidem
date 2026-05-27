from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "OUTPUT"
DEFAULT_GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_reports" / "country_analysis"


@dataclass(frozen=True)
class AgeBand:
    label: str
    lower: float
    upper: float | None


DEFAULT_AGE_BANDS = [
    AgeBand("65-74岁", 65, 75),
    AgeBand("75-84岁", 75, 85),
    AgeBand(">=85岁", 85, None),
]

COUNTRY_NORMALIZATION = {
    "ARGENTINA": "AR",
    "AUSTRALIA": "AU",
    "BELGIUM": "BE",
    "BRAZIL": "BR",
    "CANADA": "CA",
    "CHINA": "CN",
    "COLOMBIA": "CO",
    "COUNTRY NOT SPECIFIED": "UNKNOWN",
    "DENMARK": "DK",
    "FRANCE": "FR",
    "GERMANY": "DE",
    "GREAT BRITAIN": "GB",
    "INDIA": "IN",
    "ITALY": "IT",
    "JAPAN": "JP",
    "KOREA": "KR",
    "NETHERLANDS": "NL",
    "PORTUGAL": "PT",
    "REPUBLIC OF KOREA": "KR",
    "SOUTH AFRICA": "ZA",
    "SOUTH KOREA": "KR",
    "SPAIN": "ES",
    "SWITZERLAND": "CH",
    "TAIWAN": "TW",
    "UNITED KINGDOM": "GB",
    "UNITED STATES": "US",
    "UNITED STATES OF AMERICA": "US",
    "USA": "US",
}


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.where(series.notna(), "").astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"])


def _country_clean(series: pd.Series) -> pd.Series:
    country = series.where(series.notna(), "UNKNOWN").astype(str).str.strip().str.upper()
    country = country.replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
    return country.replace(COUNTRY_NORMALIZATION)


def _find_case_files(input_root: Path, start_year: int, end_year: int) -> list[Path]:
    files: list[Path] = []
    for year in range(start_year, end_year + 1):
        file_path = input_root / str(year) / f"case_dataset_{year}.parquet"
        if file_path.exists():
            files.append(file_path)
    if not files:
        raise FileNotFoundError(
            f"No annual case_dataset files found under {input_root} for {start_year}-{end_year}."
        )
    return files


def _load_cases(
    files: list[Path],
    country_column: str,
    fall_column: str,
) -> pd.DataFrame:
    required = ["caseid", "age_years", "year", "quarter", country_column, fall_column]
    frames: list[pd.DataFrame] = []
    for file_path in files:
        available = set(pd.read_parquet(file_path).columns)
        missing = [col for col in required if col not in available]
        if missing:
            raise ValueError(
                f"{file_path} missing columns: {missing}. "
                "Re-run faers_project/year_batch_runner.py so OUTPUT includes country fields."
            )
        frames.append(pd.read_parquet(file_path, columns=required))
    return pd.concat(frames, ignore_index=True)


def _apply_global_case_index(
    df: pd.DataFrame,
    global_index_file: Path | None,
) -> pd.DataFrame:
    if global_index_file is None:
        return df
    if not global_index_file.exists():
        raise FileNotFoundError(f"Global case index not found: {global_index_file}")

    index_cols = ["caseid", "year", "quarter"]
    global_index = pd.read_parquet(global_index_file, columns=index_cols)
    global_index["caseid"] = (
        global_index["caseid"].where(global_index["caseid"].notna(), "").astype(str).str.strip()
    )
    global_index["year"] = pd.to_numeric(global_index["year"], errors="coerce").astype("Int64")
    global_index["quarter"] = (
        global_index["quarter"]
        .where(global_index["quarter"].notna(), "")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    out = df.copy()
    out["caseid"] = out["caseid"].where(out["caseid"].notna(), "").astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["quarter"] = out["quarter"].where(out["quarter"].notna(), "").astype(str).str.upper().str.strip()
    return out.merge(global_index.drop_duplicates(), on=index_cols, how="inner")


def _assign_age_band(age: pd.Series, age_bands: list[AgeBand]) -> pd.Series:
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for band in age_bands:
        mask = age >= band.lower
        if band.upper is not None:
            mask &= age < band.upper
        out.loc[mask] = band.label
    return out


def _distribution_by_age_band(
    df: pd.DataFrame,
    age_bands: list[AgeBand],
    top_n: int,
    include_other: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    side_parts: list[pd.DataFrame] = []

    for band in age_bands:
        subset = df[df["age_band"] == band.label].copy()
        total = int(len(subset))
        counts = (
            subset.groupby("country", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="报告数")
        )
        counts.insert(0, "年龄段", band.label)
        counts["百分比"] = (counts["报告数"] / total * 100).round(2) if total else 0.0
        counts["年龄段总报告数"] = total
        rows.extend(counts.to_dict(orient="records"))

        display = counts.head(top_n).copy()
        if include_other and len(counts) > top_n:
            other_count = int(counts.iloc[top_n:]["报告数"].sum())
            other_row = pd.DataFrame(
                [
                    {
                        "年龄段": band.label,
                        "country": "Other",
                        "报告数": other_count,
                        "百分比": round(other_count / total * 100, 2) if total else 0.0,
                        "年龄段总报告数": total,
                    }
                ]
            )
            display = pd.concat([display, other_row], ignore_index=True)

        display = display.reset_index(drop=True)
        display.insert(0, "序号", range(1, len(display) + 1))
        display = display.rename(
            columns={
                "country": f"报告国家（{band.label}）",
                "报告数": f"报告数（{band.label}）",
                "百分比": f"百分比（{band.label}）",
            }
        )
        side_parts.append(
            display[
                [
                    "序号",
                    f"报告国家（{band.label}）",
                    f"报告数（{band.label}）",
                    f"百分比（{band.label}）",
                ]
            ]
        )

    long_df = pd.DataFrame(rows).rename(columns={"country": "报告国家"})
    side_by_side = pd.concat(side_parts, axis=1)
    return long_df, side_by_side


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        rows.append(values)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in rows:
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_markdown(
    output_file: Path,
    side_by_side_df: pd.DataFrame,
    total_n: int,
    start_year: int,
    end_year: int,
    min_age: float,
    age_operator: str,
    fall_column: str,
    country_column: str,
    global_index_file: Path | None,
) -> None:
    global_index_text = f"`{global_index_file}`" if global_index_file else "未使用"
    lines = [
        "# 跌倒病例报告国家分布表",
        "",
        f"- 数据年份：{start_year}-{end_year}",
        f"- 筛选条件：年龄 {age_operator} {min_age:g} 岁，且 `{fall_column}` 为 True",
        f"- 国家字段：`{country_column}`",
        f"- 跨年度去重索引：{global_index_text}",
        f"- 纳入病例数：{total_n:,}",
        "- 百分比：以对应年龄段内的跌倒病例数为分母",
        "",
        _dataframe_to_markdown(side_by_side_df),
        "",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")


def build_country_fall_distribution(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    global_index_file: Path | None = None,
    use_global_index: bool = True,
    start_year: int = 2004,
    end_year: int = 2025,
    min_age: float = 65,
    age_operator: str = ">=",
    fall_column: str = "is_fall",
    country_column: str = "reporter_country",
    top_n: int = 10,
    include_other: bool = True,
) -> dict[str, Path]:
    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _find_case_files(input_root, start_year, end_year)
    df = _load_cases(files, country_column=country_column, fall_column=fall_column)
    if use_global_index and global_index_file is None:
        candidate = DEFAULT_GLOBAL_DATASET_DIR / f"global_case_index_{start_year}_{end_year}.parquet"
        if candidate.exists():
            global_index_file = candidate
    df = _apply_global_case_index(df, global_index_file if use_global_index else None)
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["country"] = _country_clean(df[country_column])
    df["fall"] = _coerce_bool(df[fall_column])

    if age_operator == ">":
        age_mask = df["age_years"] > min_age
    elif age_operator == ">=":
        age_mask = df["age_years"] >= min_age
    else:
        raise ValueError("age_operator must be '>' or '>='.")

    filtered = df[age_mask & df["fall"]].copy()
    filtered["age_band"] = _assign_age_band(filtered["age_years"], DEFAULT_AGE_BANDS)
    filtered = filtered[filtered["age_band"].notna()].copy()

    long_df, side_by_side_df = _distribution_by_age_band(
        filtered,
        age_bands=DEFAULT_AGE_BANDS,
        top_n=top_n,
        include_other=include_other,
    )

    period = f"{start_year}_{end_year}"
    fall_name = fall_column.replace("is_", "")
    country_name = country_column.replace("_country", "")
    age_token = age_operator.replace(">=", "gteq").replace(">", "gt")
    output_prefix = f"country_{country_name}_{fall_name}_age{age_token}{min_age:g}_{period}"

    long_file = output_dir / f"{output_prefix}_long.csv"
    table_file = output_dir / f"{output_prefix}_side_by_side.csv"
    markdown_file = output_dir / f"{output_prefix}.md"

    long_df.to_csv(long_file, index=False, encoding="utf-8-sig")
    side_by_side_df.to_csv(table_file, index=False, encoding="utf-8-sig")
    _write_markdown(
        markdown_file,
        side_by_side_df=side_by_side_df,
        total_n=len(filtered),
        start_year=start_year,
        end_year=end_year,
        min_age=min_age,
        age_operator=age_operator,
        fall_column=fall_column,
        country_column=country_column,
        global_index_file=global_index_file,
    )

    return {
        "long": long_file,
        "side_by_side": table_file,
        "markdown": markdown_file,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build country distribution tables for fall cases.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--global-index-file",
        type=Path,
        default=None,
        help=(
            "Optional global_case_index parquet for cross-year case deduplication. "
            "Defaults to OUTPUT_GLOBAL/datasets/global_case_index_<period>.parquet when present."
        ),
    )
    parser.add_argument(
        "--no-global-index",
        action="store_true",
        help="Do not use the global case index; annual files will be concatenated as-is.",
    )
    parser.add_argument("--min-age", type=float, default=65)
    parser.add_argument("--age-operator", choices=[">", ">="], default=">=")
    parser.add_argument(
        "--fall-column",
        choices=["is_fall"],
        default="is_fall",
    )
    parser.add_argument(
        "--country-column",
        choices=["reporter_country", "occr_country"],
        default="reporter_country",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--no-other", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = build_country_fall_distribution(
        input_root=args.input_root,
        output_dir=args.output_dir,
        global_index_file=args.global_index_file,
        use_global_index=not args.no_global_index,
        start_year=args.start_year,
        end_year=args.end_year,
        min_age=args.min_age,
        age_operator=args.age_operator,
        fall_column=args.fall_column,
        country_column=args.country_column,
        top_n=args.top_n,
        include_other=not args.no_other,
    )

    print("Country fall distribution tables saved:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
