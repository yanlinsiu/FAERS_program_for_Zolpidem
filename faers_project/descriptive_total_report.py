from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import DEFAULT_OUTPUT_ROOT_PATH, PROJECT_ROOT


REPORT_ROOT = PROJECT_ROOT / "analysis_reports"


@dataclass
class ReportContext:
    case_file: Path
    report_dir: Path
    report_stem: str
    total_n: int


def _find_latest_annual_case_dataset(output_root: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for year_dir in output_root.iterdir():
        if not year_dir.is_dir():
            continue
        if not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        case_file = year_dir / f"case_dataset_{year}.parquet"
        if case_file.exists():
            candidates.append((year, case_file))

    if not candidates:
        raise FileNotFoundError(
            f"No annual case_dataset_YYYY.parquet found under: {output_root}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _coerce_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _safe_pct(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator) * 100, 2)


def _fmt_n_pct(n: int, pct: float | None) -> str:
    if pct is None:
        return f"{n}"
    return f"{n} ({pct:.2f}%)"


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _series_distribution(
    df: pd.DataFrame,
    column: str,
    total_n: int,
    label_map: dict[str, str] | None = None,
    include_missing: bool = True,
) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=["变量", "水平", "例数", "占总体百分比"])

    series = df[column].copy()
    series = series.astype("object")
    if include_missing:
        series = series.where(series.notna(), "缺失")
    else:
        series = series.dropna()

    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = ["水平", "例数"]
    counts["水平"] = counts["水平"].astype(str)
    if label_map:
        counts["水平"] = counts["水平"].map(lambda x: label_map.get(x, x))
    counts.insert(0, "变量", column)
    counts["占总体百分比"] = counts["例数"].map(lambda x: _safe_pct(x, total_n))
    return counts


def _bool_distribution(
    df: pd.DataFrame,
    columns: Iterable[str],
    total_n: int,
    true_label: str = "是",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        positive_n = int(_coerce_bool(df[column]).sum())
        rows.append(
            {
                "变量": column,
                "阳性定义": true_label,
                "例数": positive_n,
                "占总体百分比": _safe_pct(positive_n, total_n),
            }
        )
    return pd.DataFrame(rows)


def _available_bool_distribution(
    df: pd.DataFrame,
    columns: Iterable[str],
    total_n: int,
    true_label: str = "是",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        if int(df[column].notna().sum()) == 0:
            continue
        positive_n = int(_coerce_bool(df[column]).sum())
        rows.append(
            {
                "变量": column,
                "阳性定义": true_label,
                "例数": positive_n,
                "占总体百分比": _safe_pct(positive_n, total_n),
            }
        )
    return pd.DataFrame(rows)


def _age_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "age_years" not in df.columns:
        return pd.DataFrame(columns=["指标", "数值"])

    age = pd.to_numeric(df["age_years"], errors="coerce")
    valid_age = age.dropna()
    if valid_age.empty:
        return pd.DataFrame(columns=["指标", "数值"])

    rows = [
        {"指标": "年龄非缺失例数", "数值": int(valid_age.shape[0])},
        {"指标": "平均年龄", "数值": round(float(valid_age.mean()), 2)},
        {"指标": "年龄标准差", "数值": round(float(valid_age.std()), 2)},
        {"指标": "中位年龄", "数值": round(float(valid_age.median()), 2)},
        {"指标": "P25", "数值": round(float(valid_age.quantile(0.25)), 2)},
        {"指标": "P75", "数值": round(float(valid_age.quantile(0.75)), 2)},
        {"指标": "最小年龄", "数值": round(float(valid_age.min()), 2)},
        {"指标": "最大年龄", "数值": round(float(valid_age.max()), 2)},
    ]
    return pd.DataFrame(rows)


def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        "caseid",
        "primaryid",
        "fda_dt",
        "age_years",
        "age_group",
        "sex_clean",
        "serious",
        "is_serious_any",
        "is_fall_narrow",
        "is_fall_broad",
        "fall_pt_list",
        "is_zolpidem",
        "is_zolpidem_suspect",
        "is_zolpidem_suspect_ps",
        "target_drug_group",
        "target_drug_group_ps",
        "polypharmacy_5",
        "drug_n",
        "distinct_drug_n",
    ]
    rows: list[dict[str, object]] = []
    total_n = len(df)
    for column in key_columns:
        if column not in df.columns:
            continue
        if (
            column == "serious"
            and "is_serious_any" in df.columns
            and int(df["serious"].notna().sum()) == 0
            and int(df["is_serious_any"].notna().sum()) > 0
        ):
            # Newer FAERS DEMO files no longer carry raw `serious`;
            # skip the synthetic compatibility placeholder when OUTC-derived
            # seriousness is available.
            continue
        missing_n = int(df[column].isna().sum())
        rows.append(
            {
                "变量": column,
                "缺失例数": missing_n,
                "缺失比例": _safe_pct(missing_n, total_n),
            }
        )
    return pd.DataFrame(rows)


def _overview(df: pd.DataFrame, case_file: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"指标": "数据文件", "数值": str(case_file)},
        {"指标": "总病例数", "数值": int(len(df))},
        {
            "指标": "唯一 caseid 数",
            "数值": int(df["caseid"].nunique()) if "caseid" in df.columns else None,
        },
        {
            "指标": "唯一 primaryid 数",
            "数值": int(df["primaryid"].nunique()) if "primaryid" in df.columns else None,
        },
        {"指标": "变量数", "数值": int(len(df.columns))},
    ]

    if "year" in df.columns:
        years = sorted(df["year"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "指标": "覆盖年份",
                "数值": ", ".join(map(str, years)) if years else "NA",
            }
        )
    if "quarter" in df.columns:
        quarters = df["quarter"].dropna().astype(str)
        rows.append(
            {
                "指标": "覆盖季度",
                "数值": ", ".join(sorted(quarters.unique().tolist())) if not quarters.empty else "NA",
            }
        )
    if "fda_dt" in df.columns:
        dt = pd.to_datetime(df["fda_dt"], errors="coerce").dropna()
        if not dt.empty:
            rows.append({"指标": "最早 FDA_DT", "数值": str(dt.min().date())})
            rows.append({"指标": "最晚 FDA_DT", "数值": str(dt.max().date())})
    return pd.DataFrame(rows)


def _exposure_outcome_crosstab(
    df: pd.DataFrame,
    exposure_col: str,
    outcome_cols: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if exposure_col not in df.columns:
        return pd.DataFrame(
            columns=["暴露变量", "结局变量", "暴露状态", "病例数", "结局例数", "结局报告率"]
        )

    exposed = _coerce_bool(df[exposure_col])
    for outcome_col in outcome_cols:
        if outcome_col not in df.columns:
            continue
        outcome = _coerce_bool(df[outcome_col])
        for label, mask in [("暴露组", exposed), ("非暴露组", ~exposed)]:
            n_cases = int(mask.sum())
            n_outcome = int((mask & outcome).sum())
            rows.append(
                {
                    "暴露变量": exposure_col,
                    "结局变量": outcome_col,
                    "暴露状态": label,
                    "病例数": n_cases,
                    "结局例数": n_outcome,
                    "结局报告率": round(n_outcome / n_cases, 6) if n_cases else None,
                }
            )
    return pd.DataFrame(rows)


def _target_group_distribution(df: pd.DataFrame, total_n: int) -> pd.DataFrame:
    frames = []
    for column in ["target_drug_group", "target_drug_group_ps"]:
        dist = _series_distribution(df, column, total_n, include_missing=True)
        if not dist.empty:
            frames.append(dist)
    if not frames:
        return pd.DataFrame(columns=["变量", "水平", "例数", "占总体百分比"])
    return pd.concat(frames, ignore_index=True)


def _top_fall_pt(df: pd.DataFrame, total_n: int, top_n: int = 20) -> pd.DataFrame:
    if "fall_pt_list" not in df.columns:
        return pd.DataFrame(columns=["PT", "涉及病例数", "占总体百分比", "占 broad_fall 百分比"])

    broad_mask = _coerce_bool(df["is_fall_broad"]) if "is_fall_broad" in df.columns else None
    if broad_mask is None or int(broad_mask.sum()) == 0:
        return pd.DataFrame(columns=["PT", "涉及病例数", "占总体百分比", "占 broad_fall 百分比"])

    if "caseid" not in df.columns:
        return pd.DataFrame(columns=["PT", "涉及病例数", "占总体百分比", "占 broad_fall 百分比"])

    pt_df = df.loc[broad_mask, ["caseid", "fall_pt_list"]].copy()
    pt_df = pt_df.dropna(subset=["fall_pt_list"])
    pt_df["PT"] = pt_df["fall_pt_list"].astype(str).str.split("|", regex=False)
    pt_df = pt_df.explode("PT")
    pt_df["PT"] = pt_df["PT"].astype(str).str.strip()
    pt_df = pt_df[pt_df["PT"].ne("")]
    pt_df = pt_df[["caseid", "PT"]].drop_duplicates()
    if pt_df.empty:
        return pd.DataFrame(columns=["PT", "涉及病例数", "占总体百分比", "占 broad_fall 百分比"])

    counts = pt_df["PT"].value_counts().head(top_n).reset_index()
    counts.columns = ["PT", "涉及病例数"]
    broad_n = int(broad_mask.sum())
    counts["占总体百分比"] = counts["涉及病例数"].map(lambda x: _safe_pct(x, total_n))
    counts["占 broad_fall 百分比"] = counts["涉及病例数"].map(lambda x: _safe_pct(x, broad_n))
    return counts


def _write_csv_tables(tables: dict[str, pd.DataFrame], report_dir: Path) -> None:
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _build_markdown_report(
    ctx: ReportContext,
    overview_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    age_summary_df: pd.DataFrame,
    age_group_df: pd.DataFrame,
    sex_df: pd.DataFrame,
    serious_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    target_group_df: pd.DataFrame,
    med_burden_df: pd.DataFrame,
    comed_df: pd.DataFrame,
    crosstab_df: pd.DataFrame,
    top_pt_df: pd.DataFrame,
) -> str:
    total_n = ctx.total_n

    def first_count(df: pd.DataFrame, column: str) -> int:
        if df.empty or column not in df.columns:
            return 0
        return int(df.iloc[0][column])

    strict_fall_n = 0
    broad_fall_n = 0
    zolpidem_any_n = 0
    zolpidem_suspect_n = 0
    zolpidem_suspect_ps_n = 0

    if not outcome_df.empty:
        for _, row in outcome_df.iterrows():
            if row["变量"] == "is_fall_narrow":
                strict_fall_n = int(row["例数"])
            elif row["变量"] == "is_fall_broad":
                broad_fall_n = int(row["例数"])

    if not exposure_df.empty:
        for _, row in exposure_df.iterrows():
            if row["变量"] == "is_zolpidem":
                zolpidem_any_n = int(row["例数"])
            elif row["变量"] == "is_zolpidem_suspect":
                zolpidem_suspect_n = int(row["例数"])
            elif row["变量"] == "is_zolpidem_suspect_ps":
                zolpidem_suspect_ps_n = int(row["例数"])

    lines = [
        f"# 总表描述性统计报告：{ctx.report_stem}",
        "",
        "## 1. 报告说明",
        f"- 数据文件：`{ctx.case_file}`",
        f"- 报告目录：`{ctx.report_dir}`",
        "- 统计目标：对当前总表进行基线描述，不涉及年度趋势推断。",
        "- 说明：百分比默认以总病例数为分母；结局报告率以对应分组病例数为分母。",
        "",
        "## 2. 样本概况",
        f"- 总病例数：{total_n}",
        f"- zolpidem 任意暴露：{_fmt_n_pct(zolpidem_any_n, _safe_pct(zolpidem_any_n, total_n))}",
        f"- zolpidem suspect（PS+SS）：{_fmt_n_pct(zolpidem_suspect_n, _safe_pct(zolpidem_suspect_n, total_n))}",
        f"- zolpidem suspect（PS only）：{_fmt_n_pct(zolpidem_suspect_ps_n, _safe_pct(zolpidem_suspect_ps_n, total_n))}",
        f"- 狭义跌倒：{_fmt_n_pct(strict_fall_n, _safe_pct(strict_fall_n, total_n))}",
        f"- 广义跌倒相关：{_fmt_n_pct(broad_fall_n, _safe_pct(broad_fall_n, total_n))}",
        "",
        "## 3. 数据完整性",
    ]

    if missing_df.empty:
        lines.append("- 未生成缺失统计表。")
    else:
        for _, row in missing_df.sort_values("缺失比例", ascending=False).head(8).iterrows():
            lines.append(
                f"- {row['变量']}：缺失 {int(row['缺失例数'])} 例（{_fmt_num(row['缺失比例'])}%）"
            )

    lines.extend(
        [
            "",
            "## 4. 人群基本特征",
        ]
    )
    if not age_summary_df.empty:
        metrics = dict(zip(age_summary_df["指标"], age_summary_df["数值"]))
        lines.extend(
            [
                f"- 年龄均值 ± SD：{_fmt_num(metrics.get('平均年龄'))} ± {_fmt_num(metrics.get('年龄标准差'))} 岁",
                f"- 年龄中位数（IQR）：{_fmt_num(metrics.get('中位年龄'))}（{_fmt_num(metrics.get('P25'))}, {_fmt_num(metrics.get('P75'))}）岁",
                f"- 年龄范围：{_fmt_num(metrics.get('最小年龄'))} 到 {_fmt_num(metrics.get('最大年龄'))} 岁",
            ]
        )

    if not age_group_df.empty:
        age_parts = []
        for _, row in age_group_df.iterrows():
            age_parts.append(f"{row['水平']} {int(row['例数'])}例（{_fmt_num(row['占总体百分比'])}%）")
        lines.append("- 年龄组分布：" + "；".join(age_parts))

    if not sex_df.empty:
        sex_parts = []
        for _, row in sex_df.iterrows():
            sex_parts.append(f"{row['水平']} {int(row['例数'])}例（{_fmt_num(row['占总体百分比'])}%）")
        lines.append("- 性别分布：" + "；".join(sex_parts))

    if not serious_df.empty:
        serious_parts = []
        for _, row in serious_df.iterrows():
            serious_parts.append(f"{row['变量']} {int(row['例数'])}例（{_fmt_num(row['占总体百分比'])}%）")
        lines.append("- 严重性相关指标：" + "；".join(serious_parts))

    if not med_burden_df.empty:
        med_parts = []
        for _, row in med_burden_df.iterrows():
            med_parts.append(f"{row['变量']} {int(row['例数'])}例（{_fmt_num(row['占总体百分比'])}%）")
        lines.append("- 用药负担：" + "；".join(med_parts))

    if not comed_df.empty:
        lines.extend(
            [
                "",
                "## 5. 合并用药特征",
            ]
        )
        for _, row in comed_df.iterrows():
            lines.append(
                f"- {row['变量']}：{int(row['例数'])} 例（{_fmt_num(row['占总体百分比'])}%）"
            )

    if not exposure_df.empty:
        lines.extend(
            [
                "",
                "## 6. 暴露定义分布",
            ]
        )
        for _, row in exposure_df.iterrows():
            lines.append(
                f"- {row['变量']}：{int(row['例数'])} 例（{_fmt_num(row['占总体百分比'])}%）"
            )

    if not target_group_df.empty:
        lines.extend(
            [
                "",
                "## 7. 目标药物分组构成",
            ]
        )
        for column in target_group_df["变量"].drop_duplicates():
            lines.append(f"- {column}：")
            subset = target_group_df[target_group_df["变量"] == column]
            for _, row in subset.iterrows():
                lines.append(
                    f"  - {row['水平']}：{int(row['例数'])} 例（{_fmt_num(row['占总体百分比'])}%）"
                )

    if not outcome_df.empty:
        lines.extend(
            [
                "",
                "## 8. 结局概况",
            ]
        )
        for _, row in outcome_df.iterrows():
            lines.append(
                f"- {row['变量']}：{int(row['例数'])} 例（{_fmt_num(row['占总体百分比'])}%）"
            )

    if not crosstab_df.empty:
        lines.extend(
            [
                "",
                "## 9. 暴露与结局的粗分布",
            ]
        )
        for exposure_col in crosstab_df["暴露变量"].drop_duplicates():
            lines.append(f"- {exposure_col}：")
            subset = crosstab_df[crosstab_df["暴露变量"] == exposure_col]
            for outcome_col in subset["结局变量"].drop_duplicates():
                outcome_subset = subset[subset["结局变量"] == outcome_col]
                for _, row in outcome_subset.iterrows():
                    rate = (
                        f"{row['结局报告率'] * 100:.2f}%"
                        if pd.notna(row["结局报告率"])
                        else "NA"
                    )
                    lines.append(
                        f"  - {outcome_col} | {row['暴露状态']}：{int(row['结局例数'])}/{int(row['病例数'])}，报告率 {rate}"
                    )

    if not top_pt_df.empty:
        lines.extend(
            [
                "",
                "## 10. broad_fall 中最常见 PT（前 20）",
            ]
        )
        for _, row in top_pt_df.iterrows():
            lines.append(
                f"- {row['PT']}：{int(row['涉及病例数'])} 例，占总体 {_fmt_num(row['占总体百分比'])}%，占 broad_fall {_fmt_num(row['占 broad_fall 百分比'])}%"
            )

    lines.extend(
        [
            "",
            "## 11. 输出文件",
            "- `01_overview.csv`：样本概况",
            "- `02_missingness.csv`：关键变量缺失统计",
            "- `03_age_summary.csv`：年龄连续变量摘要",
            "- `04_age_group_distribution.csv`：年龄组分布",
            "- `05_sex_distribution.csv`：性别分布",
            "- `06_serious_distribution.csv`：严重性分布",
            "- `07_outcome_distribution.csv`：结局分布",
            "- `08_exposure_distribution.csv`：暴露定义分布",
            "- `09_target_drug_group_distribution.csv`：目标药物分组分布",
            "- `10_medication_burden_distribution.csv`：用药负担分布",
            "- `11_comedication_distribution.csv`：合并用药分布",
            "- `12_exposure_outcome_crosstab.csv`：暴露-结局粗交叉表",
            "- `13_top_fall_pt.csv`：broad_fall 中最常见 PT",
        ]
    )
    return "\n".join(lines) + "\n"


def build_descriptive_report(case_file: Path, output_dir: Path | None = None) -> Path:
    case_file = case_file.resolve()
    if not case_file.exists():
        raise FileNotFoundError(f"Case dataset not found: {case_file}")

    df = pd.read_parquet(case_file)
    report_stem = case_file.stem
    report_dir = (output_dir or REPORT_ROOT / report_stem).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    ctx = ReportContext(
        case_file=case_file,
        report_dir=report_dir,
        report_stem=report_stem,
        total_n=len(df),
    )

    overview_df = _overview(df, case_file)
    missing_df = _missingness(df)
    age_summary_df = _age_summary(df)
    age_group_df = _series_distribution(df, "age_group", len(df))
    sex_df = _series_distribution(df, "sex_clean", len(df))
    serious_df = _available_bool_distribution(df, ["serious", "is_serious_any"], len(df))
    outcome_df = _bool_distribution(df, ["is_fall_narrow", "is_fall_broad"], len(df))
    exposure_df = _bool_distribution(
        df,
        [
            "is_zolpidem",
            "is_zolpidem_any",
            "is_zolpidem_suspect",
            "is_zolpidem_suspect_ps",
            "is_other_zdrug_suspect",
            "is_other_zdrug_suspect_ps",
            "suspect_role_any",
            "suspect_role_any_ps",
        ],
        len(df),
    )
    target_group_df = _target_group_distribution(df, len(df))
    med_burden_df = _bool_distribution(df, ["polypharmacy_5", "polypharmacy"], len(df))
    comed_df = _bool_distribution(
        df,
        [
            "is_benzo",
            "is_antidepressant",
            "is_antipsychotic",
            "is_opioid",
            "is_antiepileptic",
        ],
        len(df),
    )
    crosstab_df = pd.concat(
        [
            _exposure_outcome_crosstab(df, "is_zolpidem", ["is_fall_narrow", "is_fall_broad"]),
            _exposure_outcome_crosstab(
                df,
                "is_zolpidem_suspect",
                ["is_fall_narrow", "is_fall_broad"],
            ),
            _exposure_outcome_crosstab(
                df,
                "is_zolpidem_suspect_ps",
                ["is_fall_narrow", "is_fall_broad"],
            ),
        ],
        ignore_index=True,
    )
    top_pt_df = _top_fall_pt(df, len(df), top_n=20)

    tables = {
        "01_overview": overview_df,
        "02_missingness": missing_df,
        "03_age_summary": age_summary_df,
        "04_age_group_distribution": age_group_df,
        "05_sex_distribution": sex_df,
        "06_serious_distribution": serious_df,
        "07_outcome_distribution": outcome_df,
        "08_exposure_distribution": exposure_df,
        "09_target_drug_group_distribution": target_group_df,
        "10_medication_burden_distribution": med_burden_df,
        "11_comedication_distribution": comed_df,
        "12_exposure_outcome_crosstab": crosstab_df,
        "13_top_fall_pt": top_pt_df,
    }
    _write_csv_tables(tables, report_dir)

    markdown = _build_markdown_report(
        ctx=ctx,
        overview_df=overview_df,
        missing_df=missing_df,
        age_summary_df=age_summary_df,
        age_group_df=age_group_df,
        sex_df=sex_df,
        serious_df=serious_df,
        outcome_df=outcome_df,
        exposure_df=exposure_df,
        target_group_df=target_group_df,
        med_burden_df=med_burden_df,
        comed_df=comed_df,
        crosstab_df=crosstab_df,
        top_pt_df=top_pt_df,
    )
    report_file = report_dir / "总表描述性统计报告.md"
    report_file.write_text(markdown, encoding="utf-8")
    return report_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Chinese descriptive report for a FAERS case dataset.")
    parser.add_argument(
        "--case-file",
        type=str,
        default="",
        help="Path to case_dataset parquet. Defaults to the latest annual case_dataset under OUTPUT.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to store analysis report files. Defaults to PROJECT_ROOT/analysis_reports/<case_dataset_stem>.",
    )
    args = parser.parse_args()

    case_file = (
        Path(args.case_file).expanduser()
        if args.case_file
        else _find_latest_annual_case_dataset(DEFAULT_OUTPUT_ROOT_PATH)
    )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else None
    report_file = build_descriptive_report(case_file=case_file, output_dir=output_dir)
    print(f"Saved report: {report_file}")


if __name__ == "__main__":
    main()
