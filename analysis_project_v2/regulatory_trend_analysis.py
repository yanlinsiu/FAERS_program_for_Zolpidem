from __future__ import annotations

import argparse
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from .data import load_analysis_frame, resolve_dataset_bundle
    from .signal_metrics import signal_metrics, two_by_two_counts
except ImportError:
    from data import load_analysis_frame, resolve_dataset_bundle
    from signal_metrics import signal_metrics, two_by_two_counts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL_COUNTRY" / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL_COUNTRY" / "regulatory_trend"

START_YEAR = 2004
END_YEAR = 2025
EVENT_YEARS = {
    2013: "FDA zolpidem dose reduction / next-morning impairment",
    2019: "FDA boxed warning for Z-drugs",
    2023: "AGS Beers Criteria update",
}


@dataclass(frozen=True)
class TrendSpec:
    analysis: str
    exposure_column: str
    suspect_column: str
    group_column: str
    outcome_name: str
    outcome_column: str
    outcome_definition: str
    comparison: str


@dataclass(frozen=True)
class EventPeriod:
    event_name: str
    event_year: int
    period_label: str
    start_year: int
    end_year: int
    interpretation_scope: str


TREND_SPECS: tuple[TrendSpec, ...] = (
    TrendSpec(
        analysis="primary_ps_ss",
        exposure_column="is_zolpidem_suspect",
        suspect_column="suspect_role_any",
        group_column="target_drug_group",
        outcome_name="strict_fall",
        outcome_column="is_fall_narrow",
        outcome_definition="OCMQ-compatible narrow fall event",
        comparison="zolpidem_suspect_vs_all_other_suspect_drugs_excluding_mixed_zdrug_cases",
    ),
    TrendSpec(
        analysis="sensitivity_ps_only",
        exposure_column="is_zolpidem_suspect_ps",
        suspect_column="suspect_role_any_ps",
        group_column="target_drug_group_ps",
        outcome_name="strict_fall",
        outcome_column="is_fall_narrow",
        outcome_definition="OCMQ-compatible narrow fall event",
        comparison="zolpidem_primary_suspect_vs_all_other_primary_suspect_drugs_excluding_mixed_zdrug_cases",
    ),
)

EVENT_PERIODS: tuple[EventPeriod, ...] = (
    EventPeriod("2013_fda_dose_reduction", 2013, "pre_2013", 2004, 2012, "main"),
    EventPeriod("2013_fda_dose_reduction", 2013, "post_2013", 2013, 2025, "main"),
    EventPeriod("2019_fda_boxed_warning", 2019, "pre_2019", 2004, 2018, "main"),
    EventPeriod("2019_fda_boxed_warning", 2019, "post_2019", 2019, 2025, "main"),
    EventPeriod("2023_ags_beers_criteria", 2023, "pre_2023", 2004, 2022, "exploratory"),
    EventPeriod("2023_ags_beers_criteria", 2023, "post_2023", 2023, 2025, "exploratory"),
)

ROLLING_WINDOW_YEARS = 3
PAPER_ANALYSIS = "primary_ps_ss"
PAPER_OUTCOME = "strict_fall"


def _coerce_year(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result = result[result["year"].between(START_YEAR, END_YEAR)].copy()
    result["year"] = result["year"].astype(int)
    return result


def _analysis_subset(df: pd.DataFrame, spec: TrendSpec) -> pd.DataFrame:
    subset = df[df[spec.suspect_column].fillna(False).astype(bool)].copy()
    subset = subset[subset[spec.group_column] != "both_zolpidem_and_other_zdrug"].copy()
    return subset


def _metrics_row(base: dict[str, Any], subset: pd.DataFrame, spec: TrendSpec) -> dict[str, Any]:
    counts = two_by_two_counts(subset[spec.exposure_column], subset[spec.outcome_column])
    metrics = signal_metrics(**counts)
    exposed_n = metrics["exposed_n"]
    return {
        **base,
        "zolpidem_cases": int(exposed_n),
        "zolpidem_fall_cases": int(metrics["a"]),
        "zolpidem_fall_reporting_rate": metrics["a"] / exposed_n if exposed_n else None,
        **metrics,
        "stability_status": (
            "stable" if metrics["a"] >= 5 and metrics["exposed_n"] >= 50 else "unstable"
        ),
    }


def build_annual_trend(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = list(range(START_YEAR, END_YEAR + 1))
    for spec in TREND_SPECS:
        subset = _analysis_subset(df, spec)
        for year in years:
            year_subset = subset[subset["year"].eq(year)]
            rows.append(
                _metrics_row(
                    {
                        "year": year,
                        "analysis": spec.analysis,
                        "comparison": spec.comparison,
                        "exposure_definition": spec.exposure_column,
                        "outcome_name": spec.outcome_name,
                        "outcome_definition": spec.outcome_definition,
                    },
                    year_subset,
                    spec,
                )
            )
    return pd.DataFrame(rows)


def build_event_period_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in TREND_SPECS:
        subset = _analysis_subset(df, spec)
        for period in EVENT_PERIODS:
            period_subset = subset[
                subset["year"].between(period.start_year, period.end_year)
            ].copy()
            rows.append(
                _metrics_row(
                    {
                        "event_name": period.event_name,
                        "event_year": period.event_year,
                        "period_label": period.period_label,
                        "start_year": period.start_year,
                        "end_year": period.end_year,
                        "interpretation_scope": period.interpretation_scope,
                        "analysis": spec.analysis,
                        "comparison": spec.comparison,
                        "exposure_definition": spec.exposure_column,
                        "outcome_name": spec.outcome_name,
                        "outcome_definition": spec.outcome_definition,
                    },
                    period_subset,
                    spec,
                )
            )
    return pd.DataFrame(rows)


def build_rolling_trend(
    df: pd.DataFrame,
    window_years: int = ROLLING_WINDOW_YEARS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if window_years < 2:
        raise ValueError("Rolling trend window must be at least 2 years.")

    for spec in TREND_SPECS:
        subset = _analysis_subset(df, spec)
        for start_year in range(START_YEAR, END_YEAR - window_years + 2):
            end_year = start_year + window_years - 1
            window_subset = subset[subset["year"].between(start_year, end_year)]
            rows.append(
                _metrics_row(
                    {
                        "window_label": f"{start_year}-{end_year}",
                        "window_years": window_years,
                        "start_year": start_year,
                        "end_year": end_year,
                        "center_year": start_year + (window_years - 1) / 2,
                        "analysis": spec.analysis,
                        "comparison": spec.comparison,
                        "exposure_definition": spec.exposure_column,
                        "outcome_name": spec.outcome_name,
                        "outcome_definition": spec.outcome_definition,
                    },
                    window_subset,
                    spec,
                )
            )
    return pd.DataFrame(rows)


def build_paper_regulatory_trend_summary(
    df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> pd.DataFrame:
    all_period_subset = _analysis_subset(df, TREND_SPECS[0])
    all_period = _metrics_row(
        {
            "summary_type": "full_period",
            "event_name": "full_period",
            "event_year": None,
            "period_label": "2004_2025",
            "start_year": START_YEAR,
            "end_year": END_YEAR,
            "interpretation_scope": "main",
            "analysis": PAPER_ANALYSIS,
            "outcome_name": PAPER_OUTCOME,
            "outcome_definition": TREND_SPECS[0].outcome_definition,
        },
        all_period_subset,
        TREND_SPECS[0],
    )

    period_rows = event_df[
        event_df["analysis"].eq(PAPER_ANALYSIS)
        & event_df["outcome_name"].eq(PAPER_OUTCOME)
        & event_df["interpretation_scope"].eq("main")
    ].copy()
    period_rows.insert(0, "summary_type", "regulatory_period")
    summary = pd.concat([pd.DataFrame([all_period]), period_rows], ignore_index=True, sort=False)
    summary["zolpidem_fall_reporting_rate_pct"] = summary["zolpidem_fall_reporting_rate"] * 100
    summary["reference_fall_reporting_rate_pct"] = summary["reporting_rate_unexposed"] * 100
    summary["ror_95ci"] = summary.apply(
        lambda row: (
            f"{row['ror']:.2f} ({row['ror_ci_low']:.2f}-{row['ror_ci_high']:.2f})"
            if pd.notna(row["ror"]) and pd.notna(row["ror_ci_low"]) and pd.notna(row["ror_ci_high"])
            else ""
        ),
        axis=1,
    )
    summary["interpretation_note"] = (
        "FAERS reporting pattern only; event-year splits are descriptive and not causal effects."
    )
    columns = [
        "summary_type",
        "event_name",
        "event_year",
        "period_label",
        "start_year",
        "end_year",
        "interpretation_scope",
        "analysis",
        "outcome_name",
        "outcome_definition",
        "zolpidem_cases",
        "zolpidem_fall_cases",
        "zolpidem_fall_reporting_rate",
        "zolpidem_fall_reporting_rate_pct",
        "reporting_rate_unexposed",
        "reference_fall_reporting_rate_pct",
        "ror",
        "ror_ci_low",
        "ror_ci_high",
        "ror_95ci",
        "prr",
        "ic",
        "ebgm",
        "stability_status",
        "interpretation_note",
    ]
    return summary[columns]


def _invalid_rate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        df["exposed_n"].gt(0)
        & (
            (
                df["zolpidem_fall_cases"] / df["zolpidem_cases"]
                - df["zolpidem_fall_reporting_rate"]
            ).abs()
            > 1e-12
        )
    ]


def _invalid_cell_total_count(df: pd.DataFrame) -> int:
    check = df.copy()
    check["cell_total"] = check[["a", "b", "c", "d"]].sum(axis=1)
    return int((check["cell_total"] != check["n"]).sum())


def build_qc(
    df: pd.DataFrame,
    annual_df: pd.DataFrame,
    event_df: pd.DataFrame,
    rolling_df: pd.DataFrame,
    paper_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "check_name": "dataset_rows",
            "status": "info",
            "value": int(len(df)),
            "expected_or_rule": "Should match current thesis population, about 4,155,023 older cases.",
        },
        {
            "check_name": "year_coverage",
            "status": (
                "pass"
                if sorted(annual_df["year"].dropna().unique().tolist())
                == list(range(START_YEAR, END_YEAR + 1))
                else "fail"
            ),
            "value": f"{int(annual_df['year'].min())}-{int(annual_df['year'].max())}",
            "expected_or_rule": "Annual rows cover 2004-2025.",
        },
    ]

    primary_full = event_df[
        event_df["event_name"].eq("2013_fda_dose_reduction")
        & event_df["period_label"].eq("post_2013")
        & event_df["analysis"].eq("primary_ps_ss")
        & event_df["outcome_name"].eq("strict_fall")
    ]
    all_period_subset = _analysis_subset(df, TREND_SPECS[0])
    all_period_row = _metrics_row(
        {
            "analysis": TREND_SPECS[0].analysis,
            "outcome_name": TREND_SPECS[0].outcome_name,
        },
        all_period_subset,
        TREND_SPECS[0],
    )
    rows.append(
        {
            "check_name": "full_period_primary_ror",
            "status": "pass" if 4.6 <= float(all_period_row["ror"]) <= 4.9 else "review",
            "value": float(all_period_row["ror"]),
            "expected_or_rule": "Should reproduce current primary ROR near 4.80.",
        }
    )
    rows.append(
        {
            "check_name": "full_period_primary_reporting_rate",
            "status": (
                "pass"
                if 0.13 <= float(all_period_row["zolpidem_fall_reporting_rate"]) <= 0.14
                else "review"
            ),
            "value": float(all_period_row["zolpidem_fall_reporting_rate"]),
            "expected_or_rule": "Should reproduce current exposed fall reporting proportion near 13.57%.",
        }
    )

    invalid_annual_cell_total = _invalid_cell_total_count(annual_df)
    invalid_event_cell_total = _invalid_cell_total_count(event_df)
    invalid_rolling_cell_total = _invalid_cell_total_count(rolling_df)
    invalid_annual_rate = _invalid_rate_rows(annual_df)
    invalid_event_rate = _invalid_rate_rows(event_df)
    invalid_rolling_rate = _invalid_rate_rows(rolling_df)
    rows.extend(
        [
            {
                "check_name": "annual_cell_totals",
                "status": "pass" if invalid_annual_cell_total == 0 else "fail",
                "value": invalid_annual_cell_total,
                "expected_or_rule": "For every annual row, a+b+c+d equals n.",
            },
            {
                "check_name": "annual_reporting_rate_formula",
                "status": "pass" if invalid_annual_rate.empty else "fail",
                "value": int(len(invalid_annual_rate)),
                "expected_or_rule": "zolpidem_fall_reporting_rate equals a/(a+b).",
            },
            {
                "check_name": "event_cell_totals",
                "status": "pass" if invalid_event_cell_total == 0 else "fail",
                "value": invalid_event_cell_total,
                "expected_or_rule": "For every event-period row, a+b+c+d equals n.",
            },
            {
                "check_name": "event_reporting_rate_formula",
                "status": "pass" if invalid_event_rate.empty else "fail",
                "value": int(len(invalid_event_rate)),
                "expected_or_rule": "zolpidem_fall_reporting_rate equals a/(a+b).",
            },
            {
                "check_name": "rolling_window_coverage",
                "status": (
                    "pass"
                    if sorted(
                        rolling_df[
                            rolling_df["analysis"].eq(PAPER_ANALYSIS)
                            & rolling_df["outcome_name"].eq(PAPER_OUTCOME)
                        ]["window_label"].tolist()
                    )
                    == [
                        f"{year}-{year + ROLLING_WINDOW_YEARS - 1}"
                        for year in range(START_YEAR, END_YEAR - ROLLING_WINDOW_YEARS + 2)
                    ]
                    else "fail"
                ),
                "value": int(
                    len(
                        rolling_df[
                            rolling_df["analysis"].eq(PAPER_ANALYSIS)
                            & rolling_df["outcome_name"].eq(PAPER_OUTCOME)
                        ]
                    )
                ),
                "expected_or_rule": "Primary rolling rows cover every 3-year window from 2004-2006 through 2023-2025.",
            },
            {
                "check_name": "rolling_cell_totals",
                "status": "pass" if invalid_rolling_cell_total == 0 else "fail",
                "value": invalid_rolling_cell_total,
                "expected_or_rule": "For every rolling row, a+b+c+d equals n.",
            },
            {
                "check_name": "rolling_reporting_rate_formula",
                "status": "pass" if invalid_rolling_rate.empty else "fail",
                "value": int(len(invalid_rolling_rate)),
                "expected_or_rule": "zolpidem_fall_reporting_rate equals a/(a+b).",
            },
            {
                "check_name": "paper_summary_scope",
                "status": (
                    "pass"
                    if set(paper_df["event_name"].dropna())
                    == {"full_period", "2013_fda_dose_reduction", "2019_fda_boxed_warning"}
                    else "fail"
                ),
                "value": ",".join(paper_df["event_name"].dropna().astype(str).unique().tolist()),
                "expected_or_rule": "Paper summary contains full period plus 2013 and 2019 main FDA periods only.",
            },
            {
                "check_name": "unstable_annual_rows",
                "status": "info",
                "value": int(annual_df["stability_status"].eq("unstable").sum()),
                "expected_or_rule": "Rows with a < 5 or exposed_n < 50 are marked unstable.",
            },
            {
                "check_name": "unused_sanity_slice",
                "status": "info",
                "value": int(len(primary_full)),
                "expected_or_rule": "Nonzero only confirms event table shape; not used for inference.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


Color = tuple[int, int, int]


class SimpleCanvas:
    def __init__(self, width: int, height: int, background: Color = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * width * height)

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx : idx + 3] = bytes(color)

    def line(self, x0: float, y0: float, x1: float, y1: float, color: Color, width: int = 1) -> None:
        dx = x1 - x0
        dy = y1 - y0
        steps = max(int(abs(dx)), int(abs(dy)), 1)
        radius = max(width // 2, 0)
        for step in range(steps + 1):
            x = int(round(x0 + dx * step / steps))
            y = int(round(y0 + dy * step / steps))
            for ox in range(-radius, radius + 1):
                for oy in range(-radius, radius + 1):
                    self.set_pixel(x + ox, y + oy, color)

    def circle(self, cx: float, cy: float, radius: int, color: Color) -> None:
        cx_i = int(round(cx))
        cy_i = int(round(cy))
        for x in range(cx_i - radius, cx_i + radius + 1):
            for y in range(cy_i - radius, cy_i + radius + 1):
                if (x - cx_i) ** 2 + (y - cy_i) ** 2 <= radius**2:
                    self.set_pixel(x, y, color)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                self.set_pixel(x, y, color)

    def text(self, x: int, y: int, text: str, color: Color, scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            for row_idx, row in enumerate(glyph):
                for col_idx, pixel in enumerate(row):
                    if pixel == "1":
                        self.rect(
                            cursor + col_idx * scale,
                            y + row_idx * scale,
                            cursor + (col_idx + 1) * scale - 1,
                            y + (row_idx + 1) * scale - 1,
                            color,
                        )
            cursor += (len(glyph[0]) + 1) * scale

    def save_png(self, path: Path) -> None:
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        raw = bytearray()
        row_bytes = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * row_bytes
            raw.extend(self.pixels[start : start + row_bytes])

        path.parent.mkdir(parents=True, exist_ok=True)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(png)


FONT_5X7 = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    ".": ["0", "0", "0", "0", "0", "0", "1"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}


def _scale(values: Iterable[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return (0.0, 1.0)
    low = min(finite)
    high = max(finite)
    if low == high:
        return (max(0.0, low - 1.0), high + 1.0)
    pad = (high - low) * 0.12
    return (max(0.0, low - pad), high + pad)


def _draw_line_chart(
    points: pd.DataFrame,
    y_col: str,
    output_file: Path,
    title: str,
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
) -> None:
    canvas = SimpleCanvas(1100, 650)
    left, right, top, bottom = 90, 1040, 75, 570
    axis_color = (50, 50, 50)
    grid_color = (225, 225, 225)
    line_color = (30, 95, 170)
    ci_color = (160, 190, 225)
    event_color = (205, 80, 65)

    values = points[y_col].dropna().astype(float).tolist()
    if ci_low_col and ci_high_col:
        values.extend(points[ci_low_col].dropna().astype(float).tolist())
        values.extend(points[ci_high_col].dropna().astype(float).tolist())
    y_min, y_max = _scale(values)

    def x_pos(year: float) -> float:
        return left + (float(year) - START_YEAR) / (END_YEAR - START_YEAR) * (right - left)

    def y_pos(value: float) -> float:
        return bottom - (float(value) - y_min) / (y_max - y_min) * (bottom - top)

    canvas.text(90, 20, title, (35, 35, 35), scale=2)
    canvas.rect(90, 48, 120, 61, line_color)
    canvas.text(128, 48, "ESTIMATE", (35, 35, 35), scale=1)
    canvas.rect(220, 48, 250, 61, event_color)
    canvas.text(258, 48, "EVENT YEAR", (35, 35, 35), scale=1)
    if ci_low_col and ci_high_col:
        canvas.rect(380, 48, 410, 61, ci_color)
        canvas.text(418, 48, "95% CI", (35, 35, 35), scale=1)

    for idx in range(6):
        y = top + (bottom - top) * idx / 5
        canvas.line(left, y, right, y, grid_color)
        label_value = y_max - (y_max - y_min) * idx / 5
        canvas.text(8, int(y) - 5, f"{label_value:.2f}", axis_color, scale=1)
    for year in range(START_YEAR, END_YEAR + 1, 2):
        x = x_pos(year)
        canvas.line(x, bottom, x, bottom + 8, axis_color)
    for year in [2004, 2013, 2019, 2023, 2025]:
        canvas.text(int(x_pos(year)) - 12, bottom + 18, str(year), axis_color, scale=1)
    for event_year in EVENT_YEARS:
        x = x_pos(event_year)
        canvas.line(x, top, x, bottom, event_color, width=3)
        canvas.text(int(x) - 12, top - 17, str(event_year), event_color, scale=1)

    canvas.line(left, top, left, bottom, axis_color, width=2)
    canvas.line(left, bottom, right, bottom, axis_color, width=2)

    clean = points[["year", y_col]].dropna().sort_values("year")
    if ci_low_col and ci_high_col:
        ci = points[["year", ci_low_col, ci_high_col]].dropna().sort_values("year")
        for _, row in ci.iterrows():
            x = x_pos(row["year"])
            canvas.line(x, y_pos(row[ci_low_col]), x, y_pos(row[ci_high_col]), ci_color, width=2)

    previous: tuple[float, float] | None = None
    for _, row in clean.iterrows():
        current = (x_pos(row["year"]), y_pos(row[y_col]))
        if previous is not None:
            canvas.line(previous[0], previous[1], current[0], current[1], line_color, width=4)
        canvas.circle(current[0], current[1], 4, line_color)
        previous = current

    canvas.save_png(output_file)


def write_figures(annual_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    primary = annual_df[
        annual_df["analysis"].eq("primary_ps_ss") & annual_df["outcome_name"].eq("strict_fall")
    ].copy()
    rate_file = figure_dir / "annual_reporting_rate.png"
    ror_file = figure_dir / "annual_ror.png"
    _draw_line_chart(
        primary,
        "zolpidem_fall_reporting_rate",
        rate_file,
        title="Annual zolpidem fall reporting rate",
    )
    _draw_line_chart(
        primary,
        "ror",
        ror_file,
        title="Annual ROR for zolpidem fall reports",
        ci_low_col="ror_ci_low",
        ci_high_col="ror_ci_high",
    )
    note_file = figure_dir / "figure_notes.md"
    note_file.write_text(
        "\n".join(
            [
                "# Figure notes",
                "",
                "- Blue line: annual primary analysis estimate for narrow fall reports.",
                "- Red vertical lines: 2013 FDA zolpidem dose reduction, 2019 FDA boxed warning for Z-drugs, and 2023 AGS Beers Criteria update.",
                "- Light blue vertical intervals in annual_ror.png: 95% CI for annual ROR.",
                "- Event lines are interpretive reference points and should not be read as causal effects.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "annual_reporting_rate_figure": rate_file,
        "annual_ror_figure": ror_file,
        "figure_notes": note_file,
    }


def run(period_token: str | None, dataset_dir: Path, output_dir: Path) -> dict[str, Path]:
    bundle = resolve_dataset_bundle(dataset_dir=dataset_dir, period_token=period_token)
    df = _coerce_year(load_analysis_frame(bundle))

    annual_df = build_annual_trend(df)
    event_df = build_event_period_comparison(df)
    rolling_df = build_rolling_trend(df)
    paper_df = build_paper_regulatory_trend_summary(df, event_df)
    qc_df = build_qc(df, annual_df, event_df, rolling_df, paper_df)

    outputs = {
        "annual_trend": output_dir / "annual_trend.csv",
        "rolling_trend": output_dir / "rolling_trend.csv",
        "event_period_comparison": output_dir / "event_period_comparison.csv",
        "paper_regulatory_trend_summary": output_dir / "paper_regulatory_trend_summary.csv",
        "annual_trend_qc": output_dir / "annual_trend_qc.csv",
    }
    _write_csv(annual_df, outputs["annual_trend"])
    _write_csv(rolling_df, outputs["rolling_trend"])
    _write_csv(event_df, outputs["event_period_comparison"])
    _write_csv(paper_df, outputs["paper_regulatory_trend_summary"])
    _write_csv(qc_df, outputs["annual_trend_qc"])
    outputs.update(write_figures(annual_df, output_dir))

    failed = qc_df[qc_df["status"].eq("fail")]
    if not failed.empty:
        raise AssertionError(f"Regulatory trend QC failed: {failed.to_dict(orient='records')}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build annual and regulatory-event trend analyses for zolpidem fall reports."
    )
    parser.add_argument("--period-token", default="2004_2025")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    outputs = run(
        period_token=args.period_token,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )
    print("regulatory_trend_analysis completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
