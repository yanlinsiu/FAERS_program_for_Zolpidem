from __future__ import annotations

import csv
import html
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
INPUT = PROJECT / "outputs" / "tables" / "table_4_phenotype_fingerprint_by_drug_group.csv"
SOURCE_CSV = HERE / "Figure_4_phenotype_heatmap_source.csv"
OUTPUT_STEM = HERE / "Figure_4_phenotype_heatmap"
PREVIEW_HTML = HERE / "Figure_4_phenotype_heatmap_preview.html"

WIDTH_MM = 180
HEIGHT_MM = 82
TIFF_DPI = 600

GROUPS = [
    ("zolpidem_only", "Zolpidem only", 745),
    ("other_z_drugs_without_zolpidem_only", "Other Z-drugs only", 438),
    ("benzodiazepines_only", "Benzodiazepines only", 1989),
]

PHENOTYPES = [
    ("phenotype_neurocognitive", "Neurocognitive/\nconsciousness"),
    ("phenotype_dizziness_syncope_hypotension", "Dizziness/\nsyncope/\nhypotension"),
    ("phenotype_gait_balance", "Gait/\nbalance"),
    ("phenotype_sedation", "Sedation/\nsomnolence"),
    ("mixed_phenotype", "Mixed\nphenotype"),
    ("no_mechanistic_co_phenotype", "No predefined\nmechanistic\nphenotype"),
]

CAPTION = (
    "Fig. 4 Phenotype fingerprint of strict-fall reports across mutually exclusive "
    "sedative-hypnotic exposure groups. Heatmap cells show the proportion of strict-fall "
    "reports containing each prespecified co-reported phenotype. Percentages were calculated "
    "within zolpidem-only, other Z-drug-only, and benzodiazepine-only strict-fall reports. "
    "The basic phenotype domains were not mutually exclusive; therefore, percentages within "
    "a row do not sum to 100%. Mixed phenotype denotes at least two predefined phenotype "
    "domains, and no predefined mechanistic phenotype denotes none of the predefined domains. "
    "Darker shading indicates a higher within-group reporting proportion. The heatmap is "
    "descriptive; adjusted pairwise phenotype comparisons are reported in Supplementary Table S8."
)


def load_and_validate() -> pd.DataFrame:
    raw = pd.read_csv(INPUT, encoding="utf-8-sig")
    required = {
        "drug_group",
        "drug_group_label",
        "phenotype_component",
        "phenotype_component_label",
        "fall_case_n",
        "phenotype_n",
        "phenotype_percent",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT}: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for group_key, group_label, expected_n in GROUPS:
        for phenotype_key, axis_label in PHENOTYPES:
            matched = raw[
                raw["drug_group"].eq(group_key)
                & raw["phenotype_component"].eq(phenotype_key)
            ]
            if len(matched) != 1:
                raise ValueError(f"Expected one row for {group_key}/{phenotype_key}, found {len(matched)}")
            source = matched.iloc[0]
            denominator = int(source["fall_case_n"])
            numerator = int(source["phenotype_n"])
            percent = float(source["phenotype_percent"])
            if denominator != expected_n:
                raise ValueError(f"Unexpected denominator for {group_label}: {denominator} != {expected_n}")
            calculated = numerator / denominator * 100
            if not np.isclose(percent, calculated, atol=1e-10):
                raise ValueError(f"Percentage mismatch for {group_label}/{phenotype_key}")
            rows.append(
                {
                    "drug_group": group_key,
                    "drug_group_label": group_label,
                    "strict_fall_reports_n": denominator,
                    "phenotype_component": phenotype_key,
                    "phenotype_label": axis_label.replace("\n", " "),
                    "phenotype_reports_n": numerator,
                    "proportion_percent": percent,
                    "display_percent": f"{percent:.1f}%",
                    "denominator_definition": "all strict-fall reports in the mutually exclusive exposure group",
                    "phenotype_overlap": "allowed",
                    "source_table": "table_4_phenotype_fingerprint_by_drug_group.csv",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(SOURCE_CSV, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    return result


def build_figure(source: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    cmap = LinearSegmentedColormap.from_list(
        "lavender_indigo",
        ["#f4f2f8", "#e1dcf0", "#c5bae2", "#9b89c8", "#6d5aa8", "#3f3379"],
    )
    norm = Normalize(vmin=0, vmax=65)
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    ax = fig.add_axes([0.245, 0.25, 0.70, 0.61])
    ax.set_xlim(0, len(PHENOTYPES))
    ax.set_ylim(len(GROUPS), 0)
    ax.axis("off")

    for row_idx, (group_key, _, _) in enumerate(GROUPS):
        for col_idx, (phenotype_key, _) in enumerate(PHENOTYPES):
            matched = source[
                source["drug_group"].eq(group_key)
                & source["phenotype_component"].eq(phenotype_key)
            ]
            if len(matched) != 1:
                raise ValueError(f"Expected one value for {group_key}/{phenotype_key}")
            value = float(matched.iloc[0]["proportion_percent"])
            face = cmap(norm(value))
            tile = FancyBboxPatch(
                (col_idx + 0.075, row_idx + 0.09),
                0.85,
                0.82,
                boxstyle="round,pad=0,rounding_size=0.12",
                linewidth=0.45,
                edgecolor=(1, 1, 1, 0.95),
                facecolor=face,
            )
            ax.add_patch(tile)
            luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="white" if luminance < 0.58 else "#252335",
            )

    for row_idx, (_, label, n) in enumerate(GROUPS):
        ax.text(-0.15, row_idx + 0.40, label, ha="right", va="center", fontsize=7.4,
                fontweight="bold" if row_idx == 0 else "normal", color="#282633", clip_on=False)
        ax.text(-0.15, row_idx + 0.64, f"n = {n:,}", ha="right", va="center", fontsize=6.4,
                color="#777381", clip_on=False)

    for col_idx, (_, label) in enumerate(PHENOTYPES):
        ax.text(col_idx + 0.5, 3.13, label, ha="center", va="top", fontsize=6.0,
                linespacing=1.0, color="#37343f", clip_on=False)

    ax.plot([0.08, 3.92], [-0.06, -0.06], color="#9287ad", lw=1.1, clip_on=False)
    ax.text(2.0, -0.12, "CO-REPORTED PHENOTYPES", ha="center", va="bottom", fontsize=5.8,
            fontweight="bold", color="#73698c", clip_on=False)
    ax.plot([4.08, 5.92], [-0.06, -0.06], color="#b5b0bd", lw=1.1, clip_on=False)
    ax.text(5.0, -0.12, "DERIVED SUMMARY", ha="center", va="bottom", fontsize=5.8,
            fontweight="bold", color="#807b86", clip_on=False)

    cax = fig.add_axes([0.735, 0.945, 0.21, 0.022])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 20, 40, 60])
    cbar.ax.tick_params(labelsize=5.8, length=2, width=0.45, pad=1.5, colors="#5f5b68")
    cbar.outline.set_visible(False)
    fig.text(0.725, 0.956, "Proportion (%)", ha="right", va="center", fontsize=6.1, color="#5f5b68")

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=TIFF_DPI, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    plt.close(fig)


def write_preview() -> None:
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure 4 phenotype heatmap preview</title>
<style>body{{margin:0;background:#eef2f5;color:#1f2933;font-family:Arial,sans-serif}}main{{max-width:1100px;margin:32px auto;padding:28px;background:white;box-shadow:0 4px 24px #0002}}img{{display:block;width:100%;height:auto}}p{{font-size:14px;line-height:1.6;margin:22px 0 0}}</style></head>
<body><main><img src="{OUTPUT_STEM.with_suffix('.png').name}" alt="Figure 4 phenotype heatmap"><p>{html.escape(CAPTION)}</p></main></body></html>"""
    PREVIEW_HTML.write_text(body, encoding="utf-8")


def main() -> None:
    source = load_and_validate()
    build_figure(source)
    write_preview()
    print(f"Validated {len(source)} heatmap cells from {INPUT.name}")
    for path in [
        SOURCE_CSV,
        OUTPUT_STEM.with_suffix(".png"),
        OUTPUT_STEM.with_suffix(".tiff"),
        OUTPUT_STEM.with_suffix(".pdf"),
        OUTPUT_STEM.with_suffix(".svg"),
        PREVIEW_HTML,
    ]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
