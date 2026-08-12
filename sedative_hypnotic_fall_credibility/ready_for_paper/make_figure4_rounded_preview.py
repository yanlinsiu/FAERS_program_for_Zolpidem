from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Figure_4_phenotype_heatmap_source.csv"
OUTPUT = HERE / "Figure_4_phenotype_fingerprint_rounded_preview"

GROUPS = [
    ("zolpidem_only", "Zolpidem only", 745),
    ("other_z_drugs_without_zolpidem_only", "Other Z-drugs only", 438),
    ("benzodiazepines_only", "Benzodiazepines only", 1989),
]

PHENOTYPES = [
    ("phenotype_neurocognitive", "Neurocognitive/\nconsciousness"),
    ("phenotype_dizziness_syncope_hypotension", "Dizziness/syncope/\nhypotension"),
    ("phenotype_gait_balance", "Gait/\nbalance"),
    ("phenotype_sedation", "Sedation/\nsomnolence"),
    ("mixed_phenotype", "Mixed\nphenotype"),
    ("no_mechanistic_co_phenotype", "No predefined\nphenotype"),
]


def main() -> None:
    data = pd.read_csv(SOURCE, encoding="utf-8-sig")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )

    cmap = LinearSegmentedColormap.from_list(
        "lavender_indigo",
        ["#f4f2f8", "#e1dcf0", "#c5bae2", "#9b89c8", "#6d5aa8", "#3f3379"],
    )
    norm = Normalize(vmin=0, vmax=65)

    fig = plt.figure(figsize=(180 / 25.4, 82 / 25.4), facecolor="white")
    ax = fig.add_axes([0.245, 0.25, 0.70, 0.61])
    ax.set_xlim(0, 6)
    ax.set_ylim(3, 0)
    ax.set_aspect("auto")
    ax.axis("off")

    for row, (group_key, _, _) in enumerate(GROUPS):
        for col, (phenotype_key, _) in enumerate(PHENOTYPES):
            matched = data[
                data["drug_group"].eq(group_key)
                & data["phenotype_component"].eq(phenotype_key)
            ]
            if len(matched) != 1:
                raise ValueError(f"Expected one value for {group_key}/{phenotype_key}")
            value = float(matched.iloc[0]["proportion_percent"])
            face = cmap(norm(value))
            tile = FancyBboxPatch(
                (col + 0.075, row + 0.09),
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
                col + 0.5,
                row + 0.5,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                color="white" if luminance < 0.58 else "#252335",
            )

    for row, (_, label, n) in enumerate(GROUPS):
        ax.text(
            -0.15,
            row + 0.40,
            label,
            ha="right",
            va="center",
            fontsize=7.4,
            fontweight="bold" if row == 0 else "normal",
            color="#282633",
            clip_on=False,
        )
        ax.text(
            -0.15,
            row + 0.64,
            f"n = {n:,}",
            ha="right",
            va="center",
            fontsize=6.4,
            color="#777381",
            clip_on=False,
        )

    for col, (_, label) in enumerate(PHENOTYPES):
        ax.text(
            col + 0.5,
            3.13,
            label,
            ha="center",
            va="top",
            fontsize=6.15,
            linespacing=1.0,
            color="#37343f",
            clip_on=False,
        )

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

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".svg"))
    plt.close(fig)
    print(f"Wrote {OUTPUT.with_suffix('.png')}")
    print(f"Wrote {OUTPUT.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
