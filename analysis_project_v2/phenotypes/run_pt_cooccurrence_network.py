from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

try:
    from .build_phenotype_features import PHENOTYPE_SPECS, _find_meddra_file, _norm_term
    from .build_phenotype_features import build_meddra_llt_to_pt_map
except ImportError:
    from build_phenotype_features import PHENOTYPE_SPECS, _find_meddra_file, _norm_term
    from build_phenotype_features import build_meddra_llt_to_pt_map


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
CLEANED_OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "pt_cooccurrence_network"

EXCLUDED_NETWORK_TERMS = {
    # In a fall-restricted cohort this term is constant by design, so it
    # connects to every retained PT and hides the more useful structure.
    "FALL",
    # Very broad administrative/outcome terms tend to dominate the graph and
    # add little mechanistic detail. Keep them in the raw case-PT table.
    "HOSPITALISATION",
    "DEATH",
}

COMMUNITY_COLORS = (
    "#2f6f9f",
    "#c45a3c",
    "#5f8d4e",
    "#9b5b8d",
    "#d49b2a",
    "#4f7f7a",
    "#7a6bb0",
    "#8a6f4d",
)


def _load_signal_cohort(signal_file: Path, cohort: str) -> pd.DataFrame:
    signal = pd.read_parquet(signal_file)
    for col in [
        "is_zolpidem_suspect",
        "is_zolpidem_suspect_ps",
        "suspect_role_any",
        "suspect_role_any_ps",
        "is_fall_narrow",
    ]:
        if col in signal.columns:
            signal[col] = signal[col].fillna(False).astype(bool)

    if cohort == "zolpidem_ps_ss_strict_fall":
        mask = (
            signal["is_zolpidem_suspect"]
            & signal["suspect_role_any"]
            & signal["target_drug_group"].ne("both_zolpidem_and_other_zdrug")
            & signal["is_fall_narrow"]
        )
    elif cohort == "zolpidem_ps_ss_all":
        mask = (
            signal["is_zolpidem_suspect"]
            & signal["suspect_role_any"]
            & signal["target_drug_group"].ne("both_zolpidem_and_other_zdrug")
        )
    elif cohort == "zolpidem_ps_strict_fall":
        mask = (
            signal["is_zolpidem_suspect_ps"]
            & signal["suspect_role_any_ps"]
            & signal["target_drug_group_ps"].ne("both_zolpidem_and_other_zdrug")
            & signal["is_fall_narrow"]
        )
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    out = signal.loc[mask, ["caseid", "year", "quarter"]].copy()
    out["caseid"] = out["caseid"].where(out["caseid"].notna(), "").astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["quarter"] = out["quarter"].where(out["quarter"].notna(), "").astype(str).str.upper().str.strip()
    out = out[(out["caseid"] != "") & out["year"].notna() & out["quarter"].ne("")]
    return out.drop_duplicates()


def _load_case_index(case_index_file: Path, cohort_cases: pd.DataFrame) -> pd.DataFrame:
    case_index = pd.read_parquet(case_index_file, columns=["caseid", "primaryid", "year", "quarter"])
    case_index["caseid"] = case_index["caseid"].where(case_index["caseid"].notna(), "").astype(str).str.strip()
    case_index["primaryid"] = pd.to_numeric(case_index["primaryid"], errors="coerce")
    case_index["year"] = pd.to_numeric(case_index["year"], errors="coerce").astype("Int64")
    case_index["quarter"] = (
        case_index["quarter"].where(case_index["quarter"].notna(), "").astype(str).str.upper().str.strip()
    )
    cohort_ids = cohort_cases[["caseid"]].drop_duplicates()
    out = case_index.merge(cohort_ids, on="caseid", how="inner")
    out = out[(out["caseid"] != "") & out["primaryid"].notna() & out["year"].notna() & out["quarter"].ne("")]
    return out.drop_duplicates()


def _read_reac_for_cohort(
    cohort_index: pd.DataFrame,
    cleaned_output_root: Path,
    llt_to_pt: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    for (year, quarter), quarter_cases in cohort_index.groupby(["year", "quarter"], sort=True):
        year_int = int(year)
        quarter_str = str(quarter).upper()
        file_path = cleaned_output_root / str(year_int) / "quarterly" / f"reac_event_{year_int}{quarter_str.lower()}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Cleaned REAC event file not found: {file_path}")

        reac = pd.read_parquet(file_path, columns=["caseid", "primaryid", "pt"])
        reac["caseid"] = reac["caseid"].where(reac["caseid"].notna(), "").astype(str).str.strip()
        reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
        reac["reported_term"] = reac["pt"].map(_norm_term)
        reac = reac[(reac["caseid"] != "") & reac["primaryid"].notna() & reac["reported_term"].ne("")]

        matched = reac.merge(
            quarter_cases[["caseid", "primaryid"]].drop_duplicates(),
            on=["caseid", "primaryid"],
            how="inner",
        )
        if not matched.empty:
            matched["pt"] = matched["reported_term"].map(llt_to_pt).fillna(matched["reported_term"])
            parts.append(matched[["caseid", "primaryid", "pt", "reported_term"]])

        qc_rows.append(
            {
                "year": year_int,
                "quarter": quarter_str,
                "cohort_cases": int(quarter_cases["caseid"].nunique()),
                "matched_reac_rows": int(len(matched)),
                "matched_cases_with_reac": int(matched["caseid"].nunique()) if not matched.empty else 0,
                "unmapped_reported_term_rows": int((~matched["reported_term"].isin(llt_to_pt)).sum())
                if not matched.empty
                else 0,
            }
        )

    if not parts:
        return pd.DataFrame(columns=["caseid", "primaryid", "pt", "reported_term"]), pd.DataFrame(qc_rows)
    case_pt = pd.concat(parts, ignore_index=True).drop_duplicates()
    return case_pt, pd.DataFrame(qc_rows)


def _resolve_cleaned_output_root(cleaned_output_root: Path, cohort_index: pd.DataFrame) -> Path:
    sample = cohort_index.sort_values(["year", "quarter"]).iloc[0]
    year = int(sample["year"])
    quarter = str(sample["quarter"]).upper()
    expected = cleaned_output_root / str(year) / "quarterly" / f"reac_event_{year}{quarter.lower()}.parquet"
    if expected.exists():
        return cleaned_output_root

    candidates = sorted((PROJECT_ROOT / "runs").glob("*/OUTPUT"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        candidate_file = candidate / str(year) / "quarterly" / f"reac_event_{year}{quarter.lower()}.parquet"
        if candidate_file.exists():
            return candidate
    return cleaned_output_root


def _phenotype_lookup() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for spec in PHENOTYPE_SPECS:
        for pt in spec.pt_terms:
            lookup[pt] = {
                "phenotype_column": spec.column,
                "phenotype_layer": spec.layer,
                "phenotype_label": spec.label,
            }
    return lookup


def _build_edges(case_pt: pd.DataFrame, selected_terms: set[str], n_cases: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = case_pt[case_pt["pt"].isin(selected_terms)][["caseid", "pt"]].drop_duplicates()
    term_counts = filtered.groupby("pt")["caseid"].nunique().to_dict()

    pair_counts: Counter[tuple[str, str]] = Counter()
    for _, terms in filtered.groupby("caseid")["pt"]:
        unique_terms = sorted(set(terms))
        if len(unique_terms) < 2:
            continue
        pair_counts.update(itertools.combinations(unique_terms, 2))

    edge_rows: list[dict[str, Any]] = []
    for (pt1, pt2), co_count in pair_counts.items():
        n1 = int(term_counts[pt1])
        n2 = int(term_counts[pt2])
        union_n = n1 + n2 - co_count
        a = co_count
        b = n1 - co_count
        c = n2 - co_count
        d = n_cases - a - b - c
        denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
        edge_rows.append(
            {
                "pt1": pt1,
                "pt2": pt2,
                "co_count": int(co_count),
                "pt1_case_count": n1,
                "pt2_case_count": n2,
                "jaccard": co_count / union_n if union_n else 0.0,
                "lift": (co_count * n_cases) / (n1 * n2) if n1 and n2 else 0.0,
                "phi": ((a * d) - (b * c)) / denom if denom else 0.0,
            }
        )

    edges = pd.DataFrame(edge_rows)
    if not edges.empty:
        edges = edges.sort_values(["co_count", "jaccard", "pt1", "pt2"], ascending=[False, False, True, True])

    nodes = (
        filtered.groupby("pt")["caseid"]
        .nunique()
        .reset_index(name="case_count")
        .sort_values(["case_count", "pt"], ascending=[False, True])
    )
    nodes["case_percent"] = nodes["case_count"].map(lambda n: round(n / n_cases * 100, 4) if n_cases else 0.0)
    return nodes, edges


def _default_community_min_weight(weight_col: str) -> float:
    if weight_col == "jaccard":
        return 0.15
    if weight_col == "phi":
        return 0.25
    if weight_col == "lift":
        return 2.0
    return 3.0


def _detect_communities(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    weight_col: str,
    min_weight: float,
) -> dict[str, int]:
    terms = sorted(nodes["pt"].tolist())
    graph = nx.Graph()
    graph.add_nodes_from(terms)
    for row in edges.itertuples(index=False):
        weight = float(getattr(row, weight_col))
        if weight < min_weight:
            continue
        graph.add_edge(row.pt1, row.pt2, weight=weight)

    communities = nx.community.louvain_communities(graph, weight="weight", seed=20260526)
    components = [sorted(component) for component in communities]

    count_lookup = nodes.set_index("pt")["case_count"].to_dict()
    components = sorted(
        components,
        key=lambda component: (sum(int(count_lookup[term]) for term in component), len(component)),
        reverse=True,
    )
    assignments: dict[str, int] = {}
    for community_id, component in enumerate(components, start=1):
        for term in component:
            assignments[term] = community_id
    return assignments


def _name_communities(nodes: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    for community_id, subset in nodes.groupby("community_id"):
        phenotype_labels = [
            label
            for label in subset["phenotype_label"].dropna().astype(str).tolist()
            if label and label != "Unmapped / other PT"
        ]
        if phenotype_labels:
            most_common = Counter(phenotype_labels).most_common(1)[0][0]
            names[int(community_id)] = most_common
            continue
        top_terms = subset.sort_values(["case_count", "pt"], ascending=[False, True])["pt"].head(3).tolist()
        names[int(community_id)] = " / ".join(top_terms)
    return names


def _attach_node_metrics(nodes: pd.DataFrame, edges: pd.DataFrame, weight_col: str) -> pd.DataFrame:
    degree = Counter()
    co_strength = Counter()
    weighted_strength = Counter()
    if not edges.empty:
        for row in edges.itertuples(index=False):
            degree[row.pt1] += 1
            degree[row.pt2] += 1
            co_strength[row.pt1] += int(row.co_count)
            co_strength[row.pt2] += int(row.co_count)
            weighted_strength[row.pt1] += float(getattr(row, weight_col))
            weighted_strength[row.pt2] += float(getattr(row, weight_col))
    out = nodes.copy()
    out["degree"] = out["pt"].map(lambda value: int(degree[value]))
    out["co_count_strength"] = out["pt"].map(lambda value: int(co_strength[value]))
    out[f"{weight_col}_strength"] = out["pt"].map(lambda value: round(float(weighted_strength[value]), 6))
    return out


def _write_metric_matrix(edges: pd.DataFrame, nodes: pd.DataFrame, metric: str, output_file: Path) -> None:
    terms = nodes["pt"].tolist()
    matrix = pd.DataFrame(0.0, index=terms, columns=terms)
    for row in edges.itertuples(index=False):
        value = float(getattr(row, metric))
        matrix.loc[row.pt1, row.pt2] = value
        matrix.loc[row.pt2, row.pt1] = value
    matrix.to_csv(output_file, encoding="utf-8-sig")


def _layout_graph(nodes: pd.DataFrame, edges: pd.DataFrame, weight_col: str) -> dict[str, tuple[float, float]]:
    terms = nodes["pt"].tolist()
    rng = random.Random(20260526)
    if not terms:
        return {}
    positions = {
        term: [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
        for term in terms
    }
    max_weight = max(edges[weight_col].max(), 1e-9) if not edges.empty else 1.0
    area = 4.0
    k = math.sqrt(area / len(terms))
    edge_rows = list(edges[["pt1", "pt2", weight_col]].itertuples(index=False, name=None))

    for iteration in range(250):
        disp = {term: [0.0, 0.0] for term in terms}
        temperature = 0.08 * (1 - iteration / 250)
        for i, v in enumerate(terms):
            for u in terms[i + 1 :]:
                dx = positions[v][0] - positions[u][0]
                dy = positions[v][1] - positions[u][1]
                dist = math.hypot(dx, dy) + 1e-6
                force = (k * k) / dist
                disp[v][0] += dx / dist * force
                disp[v][1] += dy / dist * force
                disp[u][0] -= dx / dist * force
                disp[u][1] -= dy / dist * force

        for v, u, raw_weight in edge_rows:
            dx = positions[v][0] - positions[u][0]
            dy = positions[v][1] - positions[u][1]
            dist = math.hypot(dx, dy) + 1e-6
            weight = 0.35 + float(raw_weight) / max_weight
            force = (dist * dist / k) * weight
            disp[v][0] -= dx / dist * force
            disp[v][1] -= dy / dist * force
            disp[u][0] += dx / dist * force
            disp[u][1] += dy / dist * force

        for term in terms:
            dx, dy = disp[term]
            length = math.hypot(dx, dy) + 1e-6
            positions[term][0] += dx / length * min(length, temperature)
            positions[term][1] += dy / length * min(length, temperature)

    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width, height = 1400.0, 1000.0
    pad = 90.0
    scaled: dict[str, tuple[float, float]] = {}
    for term, (x, y) in positions.items():
        sx = pad + (x - min_x) / (max_x - min_x + 1e-9) * (width - 2 * pad)
        sy = pad + (y - min_y) / (max_y - min_y + 1e-9) * (height - 2 * pad)
        scaled[term] = (sx, sy)
    return scaled


def _write_svg(nodes: pd.DataFrame, edges: pd.DataFrame, weight_col: str, output_file: Path) -> None:
    width, height = 1400, 1000
    positions = _layout_graph(nodes, edges, weight_col)
    max_count = max(nodes["case_count"].max(), 1) if not nodes.empty else 1
    max_weight = max(edges[weight_col].max(), 1e-9) if not edges.empty else 1.0
    node_lookup = nodes.set_index("pt").to_dict(orient="index")

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="50" y="48" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#1f2933">PT co-occurrence network in zolpidem-related fall reports</text>',
        '<text x="50" y="78" font-family="Arial, sans-serif" font-size="16" fill="#52606d">Node size: PT case count; edge width: association strength; color: detected community</text>',
    ]

    for row in edges.sort_values(weight_col).itertuples(index=False):
        x1, y1 = positions[row.pt1]
        x2, y2 = positions[row.pt2]
        width_px = 0.8 + 7.0 * float(getattr(row, weight_col)) / max_weight
        opacity = 0.18 + 0.45 * float(getattr(row, weight_col)) / max_weight
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#6b7280" stroke-width="{width_px:.2f}" stroke-opacity="{opacity:.3f}"/>'
        )

    for row in nodes.sort_values("case_count").itertuples(index=False):
        x, y = positions[row.pt]
        radius = 7.0 + 24.0 * math.sqrt(float(row.case_count) / max_count)
        color = COMMUNITY_COLORS[(int(row.community_id) - 1) % len(COMMUNITY_COLORS)]
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.2f}" fill="{color}" '
            f'fill-opacity="0.88" stroke="#ffffff" stroke-width="2"/>'
        )

    for row in nodes.sort_values("case_count", ascending=False).itertuples(index=False):
        x, y = positions[row.pt]
        font_size = 11 + min(9, int(8 * math.sqrt(float(row.case_count) / max_count)))
        label = escape(str(row.pt).title())
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y - 8:.1f}" font-family="Arial, sans-serif" '
            f'font-size="{font_size}" font-weight="600" fill="#111827">{label}</text>'
        )

    legend_x, legend_y = 50, 910
    for idx, (community_id, community_name) in enumerate(
        nodes[["community_id", "community_name"]].drop_duplicates().sort_values("community_id").itertuples(index=False)
    ):
        x = legend_x + (idx % 3) * 420
        y = legend_y + (idx // 3) * 28
        color = COMMUNITY_COLORS[(int(community_id) - 1) % len(COMMUNITY_COLORS)]
        parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 16}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#374151">'
            f'Community {int(community_id)}: {escape(str(community_name))}</text>'
        )

    parts.append("</svg>")
    output_file.write_text("\n".join(parts), encoding="utf-8")


def _write_networkx_figures(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    weight_col: str,
    svg_file: Path,
    png_file: Path,
    pdf_file: Path,
) -> None:
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.pt,
            case_count=int(row.case_count),
            community_id=int(row.community_id),
            community_name=str(row.community_name),
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.pt1,
            row.pt2,
            weight=float(getattr(row, weight_col)),
            co_count=int(row.co_count),
            jaccard=float(row.jaccard),
            lift=float(row.lift),
            phi=float(row.phi),
        )

    if graph.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, "No retained PT network", ha="center", va="center", fontsize=18)
        fig.savefig(svg_file, bbox_inches="tight")
        fig.savefig(png_file, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_file, bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(graph, weight="weight", seed=20260526, iterations=400, k=0.75)
    fig, ax = plt.subplots(figsize=(15.5, 11), constrained_layout=True)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.axis("off")

    max_count = max(nodes["case_count"].max(), 1)
    max_weight = max([data["weight"] for _, _, data in graph.edges(data=True)] or [1.0])
    node_sizes = [
        180 + 1850 * math.sqrt(graph.nodes[node]["case_count"] / max_count)
        for node in graph.nodes
    ]
    node_colors = [
        COMMUNITY_COLORS[(graph.nodes[node]["community_id"] - 1) % len(COMMUNITY_COLORS)]
        for node in graph.nodes
    ]
    edge_widths = [
        0.5 + 5.5 * data["weight"] / max_weight
        for _, _, data in graph.edges(data=True)
    ]
    edge_alphas = [
        0.16 + 0.46 * data["weight"] / max_weight
        for _, _, data in graph.edges(data=True)
    ]

    for alpha, width, edge in sorted(zip(edge_alphas, edge_widths, graph.edges), key=lambda item: item[0]):
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[edge],
            width=width,
            alpha=alpha,
            edge_color="#5f6b7a",
            ax=ax,
        )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#ffffff",
        linewidths=1.4,
        alpha=0.92,
        ax=ax,
    )

    labels = {node: node.title() for node in graph.nodes}
    label_sizes = {
        node: 7.5 + 5.0 * math.sqrt(graph.nodes[node]["case_count"] / max_count)
        for node in graph.nodes
    }
    for node, label in labels.items():
        x, y = pos[node]
        ax.text(
            x + 0.015,
            y + 0.015,
            label,
            fontsize=label_sizes[node],
            fontweight="semibold",
            color="#111827",
            ha="left",
            va="bottom",
        )

    ax.set_title(
        "PT co-occurrence network in zolpidem-related fall reports",
        loc="left",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.0,
        0.985,
        f"Node size = PT case count; edge width = {weight_col}; color = Louvain community",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#52606d",
        va="top",
    )

    legend_entries = (
        nodes[["community_id", "community_name"]]
        .drop_duplicates()
        .sort_values("community_id")
        .itertuples(index=False)
    )
    handles = []
    labels_for_legend = []
    for community_id, community_name in legend_entries:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COMMUNITY_COLORS[(int(community_id) - 1) % len(COMMUNITY_COLORS)],
                markersize=9,
            )
        )
        labels_for_legend.append(f"{int(community_id)}: {community_name}")
    if handles:
        ax.legend(
            handles,
            labels_for_legend,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.02),
            frameon=False,
            fontsize=8.5,
            ncol=2,
        )

    fig.savefig(svg_file, format="svg", bbox_inches="tight")
    fig.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def run(
    period_token: str,
    cohort: str,
    term_set: str,
    min_pt_case_count: int,
    min_edge_count: int,
    top_n: int,
    weight_col: str,
    community_min_weight: float | None,
    output_dir: Path,
    dataset_dir: Path = GLOBAL_DATASET_DIR,
    cleaned_output_root: Path = CLEANED_OUTPUT_ROOT,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_file = dataset_dir / f"signal_dataset_{period_token}.parquet"
    case_index_file = dataset_dir / f"global_case_index_{period_token}.parquet"
    if not signal_file.exists():
        raise FileNotFoundError(f"Signal dataset not found: {signal_file}")
    if not case_index_file.exists():
        raise FileNotFoundError(f"Case index dataset not found: {case_index_file}")

    cohort_cases = _load_signal_cohort(signal_file, cohort)
    cohort_index = _load_case_index(case_index_file, cohort_cases)
    cleaned_output_root = _resolve_cleaned_output_root(cleaned_output_root, cohort_index)
    llt_to_pt = build_meddra_llt_to_pt_map(_find_meddra_file())
    case_pt, qc = _read_reac_for_cohort(cohort_index, cleaned_output_root, llt_to_pt)
    case_pt = case_pt[["caseid", "primaryid", "pt", "reported_term"]].drop_duplicates()

    n_cases = int(cohort_cases["caseid"].nunique())
    term_counts = case_pt.groupby("pt")["caseid"].nunique().sort_values(ascending=False)
    eligible = term_counts[(term_counts >= min_pt_case_count) & (~term_counts.index.isin(EXCLUDED_NETWORK_TERMS))]
    if term_set == "all_top_pt":
        selected_terms = set(eligible.head(top_n).index)
    elif term_set == "phenotype_dictionary":
        phenotype_terms = {pt for spec in PHENOTYPE_SPECS for pt in spec.pt_terms}
        selected_terms = set(eligible[eligible.index.isin(phenotype_terms)].head(top_n).index)
    else:
        raise ValueError(f"Unsupported term set: {term_set}")

    nodes, edges = _build_edges(case_pt, selected_terms, n_cases)
    edges = edges[edges["co_count"].ge(min_edge_count)].copy()
    if weight_col not in {"co_count", "jaccard", "lift", "phi"}:
        raise ValueError(f"Unsupported weight column: {weight_col}")

    community_threshold = community_min_weight
    if community_threshold is None:
        community_threshold = _default_community_min_weight(weight_col)
    communities = _detect_communities(edges, nodes, weight_col, community_threshold)
    nodes["community_id"] = nodes["pt"].map(communities).fillna(0).astype(int)

    phenotype_lookup = _phenotype_lookup()
    nodes["phenotype_column"] = nodes["pt"].map(lambda term: phenotype_lookup.get(term, {}).get("phenotype_column", ""))
    nodes["phenotype_layer"] = nodes["pt"].map(lambda term: phenotype_lookup.get(term, {}).get("phenotype_layer", ""))
    nodes["phenotype_label"] = nodes["pt"].map(
        lambda term: phenotype_lookup.get(term, {}).get("phenotype_label", "Unmapped / other PT")
    )
    community_names = _name_communities(nodes)
    nodes["community_name"] = nodes["community_id"].map(community_names).fillna("")
    nodes = _attach_node_metrics(nodes, edges, weight_col)
    nodes = nodes.sort_values(["community_id", "case_count", "pt"], ascending=[True, False, True])

    if not edges.empty:
        edges["community_id_pt1"] = edges["pt1"].map(communities).fillna(0).astype(int)
        edges["community_id_pt2"] = edges["pt2"].map(communities).fillna(0).astype(int)
        edges["same_community"] = edges["community_id_pt1"].eq(edges["community_id_pt2"])

    cluster_rows = []
    for community_id, subset in nodes.groupby("community_id"):
        cluster_rows.append(
            {
                "community_id": int(community_id),
                "community_name": subset["community_name"].iloc[0],
                "node_count": int(len(subset)),
                "top_terms": "|".join(subset.sort_values(["case_count", "pt"], ascending=[False, True])["pt"].head(12)),
                "total_node_case_count": int(subset["case_count"].sum()),
            }
        )
    clusters = pd.DataFrame(cluster_rows).sort_values(["node_count", "total_node_case_count"], ascending=[False, False])

    prefix = f"{cohort}_{term_set}_{period_token}"
    outputs = {
        "case_pt": output_dir / f"{prefix}_case_pt.csv",
        "nodes": output_dir / f"{prefix}_nodes.csv",
        "edges": output_dir / f"{prefix}_edges.csv",
        "clusters": output_dir / f"{prefix}_clusters.csv",
        "qc": output_dir / f"{prefix}_qc.csv",
        "jaccard_matrix": output_dir / f"{prefix}_jaccard_matrix.csv",
        "lift_matrix": output_dir / f"{prefix}_lift_matrix.csv",
        "phi_matrix": output_dir / f"{prefix}_phi_matrix.csv",
        "svg": output_dir / f"{prefix}_network.svg",
        "png": output_dir / f"{prefix}_network.png",
        "pdf": output_dir / f"{prefix}_network.pdf",
        "notes": output_dir / f"{prefix}_figure_notes.md",
    }
    case_pt.sort_values(["caseid", "pt"]).to_csv(outputs["case_pt"], index=False, encoding="utf-8-sig")
    nodes.to_csv(outputs["nodes"], index=False, encoding="utf-8-sig")
    edges.to_csv(outputs["edges"], index=False, encoding="utf-8-sig")
    clusters.to_csv(outputs["clusters"], index=False, encoding="utf-8-sig")
    qc.to_csv(outputs["qc"], index=False, encoding="utf-8-sig")
    for metric in ["jaccard", "lift", "phi"]:
        _write_metric_matrix(edges, nodes, metric, outputs[f"{metric}_matrix"])
    _write_networkx_figures(nodes, edges, weight_col, outputs["svg"], outputs["png"], outputs["pdf"])

    notes = [
        "# PT co-occurrence network notes",
        "",
        f"- Period: `{period_token}`",
        f"- Cohort: `{cohort}`",
        f"- Term set: `{term_set}`",
        f"- Cohort cases: `{n_cases}`",
        f"- Node inclusion: top `{top_n}` PTs with at least `{min_pt_case_count}` cases",
        f"- Edge inclusion: co-occurrence count at least `{min_edge_count}`",
        f"- Edge metrics: co-occurrence count, Jaccard index, lift, and phi coefficient",
        f"- Figure edge width/layout weight: `{weight_col}`",
        f"- Community detection: NetworkX Louvain communities over strong retained edges with `{weight_col} >= {community_threshold}`.",
        "- Important: this is a reported-event phenotype network, not a causal mechanism graph.",
    ]
    outputs["notes"].write_text("\n".join(notes) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MedDRA PT co-occurrence network.")
    parser.add_argument("--period-token", default="2004_2025")
    parser.add_argument(
        "--cohort",
        default="zolpidem_ps_ss_strict_fall",
        choices=("zolpidem_ps_ss_strict_fall", "zolpidem_ps_ss_all", "zolpidem_ps_strict_fall"),
    )
    parser.add_argument(
        "--term-set",
        default="all_top_pt",
        choices=("all_top_pt", "phenotype_dictionary"),
        help="Use all frequent PTs or only PTs from the predefined fall-mechanism phenotype dictionary.",
    )
    parser.add_argument("--min-pt-case-count", type=int, default=5)
    parser.add_argument("--min-edge-count", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--weight-col", default="jaccard", choices=("co_count", "jaccard", "lift", "phi"))
    parser.add_argument(
        "--community-min-weight",
        type=float,
        default=None,
        help="Minimum edge weight used only for assigning communities. Defaults depend on --weight-col.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=GLOBAL_DATASET_DIR)
    parser.add_argument("--cleaned-output-root", type=Path, default=CLEANED_OUTPUT_ROOT)
    args = parser.parse_args()

    outputs = run(
        period_token=args.period_token,
        cohort=args.cohort,
        term_set=args.term_set,
        min_pt_case_count=args.min_pt_case_count,
        min_edge_count=args.min_edge_count,
        top_n=args.top_n,
        weight_col=args.weight_col,
        community_min_weight=args.community_min_weight,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        cleaned_output_root=args.cleaned_output_root,
    )
    print("PT co-occurrence network analysis completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
