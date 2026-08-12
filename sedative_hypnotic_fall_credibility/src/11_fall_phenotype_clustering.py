from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_DATASET = PROJECT_DIR / "outputs" / "intermediate" / "05_main_analysis_dataset.parquet"
DEFAULT_CLUSTERS_OUT = PROJECT_DIR / "outputs" / "intermediate" / "11_fall_phenotype_clusters.parquet"
DEFAULT_MODEL_SELECTION_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s11_cluster_model_selection.csv"
DEFAULT_SUMMARY_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s12_fall_phenotype_cluster_summary.csv"
DEFAULT_DISTRIBUTION_OUT = PROJECT_DIR / "outputs" / "tables" / "table_s13_drug_group_by_cluster_distribution.csv"
DEFAULT_HEATMAP_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_s11_cluster_profile_heatmap.png"
DEFAULT_SCATTER_OUT = PROJECT_DIR / "outputs" / "figures" / "figure_s12_pca_cluster_scatter.png"
DEFAULT_QC_OUT = PROJECT_DIR / "outputs" / "qc" / "11_fall_phenotype_clustering_qc.csv"

RANDOM_STATE = 20260608
K_CANDIDATES = [3, 4, 5, 6]
DEFAULT_K = 4
MIN_CLUSTER_N = 50

EXPOSURE_COLUMNS = [
    "exposure_zolpidem_ps_ss",
    "exposure_other_z_drug_ps_ss",
    "exposure_z_drug_ps_ss",
    "exposure_benzodiazepine_ps_ss",
    "exposure_orexin_antagonist_ps_ss",
    "exposure_other_insomnia_related_ps_ss",
]

DRUG_GROUPS = [
    ("zolpidem", "Zolpidem", "exposure_zolpidem_ps_ss"),
    ("other_z_drugs", "Other Z-drugs", "exposure_other_z_drug_ps_ss"),
    ("benzodiazepines", "Benzodiazepines", "exposure_benzodiazepine_ps_ss"),
    ("orexin_antagonists", "Orexin receptor antagonists", "exposure_orexin_antagonist_ps_ss"),
    ("other_insomnia_related", "Other insomnia-related drugs", "exposure_other_insomnia_related_ps_ss"),
]

PHENOTYPE_FEATURES = [
    ("pheno_sedation", "Sedation/somnolence"),
    ("pheno_neurocognitive", "Neurocognitive"),
    ("pheno_dizziness_syncope", "Dizziness/syncope"),
    ("pheno_gait_balance", "Gait/balance"),
    ("pheno_hypotension", "Hypotension"),
    ("pheno_visual_disturbance", "Visual disturbance"),
]

SERIOUS_FEATURES = [
    ("serious_hospitalization", "Hospitalization"),
    ("serious_death", "Death"),
    ("serious_disability", "Disability"),
    ("serious_life_threatening", "Life-threatening"),
]

MEDICATION_COMPLEXITY_FEATURES = [
    ("polypharmacy_5", "Polypharmacy >=5 drugs"),
    ("is_antidepressant", "Concomitant antidepressant"),
    ("is_antipsychotic", "Concomitant antipsychotic"),
    ("is_opioid", "Concomitant opioid"),
    ("is_antiepileptic", "Concomitant antiepileptic"),
]

CLUSTER_FEATURES = [column for column, _ in PHENOTYPE_FEATURES + SERIOUS_FEATURES + MEDICATION_COMPLEXITY_FEATURES]

BASE_COLUMNS = [
    "caseid",
    "analysis_eligible_main",
    "strict_fall",
    "year",
    "age_group_3",
    "sex_clean",
    "country_group",
    "rept_cod",
    "e_sub",
    "n_sedative_hypnotic_drugs_ps_ss",
    "n_sedative_hypnotic_groups_ps_ss",
    "mixed_sedative_hypnotic_group_ps_ss",
    *EXPOSURE_COLUMNS,
    *CLUSTER_FEATURES,
]


@dataclass(frozen=True)
class ClusterLabel:
    label: str
    rationale: str


def safe_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def percent(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator * 100 if denominator else np.nan


def read_main_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Main analysis dataset not found: {path}")
    available = pq.ParquetFile(path).schema.names
    missing = [column for column in BASE_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"Main analysis dataset is missing required clustering columns: {missing}")
    return pd.read_parquet(path, columns=BASE_COLUMNS)


def prepare_analysis_dataset(df: pd.DataFrame) -> pd.DataFrame:
    eligible = safe_bool(df["analysis_eligible_main"])
    strict_fall = safe_bool(df["strict_fall"])
    sedative_exposed = pd.Series(False, index=df.index)
    for column in EXPOSURE_COLUMNS:
        sedative_exposed = sedative_exposed | safe_bool(df[column])

    analysis_df = df.loc[eligible & strict_fall & sedative_exposed].copy()
    if analysis_df.empty:
        raise ValueError("No sedative-hypnotic exposed strict fall reports were available for clustering.")

    for column in CLUSTER_FEATURES + EXPOSURE_COLUMNS + ["mixed_sedative_hypnotic_group_ps_ss"]:
        analysis_df[column] = safe_bool(analysis_df[column])
    analysis_df["feature_positive_count"] = analysis_df[CLUSTER_FEATURES].sum(axis=1).astype("int16")
    analysis_df["phenotype_positive_count"] = analysis_df[[column for column, _ in PHENOTYPE_FEATURES]].sum(axis=1).astype("int8")
    analysis_df["serious_component_count"] = analysis_df[[column for column, _ in SERIOUS_FEATURES]].sum(axis=1).astype("int8")
    analysis_df["cns_comedication_marker"] = (
        safe_bool(analysis_df["is_antidepressant"])
        | safe_bool(analysis_df["is_antipsychotic"])
        | safe_bool(analysis_df["is_opioid"])
        | safe_bool(analysis_df["is_antiepileptic"])
    )
    analysis_df["drug_group_count"] = analysis_df[[column for _, _, column in DRUG_GROUPS]].sum(axis=1).astype("int8")
    return analysis_df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[CLUSTER_FEATURES].astype(int).to_numpy(dtype=float)


def fit_kmeans_models(x: np.ndarray, k_candidates: list[int]) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, object]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_candidates:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(x)
        labels_by_k[k] = labels
        counts = pd.Series(labels).value_counts()
        if len(counts) > 1:
            sample_size = min(10000, len(x))
            silhouette = silhouette_score(x, labels, sample_size=sample_size, random_state=RANDOM_STATE)
        else:
            silhouette = np.nan
        rows.append(
            {
                "method": "kmeans",
                "k": k,
                "n_reports": len(x),
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
                "min_cluster_n": int(counts.min()),
                "max_cluster_n": int(counts.max()),
                "small_cluster_flag": bool(counts.min() < MIN_CLUSTER_N),
                "selected": False,
                "selection_note": "",
            }
        )
    return pd.DataFrame(rows), labels_by_k


def select_k(model_selection: pd.DataFrame) -> int:
    default_row = model_selection[model_selection["k"].eq(DEFAULT_K)].iloc[0]
    if not bool(default_row["small_cluster_flag"]):
        return DEFAULT_K

    candidates = model_selection[~model_selection["small_cluster_flag"]].copy()
    if candidates.empty:
        return int(model_selection.sort_values(["min_cluster_n", "silhouette_score"], ascending=[False, False]).iloc[0]["k"])
    return int(candidates.sort_values("silhouette_score", ascending=False).iloc[0]["k"])


def agglomerative_stability(x: np.ndarray, k: int, kmeans_labels: np.ndarray) -> tuple[float, str]:
    if len(x) > 12000:
        rng = np.random.default_rng(RANDOM_STATE)
        indices = np.sort(rng.choice(len(x), size=12000, replace=False))
        x_fit = x[indices]
        reference = kmeans_labels[indices]
        note = "Agglomerative stability was estimated on a deterministic 12,000-report sample."
    else:
        x_fit = x
        reference = kmeans_labels
        note = "Agglomerative stability was estimated on the full clustering set."

    labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x_fit)
    return adjusted_rand_score(reference, labels), note


def mark_selected_model(model_selection: pd.DataFrame, selected_k: int, agglomerative_ari: float, stability_note: str) -> pd.DataFrame:
    result = model_selection.copy()
    selected = result["k"].eq(selected_k)
    result.loc[selected, "selected"] = True
    result.loc[selected, "selection_note"] = (
        f"Selected k={selected_k}; default k=4 is preferred unless a small cluster appears. "
        f"Agglomerative adjusted Rand index={agglomerative_ari:.3f}. {stability_note}"
    )
    result["agglomerative_adjusted_rand_index_for_selected_k"] = np.nan
    result.loc[selected, "agglomerative_adjusted_rand_index_for_selected_k"] = agglomerative_ari
    return result


def pca_coordinates(x: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(x)
    return coords, pca


def high_features(row: pd.Series, features: list[tuple[str, str]], threshold: float = 20.0) -> list[str]:
    return [label for column, label in features if float(row[f"{column}_percent"]) >= threshold]


def label_cluster(row: pd.Series) -> ClusterLabel:
    phenotype_rates = {column: float(row[f"{column}_percent"]) for column, _ in PHENOTYPE_FEATURES}
    top_column = max(phenotype_rates, key=phenotype_rates.get)
    top_rate = phenotype_rates[top_column]
    high_phenotypes = high_features(row, PHENOTYPE_FEATURES, threshold=20.0)
    serious_high = (
        float(row["serious_hospitalization_percent"]) >= 70.0
        or float(row["serious_death_percent"]) >= 20.0
        or float(row["serious_life_threatening_percent"]) >= 10.0
    )
    high_polypharmacy = float(row["polypharmacy_5_percent"]) >= 80.0

    if top_rate < 10.0 and serious_high:
        return ClusterLabel(
            "low-phenotype serious-outcome",
            "Mechanistic co-reported phenotypes were sparse, but serious outcomes were frequent.",
        )
    if len(high_phenotypes) >= 2:
        return ClusterLabel(
            "mixed-mechanistic",
            "Two or more mechanistic phenotype modules were common in this cluster.",
        )
    if top_column == "pheno_neurocognitive":
        if high_polypharmacy:
            return ClusterLabel(
                "polypharmacy-associated neurocognitive",
                "Neurocognitive co-reported phenotypes were common, and polypharmacy was highly prevalent.",
            )
        return ClusterLabel(
            "neurocognitive-dominant",
            "Neurocognitive co-reported phenotypes were the leading mechanistic feature.",
        )
    if top_column in {"pheno_dizziness_syncope", "pheno_hypotension"}:
        return ClusterLabel(
            "dizziness-syncope-hypotension",
            "Dizziness, syncope, or hypotension was the leading mechanistic feature.",
        )
    if top_column == "pheno_gait_balance":
        return ClusterLabel(
            "gait-balance",
            "Gait or balance co-reported phenotypes were the leading mechanistic feature.",
        )
    if top_column == "pheno_sedation":
        return ClusterLabel(
            "sedation-dominant",
            "Sedation or somnolence was the leading mechanistic feature.",
        )
    return ClusterLabel(
        "weakly characterized",
        "No mechanistic phenotype module clearly dominated this cluster.",
    )


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(df)
    for cluster_id in sorted(df["phenotype_cluster"].unique()):
        cluster_df = df[df["phenotype_cluster"].eq(cluster_id)]
        row: dict[str, object] = {
            "phenotype_cluster": int(cluster_id),
            "cluster_n": len(cluster_df),
            "cluster_percent": percent(len(cluster_df), total),
            "feature_positive_count_mean": float(cluster_df["feature_positive_count"].mean()),
            "phenotype_positive_count_mean": float(cluster_df["phenotype_positive_count"].mean()),
            "serious_component_count_mean": float(cluster_df["serious_component_count"].mean()),
            "mixed_sedative_hypnotic_group_percent": percent(int(cluster_df["mixed_sedative_hypnotic_group_ps_ss"].sum()), len(cluster_df)),
            "cns_comedication_marker_percent": percent(int(cluster_df["cns_comedication_marker"].sum()), len(cluster_df)),
        }
        for column, _ in PHENOTYPE_FEATURES + SERIOUS_FEATURES + MEDICATION_COMPLEXITY_FEATURES:
            n = int(cluster_df[column].sum())
            row[f"{column}_n"] = n
            row[f"{column}_percent"] = percent(n, len(cluster_df))
        label = label_cluster(pd.Series(row))
        row["cluster_label"] = label.label
        row["cluster_rationale"] = label.rationale
        rows.append(row)
    return pd.DataFrame(rows).sort_values("phenotype_cluster")


def add_labels_to_cases(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    label_map = summary.set_index("phenotype_cluster")["cluster_label"].to_dict()
    result = df.copy()
    result["phenotype_cluster_label"] = result["phenotype_cluster"].map(label_map)
    keep_columns = [
        "caseid",
        "phenotype_cluster",
        "phenotype_cluster_label",
        "pca_1",
        "pca_2",
        "year",
        "age_group_3",
        "sex_clean",
        "country_group",
        "rept_cod",
        "e_sub",
        "feature_positive_count",
        "phenotype_positive_count",
        "serious_component_count",
        "cns_comedication_marker",
        "drug_group_count",
        "n_sedative_hypnotic_drugs_ps_ss",
        "n_sedative_hypnotic_groups_ps_ss",
        "mixed_sedative_hypnotic_group_ps_ss",
        *EXPOSURE_COLUMNS,
        *CLUSTER_FEATURES,
    ]
    return result[keep_columns].copy()


def build_drug_group_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_key, group_label, exposure_column in DRUG_GROUPS:
        group_mask = safe_bool(df[exposure_column])
        group_total = int(group_mask.sum())
        for cluster_id in sorted(df["phenotype_cluster"].unique()):
            cluster_mask = df["phenotype_cluster"].eq(cluster_id)
            n = int((group_mask & cluster_mask).sum())
            rows.append(
                {
                    "drug_group": group_key,
                    "drug_group_label": group_label,
                    "phenotype_cluster": int(cluster_id),
                    "phenotype_cluster_label": str(df.loc[cluster_mask, "phenotype_cluster_label"].iloc[0]),
                    "drug_group_fall_n": group_total,
                    "cluster_n": n,
                    "cluster_percent_within_drug_group": percent(n, group_total),
                }
            )
    return pd.DataFrame(rows)


def build_qc(
    source_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    pca: PCA,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = safe_bool(source_df["analysis_eligible_main"])
    strict_fall = safe_bool(source_df["strict_fall"])
    sedative_exposed = pd.Series(False, index=source_df.index)
    for column in EXPOSURE_COLUMNS:
        sedative_exposed = sedative_exposed | safe_bool(source_df[column])

    def add(metric: str, value: object, note: str = "") -> None:
        rows.append({"qc_domain": "fall_phenotype_clustering", "metric": metric, "value": value, "note": note})

    add("input_rows", len(source_df))
    add("analysis_eligible_strict_fall_rows", int((eligible & strict_fall).sum()))
    add("sedative_hypnotic_exposed_strict_fall_rows", int((eligible & strict_fall & sedative_exposed).sum()))
    add("clustering_rows", len(analysis_df))
    add("duplicate_caseid", int(analysis_df["caseid"].duplicated().sum()))
    add("selected_k", int(model_selection.loc[model_selection["selected"], "k"].iloc[0]))
    add("pca_explained_variance_pc1", float(pca.explained_variance_ratio_[0]))
    add("pca_explained_variance_pc2", float(pca.explained_variance_ratio_[1]))
    add("small_cluster_threshold", MIN_CLUSTER_N)
    add(
        "interpretation_limit",
        "exploratory_report_pattern_only",
        "Clusters describe FAERS co-reported phenotype patterns and should not be interpreted as clinical subtypes or causal mechanisms.",
    )
    for _, row in summary.iterrows():
        cluster = int(row["phenotype_cluster"])
        add(f"cluster_{cluster}_n", int(row["cluster_n"]), str(row["cluster_label"]))
        add(f"cluster_{cluster}_small_cluster_flag", bool(int(row["cluster_n"]) < MIN_CLUSTER_N), str(row["cluster_label"]))
    return pd.DataFrame(rows)


def plot_cluster_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    plot_columns = [column for column, _ in PHENOTYPE_FEATURES + SERIOUS_FEATURES + MEDICATION_COMPLEXITY_FEATURES]
    plot_labels = [label for _, label in PHENOTYPE_FEATURES + SERIOUS_FEATURES + MEDICATION_COMPLEXITY_FEATURES]
    plot_df = summary.set_index("cluster_label")[[f"{column}_percent" for column in plot_columns]]
    plot_df.columns = plot_labels

    fig, ax = plt.subplots(figsize=(12, max(4.5, len(plot_df) * 1.0)))
    image = ax.imshow(plot_df.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns, rotation=35, ha="right")
    ax.set_yticks(range(plot_df.shape[0]))
    ax.set_yticklabels([f"{idx}\n(n={int(summary.iloc[i]['cluster_n'])})" for i, idx in enumerate(plot_df.index)])
    ax.set_title("Unsupervised phenotype clusters among sedative-hypnotic strict fall reports")
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = plot_df.iloc[i, j]
            text_color = "white" if value >= 55 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Percent within cluster")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_pca_scatter(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.copy()
    if len(plot_df) > 8000:
        plot_df = plot_df.sample(n=8000, random_state=RANDOM_STATE)

    labels = sorted(plot_df["phenotype_cluster_label"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    color_map = {label: colors[i] for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for label in labels:
        subset = plot_df[plot_df["phenotype_cluster_label"].eq(label)]
        ax.scatter(subset["pca_1"], subset["pca_2"], s=15, alpha=0.58, label=label, color=color_map[label], linewidths=0)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("PCA view of phenotype cluster assignments")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def validate_outputs(cases: pd.DataFrame, summary: pd.DataFrame, distribution: pd.DataFrame) -> None:
    if cases.empty or summary.empty or distribution.empty:
        raise ValueError("Clustering outputs must not be empty.")
    if cases["caseid"].duplicated().any():
        raise ValueError("Cluster case-label output contains duplicate caseid values.")
    if int(summary["cluster_n"].sum()) != len(cases):
        raise ValueError("Cluster summary counts do not match case-label rows.")
    required_groups = {"zolpidem", "other_z_drugs", "benzodiazepines"}
    missing_groups = required_groups - set(distribution["drug_group"])
    if missing_groups:
        raise ValueError(f"Drug group distribution is missing required groups: {sorted(missing_groups)}")
    finite_columns = ["pca_1", "pca_2"]
    if not np.isfinite(cases[finite_columns].to_numpy(dtype=float)).all():
        raise ValueError("PCA coordinates contain non-finite values.")


def write_outputs(
    cases: pd.DataFrame,
    model_selection: pd.DataFrame,
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    qc: pd.DataFrame,
    clusters_out: Path,
    model_selection_out: Path,
    summary_out: Path,
    distribution_out: Path,
    qc_out: Path,
) -> None:
    for path in [clusters_out, model_selection_out, summary_out, distribution_out, qc_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    cases.to_parquet(clusters_out, index=False)
    model_selection.to_csv(model_selection_out, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_out, index=False, encoding="utf-8-sig")
    distribution.to_csv(distribution_out, index=False, encoding="utf-8-sig")
    qc.to_csv(qc_out, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dataset", type=Path, default=DEFAULT_MAIN_DATASET)
    parser.add_argument("--clusters-out", type=Path, default=DEFAULT_CLUSTERS_OUT)
    parser.add_argument("--model-selection-out", type=Path, default=DEFAULT_MODEL_SELECTION_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--distribution-out", type=Path, default=DEFAULT_DISTRIBUTION_OUT)
    parser.add_argument("--heatmap-out", type=Path, default=DEFAULT_HEATMAP_OUT)
    parser.add_argument("--scatter-out", type=Path, default=DEFAULT_SCATTER_OUT)
    parser.add_argument("--qc-out", type=Path, default=DEFAULT_QC_OUT)
    args = parser.parse_args()

    source_df = read_main_dataset(args.main_dataset)
    analysis_df = prepare_analysis_dataset(source_df)
    x = feature_matrix(analysis_df)

    model_selection, labels_by_k = fit_kmeans_models(x, K_CANDIDATES)
    selected_k = select_k(model_selection)
    selected_labels = labels_by_k[selected_k]
    agglomerative_ari, stability_note = agglomerative_stability(x, selected_k, selected_labels)
    model_selection = mark_selected_model(model_selection, selected_k, agglomerative_ari, stability_note)

    coords, pca = pca_coordinates(x)
    analysis_df = analysis_df.copy()
    analysis_df["phenotype_cluster"] = selected_labels.astype(int) + 1
    analysis_df["pca_1"] = coords[:, 0]
    analysis_df["pca_2"] = coords[:, 1]

    summary = build_cluster_summary(analysis_df)
    cases = add_labels_to_cases(analysis_df, summary)
    distribution = build_drug_group_distribution(cases)
    qc = build_qc(source_df, analysis_df, summary, model_selection, pca)

    validate_outputs(cases, summary, distribution)
    write_outputs(
        cases,
        model_selection,
        summary,
        distribution,
        qc,
        args.clusters_out,
        args.model_selection_out,
        args.summary_out,
        args.distribution_out,
        args.qc_out,
    )
    plot_cluster_heatmap(summary, args.heatmap_out)
    plot_pca_scatter(cases, args.scatter_out)

    print(f"Wrote {args.clusters_out}")
    print(f"Wrote {args.model_selection_out}")
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.distribution_out}")
    print(f"Wrote {args.heatmap_out}")
    print(f"Wrote {args.scatter_out}")
    print(f"Wrote {args.qc_out}")
    print(f"Clustering rows: {len(cases):,}")
    print(f"Selected k: {selected_k}")
    for _, row in summary.iterrows():
        print(f"Cluster {int(row['phenotype_cluster'])}: {row['cluster_label']} ({int(row['cluster_n']):,})")


if __name__ == "__main__":
    main()
