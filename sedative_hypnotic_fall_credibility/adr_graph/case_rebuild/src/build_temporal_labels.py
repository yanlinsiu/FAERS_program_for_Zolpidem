from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REBUILD_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = REBUILD_DIR.parents[1]
DEFAULT_CONFIG = REBUILD_DIR / "temporal_labels_config.json"
DEFAULT_OUTPUT = REBUILD_DIR / "outputs_v3" / "temporal_labels"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild import (  # noqa: E402
    MODERN_FAERS_FIRST_SOURCE_ORDER,
    QuarterFiles,
    copy_query,
    discover_quarters,
    load_config as load_rebuild_config,
    load_deleted_caseids,
    normalize_id,
    raw_reader,
    sql_path,
)


INVALID_DRUG_TERMS = {
    "", "UNKNOWN", "UNK", "UNSPECIFIED", "NOT SPECIFIED", "NOT REPORTED",
    "MULTIPLE", "OTHER", "NONE", "NA", "N A",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_drug_series(series: pd.Series) -> pd.Series:
    raw = series.fillna("").astype(str)
    # FAERS contains millions of rows but far fewer unique reported names.
    # Normalize each distinct value once, then map it back to preserve the
    # exact result while avoiding repeated Unicode work.
    unique = pd.Series(pd.unique(raw), dtype="string")
    normalized_unique = (
        unique
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    normalized_unique = normalized_unique.mask(normalized_unique.isin(INVALID_DRUG_TERMS), "")
    lookup = dict(zip(unique.astype(str), normalized_unique.astype(str)))
    return raw.map(lookup)


def stable_drug_id(canonical_key: str) -> str:
    return "drug_" + hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:16]


def build_snapshot_cases(
    demo_glob: Path,
    final_case_path: Path,
    raw_root: Path,
    cutoff_year: int,
    output_dir: Path,
) -> tuple[Path, Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "01_train_snapshot_cases.parquet"
    future_path = output_dir / "02_future_new_cases.parquet"
    cutoff_order = cutoff_year * 4 + 4
    deleted, _ = load_deleted_caseids(raw_root, cutoff_year)
    deleted_frame = pd.DataFrame({"caseid": pd.Series(sorted(deleted), dtype="string")})
    (output_dir / "tmp").mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='20GB'")
    con.execute(f"SET temp_directory='{sql_path(output_dir / 'tmp')}'")
    con.register("deleted_at_cutoff", deleted_frame)
    source = sql_path(demo_glob)
    age_expression = """
        CASE upper(trim(age_cod))
            WHEN 'YR' THEN try_cast(age AS DOUBLE)
            WHEN 'MON' THEN try_cast(age AS DOUBLE) / 12.0
            WHEN 'WK' THEN try_cast(age AS DOUBLE) / 52.1429
            WHEN 'DY' THEN try_cast(age AS DOUBLE) / 365.25
            WHEN 'HR' THEN try_cast(age AS DOUBLE) / 8766.0
            WHEN 'DEC' THEN try_cast(age AS DOUBLE) * 10.0
            ELSE NULL
        END
    """
    train_query = f"""
        WITH source AS (
            SELECT * FROM read_parquet('{source}', union_by_name=true)
            WHERE source_order <= {cutoff_order}
        ), primary_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY primaryid
                ORDER BY source_order DESC, caseversion_num DESC NULLS LAST,
                         caseid DESC, primaryid_num DESC NULLS LAST
            ) AS primary_rank
            FROM source
        ), primary_deduplicated AS (
            SELECT * EXCLUDE(primary_rank) FROM primary_ranked WHERE primary_rank=1
        ), case_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY caseid
                ORDER BY
                    CASE WHEN source_order >= {MODERN_FAERS_FIRST_SOURCE_ORDER} THEN 1 ELSE 0 END DESC,
                    CASE WHEN source_order >= {MODERN_FAERS_FIRST_SOURCE_ORDER}
                         THEN caseversion_num ELSE NULL END DESC NULLS LAST,
                    CASE WHEN source_order < {MODERN_FAERS_FIRST_SOURCE_ORDER}
                         THEN primaryid_num ELSE NULL END DESC NULLS LAST,
                    source_order DESC, primaryid_num DESC NULLS LAST
            ) AS version_rank
            FROM primary_deduplicated
        ), selected AS (
            SELECT *, {age_expression} AS age_years
            FROM case_ranked
            WHERE version_rank=1
              AND caseid NOT IN (SELECT caseid FROM deleted_at_cutoff)
        )
        SELECT caseid, primaryid, source_period, source_year, source_order, age_years
        FROM selected WHERE age_years BETWEEN 65 AND 120
    """
    copy_query(con, train_query, train_path)
    future_query = f"""
        WITH first_seen AS (
            SELECT caseid, min(source_order) AS first_source_order
            FROM read_parquet('{source}', union_by_name=true)
            GROUP BY caseid
        )
        SELECT b.*, f.first_source_order
        FROM read_parquet('{sql_path(final_case_path)}') b
        JOIN first_seen f USING(caseid)
        WHERE f.first_source_order > {cutoff_order}
    """
    copy_query(con, future_query, future_path)
    overlap = con.execute(
        f"""SELECT count(*) FROM read_parquet('{sql_path(train_path)}') t
            JOIN read_parquet('{sql_path(future_path)}') f USING(caseid)"""
    ).fetchone()[0]
    con.close()
    return train_path, future_path, int(overlap)


def extract_training_children(
    quarters: list[QuarterFiles],
    train_case_path: Path,
    stage_dir: Path,
    chunksize: int,
) -> tuple[Path, Path]:
    train = pd.read_parquet(train_case_path, columns=["caseid", "primaryid", "source_period"])
    period_maps = {
        period: dict(zip(group["primaryid"].astype(str), group["caseid"].astype(str)))
        for period, group in train.groupby("source_period", observed=True)
    }
    aliases_by_table = {
        "DRUG": {
            "primaryid": ("primaryid", "isr"), "role_cod": ("role_cod",),
            "drugname": ("drugname",), "prod_ai": ("prod_ai",),
        },
        "REAC": {"primaryid": ("primaryid", "isr"), "pt": ("pt", "reac_pt")},
    }
    for table in aliases_by_table:
        (stage_dir / table.lower()).mkdir(parents=True, exist_ok=True)
    for index, quarter in enumerate(quarters, start=1):
        id_map = period_maps.get(quarter.token, {})
        if not id_map:
            continue
        selected_ids = set(id_map)
        for table, aliases in aliases_by_table.items():
            output = stage_dir / table.lower() / f"{table.lower()}_{quarter.token}.parquet"
            if output.exists():
                continue
            frames: list[pd.DataFrame] = []
            for chunk in raw_reader(quarter.files[table], aliases, chunksize=chunksize):
                chunk["primaryid"] = normalize_id(chunk["primaryid"])
                chunk = chunk[chunk["primaryid"].isin(selected_ids)].copy()
                if chunk.empty:
                    continue
                chunk["caseid"] = chunk["primaryid"].map(id_map)
                chunk["source_period"] = quarter.token
                frames.append(chunk)
            data = (
                pd.concat(frames, ignore_index=True).drop_duplicates()
                if frames else pd.DataFrame(columns=[*aliases, "caseid", "source_period"])
            )
            data.to_parquet(output, index=False, compression="zstd")
        print(f"[snapshot children {index:02d}/{len(quarters)}] {quarter.token}", flush=True)
    del train, period_maps
    gc.collect()
    con = duckdb.connect()
    drug_path = stage_dir.parent / "03_train_drug_rows.parquet"
    reac_path = stage_dir.parent / "04_train_reaction_rows.parquet"
    copy_query(
        con,
        f"SELECT * FROM read_parquet('{sql_path(stage_dir / 'drug' / '*.parquet')}', union_by_name=true)",
        drug_path,
    )
    copy_query(
        con,
        f"SELECT * FROM read_parquet('{sql_path(stage_dir / 'reac' / '*.parquet')}', union_by_name=true)",
        reac_path,
    )
    con.close()
    return drug_path, reac_path


def filter_future_children(
    final_drug_path: Path,
    final_reac_path: Path,
    future_case_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    drug_path = output_dir / "05_future_drug_rows.parquet"
    reac_path = output_dir / "06_future_reaction_rows.parquet"
    con = duckdb.connect()
    future = sql_path(future_case_path)
    copy_query(
        con,
        f"SELECT d.* FROM read_parquet('{sql_path(final_drug_path)}') d "
        f"JOIN read_parquet('{future}') f USING(caseid)",
        drug_path,
    )
    copy_query(
        con,
        f"SELECT r.* FROM read_parquet('{sql_path(final_reac_path)}') r "
        f"JOIN read_parquet('{future}') f USING(caseid)",
        reac_path,
    )
    con.close()
    return drug_path, reac_path


def quoted_terms(terms: list[str]) -> str:
    return ",".join("'" + term.replace("'", "''") + "'" for term in terms)


def build_event_cases(
    case_path: Path,
    reaction_path: Path,
    events: list[str],
    rebuild_config: dict[str, Any],
    output_path: Path,
) -> Path:
    expressions: dict[str, str] = {
        "strict_fall": f"upper(trim(pt)) IN ({quoted_terms(rebuild_config['strict_fall_terms'])})",
        "broad_fall": f"upper(trim(pt)) IN ({quoted_terms(rebuild_config['broad_fall_terms'])})",
    }
    for event, terms in rebuild_config["phenotype_terms"].items():
        expressions[event] = f"upper(trim(pt)) IN ({quoted_terms(terms)})"
    missing = sorted(set(events) - set(expressions))
    if missing:
        raise ValueError(f"No reaction-term definition for events: {missing}")
    aggregate = ", ".join(f"bool_or({expressions[event]}) AS {event}" for event in events)
    selects = ", ".join(f"coalesce(r.{event}, false) AS {event}" for event in events)
    con = duckdb.connect()
    query = f"""
        WITH reaction AS (
            SELECT caseid, {aggregate}
            FROM read_parquet('{sql_path(reaction_path)}') GROUP BY caseid
        )
        SELECT c.caseid, c.primaryid, {selects}
        FROM read_parquet('{sql_path(case_path)}') c
        LEFT JOIN reaction r USING(caseid)
    """
    copy_query(con, query, output_path)
    con.close()
    return output_path


def build_alias_map(rows: pd.DataFrame, min_rows: int, min_share: float) -> pd.DataFrame:
    usable = rows[rows["drugname_norm"].ne("") & rows["prod_ai_norm"].ne("")]
    counts = (
        usable.groupby(["drugname_norm", "prod_ai_norm"], observed=True)
        .size().rename("row_count").reset_index()
    )
    totals = counts.groupby("drugname_norm", observed=True)["row_count"].sum().rename("total_rows")
    counts = counts.merge(totals, on="drugname_norm", how="left")
    counts["top_share"] = counts["row_count"] / counts["total_rows"]
    best = counts.sort_values(
        ["drugname_norm", "row_count", "prod_ai_norm"],
        ascending=[True, False, True], kind="mergesort",
    ).drop_duplicates("drugname_norm", keep="first")
    return best[best["total_rows"].ge(min_rows) & best["top_share"].ge(min_share)].reset_index(drop=True)


def prepare_drug_rows(path: Path, roles: set[str]) -> pd.DataFrame:
    rows = pd.read_parquet(path, columns=["caseid", "role_cod", "drugname", "prod_ai"])
    rows["caseid"] = normalize_id(rows["caseid"])
    rows["role_cod"] = rows["role_cod"].fillna("").astype(str).str.upper().str.strip()
    rows = rows[rows["role_cod"].isin(roles)].copy()
    rows["drugname_norm"] = normalize_drug_series(rows["drugname"])
    rows["prod_ai_norm"] = normalize_drug_series(rows["prod_ai"])
    return rows


def assign_canonical(rows: pd.DataFrame, alias_lookup: dict[str, str]) -> pd.DataFrame:
    learned = rows["drugname_norm"].map(alias_lookup).fillna("")
    direct = rows["prod_ai_norm"]
    resolved = direct.where(direct.ne(""), learned)
    has_ai = resolved.ne("")
    rows = rows.copy()
    rows["canonical_name"] = resolved.where(has_ai, rows["drugname_norm"])
    rows["canonical_basis"] = np.where(has_ai, "active_ingredient_resolved", "name_only")
    rows = rows[rows["canonical_name"].ne("")].copy()
    rows["canonical_key"] = np.where(
        rows["canonical_basis"].eq("active_ingredient_resolved"),
        "AI|" + rows["canonical_name"], "NAME|" + rows["canonical_name"],
    )
    rows["drug_id"] = rows["canonical_key"].map(
        {key: stable_drug_id(str(key)) for key in rows["canonical_key"].drop_duplicates()}
    )
    return rows


def aggregate_period(
    drug_rows: pd.DataFrame,
    event_path: Path,
    events: list[str],
    alias_lookup: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    rows = assign_canonical(drug_rows, alias_lookup)
    rows = rows.drop_duplicates(["caseid", "drug_id"], keep="first")
    event_data = pd.read_parquet(event_path)
    links = rows[["caseid", "drug_id"]]
    vocabulary = rows[["drug_id", "canonical_name", "canonical_basis"]].drop_duplicates("drug_id")
    matched_cases = int(links["caseid"].nunique())
    con = duckdb.connect()
    con.register("links", links)
    con.register("events", event_data)
    sums = ", ".join(f"sum(CASE WHEN e.{event} THEN 1 ELSE 0 END)::BIGINT AS {event}" for event in events)
    aggregate = con.execute(
        f"""SELECT l.drug_id, count(*)::BIGINT AS exposed_cases, {sums}
            FROM links l JOIN events e USING(caseid) GROUP BY l.drug_id"""
    ).fetchdf().merge(vocabulary, on="drug_id", how="left", validate="one_to_one")
    aggregate = aggregate[["drug_id", "canonical_name", "canonical_basis", "exposed_cases", *events]]
    con.close()
    return aggregate, matched_cases


def ror_arrays(a: Iterable[int], b: Iterable[int], c: Iterable[int], d: Iterable[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.column_stack([a, b, c, d]).astype(float)
    if np.any(cells < 0):
        raise ValueError("Negative 2x2 cell detected")
    correction = np.any(cells == 0, axis=1)
    cells[correction] += 0.5
    aa, bb, cc, dd = cells.T
    ror = (aa * dd) / (bb * cc)
    se = np.sqrt((1 / aa) + (1 / bb) + (1 / cc) + (1 / dd))
    log_ror = np.log(ror)
    return ror, np.exp(log_ror - 1.96 * se), np.exp(log_ror + 1.96 * se)


def pair_rows(
    aggregate: pd.DataFrame,
    event_data: pd.DataFrame,
    events: list[str],
    prefix: str,
) -> pd.DataFrame:
    total = len(event_data)
    frames: list[pd.DataFrame] = []
    for event in events:
        frame = aggregate[["drug_id", "canonical_name", "canonical_basis", "exposed_cases", event]].copy()
        frame = frame.rename(columns={"exposed_cases": f"{prefix}_exposed_cases", event: f"{prefix}_pair_cases"})
        frame["event"] = event
        frame[f"{prefix}_total_cases"] = total
        frame[f"{prefix}_event_cases"] = int(event_data[event].sum())
        a = frame[f"{prefix}_pair_cases"].to_numpy(dtype=int)
        b = frame[f"{prefix}_exposed_cases"].to_numpy(dtype=int) - a
        c = frame[f"{prefix}_event_cases"].to_numpy(dtype=int) - a
        d = total - a - b - c
        ror, lower, upper = ror_arrays(a, b, c, d)
        frame[f"{prefix}_ror"] = ror
        frame[f"{prefix}_ror_lower"] = lower
        frame[f"{prefix}_ror_upper"] = upper
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_labels(
    train_aggregate: pd.DataFrame,
    future_aggregate: pd.DataFrame,
    train_events: pd.DataFrame,
    future_events: pd.DataFrame,
    events: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    train = pair_rows(train_aggregate, train_events, events, "train")
    future = pair_rows(future_aggregate, future_events, events, "test")
    labels = train.merge(
        future, on=["drug_id", "canonical_name", "canonical_basis", "event"],
        how="outer", validate="one_to_one",
    )
    for prefix, event_data in [("train", train_events), ("test", future_events)]:
        integer_cols = [f"{prefix}_exposed_cases", f"{prefix}_pair_cases", f"{prefix}_total_cases", f"{prefix}_event_cases"]
        labels[integer_cols] = labels[integer_cols].fillna(0).astype("int64")
        for event in events:
            mask = labels["event"].eq(event)
            labels.loc[mask, f"{prefix}_total_cases"] = len(event_data)
            labels.loc[mask, f"{prefix}_event_cases"] = int(event_data[event].sum())
        missing = labels[f"{prefix}_ror"].isna()
        if missing.any():
            a = labels.loc[missing, f"{prefix}_pair_cases"].to_numpy(dtype=int)
            b = labels.loc[missing, f"{prefix}_exposed_cases"].to_numpy(dtype=int) - a
            c = labels.loc[missing, f"{prefix}_event_cases"].to_numpy(dtype=int) - a
            d = labels.loc[missing, f"{prefix}_total_cases"].to_numpy(dtype=int) - a - b - c
            ror, lower, upper = ror_arrays(a, b, c, d)
            labels.loc[missing, f"{prefix}_ror"] = ror
            labels.loc[missing, f"{prefix}_ror_lower"] = lower
            labels.loc[missing, f"{prefix}_ror_upper"] = upper
    labels["train_signal_positive"] = (
        labels["train_pair_cases"].ge(int(config["positive_min_cases"]))
        & labels["train_ror_lower"].gt(float(config["positive_min_ror_lower"]))
    )
    labels["test_expected_pair_cases"] = (
        labels["test_exposed_cases"] * labels["test_event_cases"] / labels["test_total_cases"].clip(lower=1)
    )
    positive = (
        labels["test_pair_cases"].ge(int(config["positive_min_cases"]))
        & labels["test_ror_lower"].gt(float(config["positive_min_ror_lower"]))
    )
    evaluable = (
        labels["test_exposed_cases"].ge(int(config["evaluable_min_exposures"]))
        & labels["test_expected_pair_cases"].ge(float(config["evaluable_min_expected_cases"]))
    )
    labels["future_label_status"] = np.select(
        [positive, evaluable & ~positive], ["positive", "operational_negative"],
        default="insufficient_information",
    )
    labels["future_label"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels.loc[positive, "future_label"] = 1
    labels.loc[evaluable & ~positive, "future_label"] = 0
    labels["cold_start_candidate"] = (
        labels["train_exposed_cases"].le(int(config["cold_start_max_train_exposures"]))
        & labels["test_exposed_cases"].ge(int(config["cold_start_min_test_exposures"]))
    )
    return labels.sort_values(["event", "canonical_name", "drug_id"], kind="mergesort").reset_index(drop=True)


def score_metrics(frame: pd.DataFrame, score_column: str) -> dict[str, float | int]:
    y = frame["future_label"].astype(int).to_numpy()
    score = frame[score_column].to_numpy(dtype=float)
    result: dict[str, float | int] = {"n_pairs": len(frame), "n_positive": int(y.sum())}
    if np.unique(y).size < 2 or np.unique(score).size < 2:
        result.update({"auprc": math.nan, "auroc": math.nan})
    else:
        result.update({"auprc": float(average_precision_score(y, score)), "auroc": float(roc_auc_score(y, score))})
    return result


def bootstrap_ap_difference(frame: pd.DataFrame, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    drug_ids = frame["drug_id"].unique()
    if len(drug_ids) == 0:
        return math.nan, math.nan
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(drug_ids, size=len(drug_ids), replace=True)
        pieces = [frame[frame["drug_id"].eq(drug_id)] for drug_id in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        y = boot["future_label"].astype(int).to_numpy()
        if np.unique(y).size < 2:
            continue
        volume = average_precision_score(y, boot["volume_score"])
        signal = average_precision_score(y, boot["historical_ror_score"])
        values.append(float(signal - volume))
    if not values:
        return math.nan, math.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def evaluate_baselines(labels: pd.DataFrame, events: list[str], config: dict[str, Any]) -> pd.DataFrame:
    work = labels[labels["future_label"].notna()].copy()
    work["volume_score"] = np.log1p(work["train_exposed_cases"].astype(float))
    work["historical_ror_score"] = np.log(work["train_ror"].clip(lower=1e-12, upper=1e12))
    rows: list[dict[str, Any]] = []
    for event in [*events, "ALL_EVENTS"]:
        frame = work if event == "ALL_EVENTS" else work[work["event"].eq(event)]
        for model, column in [("report_volume", "volume_score"), ("historical_ror", "historical_ror_score")]:
            rows.append({"event": event, "model": model, **score_metrics(frame, column)})
        low, high = bootstrap_ap_difference(
            frame, int(config["bootstrap_replicates"]), int(config["random_seed"])
        )
        rows.append({
            "event": event, "model": "historical_ror_minus_volume",
            "n_pairs": len(frame), "n_positive": int(frame["future_label"].astype(int).sum()),
            "auprc": (
                score_metrics(frame, "historical_ror_score")["auprc"]
                - score_metrics(frame, "volume_score")["auprc"]
            ),
            "auroc": math.nan, "bootstrap_95ci_low": low, "bootstrap_95ci_high": high,
        })
    return pd.DataFrame(rows)


def target_drug_summary(labels: pd.DataFrame) -> pd.DataFrame:
    master = pd.read_csv(PROJECT_DIR / "configs" / "sedative_hypnotic_drug_master.csv")
    target_names = set(normalize_drug_series(master["generic_name"]).dropna())
    strict = labels[
        labels["event"].eq("strict_fall") & labels["canonical_name"].isin(target_names)
    ].copy()
    return strict[[
        "drug_id", "canonical_name", "canonical_basis", "train_exposed_cases", "train_pair_cases",
        "test_exposed_cases", "test_pair_cases", "future_label_status", "cold_start_candidate",
    ]].sort_values("canonical_name")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled temporal all-drug FAERS labels.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_json(args.config)
    rebuild_config = load_rebuild_config(REBUILD_DIR / "config.json")
    corrected_dir = (REBUILD_DIR / config["corrected_output_dir"]).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "snapshot_2018"
    raw_root = Path(config["raw_root"])
    demo_glob = corrected_dir / "staging" / "demo" / "*.parquet"
    train_case_path, future_case_path, overlap = build_snapshot_cases(
        demo_glob, corrected_dir / "02_elderly_case_base.parquet", raw_root,
        int(config["train_end_year"]), snapshot_dir,
    )
    quarters = discover_quarters(raw_root, int(config["first_year"]), int(config["train_end_year"]))
    train_drug_path, train_reac_path = extract_training_children(
        quarters, train_case_path, snapshot_dir / "staging", int(config["chunksize"]),
    )
    future_drug_path, future_reac_path = filter_future_children(
        corrected_dir / "03_drug_rows.parquet", corrected_dir / "04_reaction_rows.parquet",
        future_case_path, snapshot_dir,
    )
    events = list(config["events"])
    train_event_path = build_event_cases(
        train_case_path, train_reac_path, events, rebuild_config,
        snapshot_dir / "07_train_case_events.parquet",
    )
    future_event_path = build_event_cases(
        future_case_path, future_reac_path, events, rebuild_config,
        snapshot_dir / "08_future_case_events.parquet",
    )
    roles = {str(role).upper() for role in config["suspect_roles"]}
    train_drugs = prepare_drug_rows(train_drug_path, roles)
    alias_map = build_alias_map(
        train_drugs, int(config["alias_min_rows"]), float(config["alias_min_top_share"])
    )
    alias_lookup = dict(zip(alias_map["drugname_norm"], alias_map["prod_ai_norm"]))
    train_aggregate, train_matched = aggregate_period(train_drugs, train_event_path, events, alias_lookup)
    del train_drugs
    gc.collect()
    future_drugs = prepare_drug_rows(future_drug_path, roles)
    future_aggregate, future_matched = aggregate_period(future_drugs, future_event_path, events, alias_lookup)
    del future_drugs
    gc.collect()
    train_events = pd.read_parquet(train_event_path)
    future_events = pd.read_parquet(future_event_path)
    labels = build_labels(train_aggregate, future_aggregate, train_events, future_events, events, config)
    vocabulary = train_aggregate[["drug_id", "canonical_name", "canonical_basis", "exposed_cases"]].rename(
        columns={"exposed_cases": "train_exposed_cases"}
    ).merge(
        future_aggregate[["drug_id", "canonical_name", "canonical_basis", "exposed_cases"]].rename(
            columns={"exposed_cases": "test_exposed_cases"}
        ), on=["drug_id", "canonical_name", "canonical_basis"], how="outer", validate="one_to_one",
    )
    vocabulary[["train_exposed_cases", "test_exposed_cases"]] = vocabulary[
        ["train_exposed_cases", "test_exposed_cases"]
    ].fillna(0).astype("int64")
    metrics = evaluate_baselines(labels, events, config)
    targets = target_drug_summary(labels)
    status_counts = labels.groupby(["event", "future_label_status"], observed=True).size().rename("n_pairs").reset_index()
    train_source_max = pd.read_parquet(train_case_path, columns=["source_order"])["source_order"].max()
    future_first_min = pd.read_parquet(future_case_path, columns=["first_source_order"])["first_source_order"].min()
    strict_gain_row = metrics[
        metrics["event"].eq("strict_fall") & metrics["model"].eq("historical_ror_minus_volume")
    ].iloc[0]
    report = {
        "status": "TEMPORAL_LABELS_COMPLETE",
        "time_design": {
            "train_snapshot": "latest report version actually available by 2018Q4",
            "future_cohort": "cases first entering FAERS in 2019Q1-2025Q4, latest nondeleted version through 2025Q4",
            "model_fitting": "none; prespecified historical report volume and historical ROR are direct ranking scores",
        },
        "counts": {
            "train_elderly_cases": int(len(train_events)),
            "future_new_elderly_cases": int(len(future_events)),
            "train_future_case_overlap": overlap,
            "train_cases_with_ps_ss_drug": train_matched,
            "future_cases_with_ps_ss_drug": future_matched,
            "canonical_drugs": int(vocabulary["drug_id"].nunique()),
            "drug_event_pairs": int(len(labels)),
            "evaluable_pairs": int(labels["future_label"].notna().sum()),
            "future_positive_pairs": int(labels["future_label"].eq(1).sum()),
            "operational_negative_pairs": int(labels["future_label"].eq(0).sum()),
            "cold_start_pairs": int(labels["cold_start_candidate"].sum()),
        },
        "validation": {
            "no_case_overlap": overlap == 0,
            "train_source_not_after_cutoff": int(train_source_max) <= int(config["train_end_year"]) * 4 + 4,
            "future_cases_first_seen_after_cutoff": int(future_first_min) > int(config["train_end_year"]) * 4 + 4,
            "train_caseid_unique": bool(train_events["caseid"].is_unique),
            "future_caseid_unique": bool(future_events["caseid"].is_unique),
        },
        "strict_fall_baseline": {
            "historical_ror_minus_volume_auprc": float(strict_gain_row["auprc"]),
            "bootstrap_95ci_low": float(strict_gain_row["bootstrap_95ci_low"]),
            "bootstrap_95ci_high": float(strict_gain_row["bootstrap_95ci_high"]),
        },
        "interpretation": (
            "future_label=0 is an operational non-signal under exposure/expected-count thresholds, "
            "not proof that the adverse reaction is biologically absent. Drug identity remains a "
            "FAERS name/active-ingredient normalization and still requires PubChem/ChEMBL parent mapping."
        ),
    }
    labels.to_parquet(output_dir / "all_drug_event_labels.parquet", index=False, compression="zstd")
    vocabulary.to_parquet(output_dir / "drug_vocabulary.parquet", index=False, compression="zstd")
    vocabulary.to_csv(output_dir / "drug_vocabulary.csv", index=False, encoding="utf-8-sig")
    alias_map.to_csv(output_dir / "training_name_to_ingredient_map.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "temporal_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output_dir / "target_drug_strict_fall_summary.csv", index=False, encoding="utf-8-sig")
    status_counts.to_csv(output_dir / "label_status_by_event.csv", index=False, encoding="utf-8-sig")
    (output_dir / "temporal_label_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
