from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REBUILD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REBUILD_DIR / "config.json"
DEFAULT_OUTPUT = REBUILD_DIR / "outputs"
TABLE_PREFIXES = ("DEMO", "DRUG", "REAC", "OUTC", "INDI")
ROLE_ORDER = {"PS": 4, "SS": 3, "C": 2, "I": 1}
MODERN_FAERS_FIRST_SOURCE_ORDER = 2012 * 4 + 4
COMEDICATION_TERMS = {
    "is_antidepressant": [
        "SERTRALINE", "ESCITALOPRAM", "FLUOXETINE", "CITALOPRAM", "PAROXETINE",
        "VENLAFAXINE", "DULOXETINE", "AMITRIPTYLINE", "MIRTAZAPINE",
    ],
    "is_antipsychotic": ["QUETIAPINE", "OLANZAPINE", "RISPERIDONE", "ARIPIPRAZOLE", "HALOPERIDOL"],
    "is_opioid": ["OXYCODONE", "HYDROCODONE", "MORPHINE", "FENTANYL", "TRAMADOL", "CODEINE"],
    "is_antiepileptic": ["GABAPENTIN", "PREGABALIN", "VALPROATE", "CARBAMAZEPINE", "LAMOTRIGINE"],
}
INDICATION_STEMS = {
    "indi_insomnia": ["INSOMNIA", "SLEEP"],
    "indi_anxiety": ["ANXI", "PHOBIA", "STRESS DISORDER"],
    "indi_depression": ["DEPRESS"],
    "indi_pain": ["PAIN", "NEURALGIA", "ARTHRALGIA", "MYALGIA"],
    "indi_epilepsy": ["EPILEP", "SEIZURE", "CONVULSION"],
}


@dataclass(frozen=True)
class QuarterFiles:
    year: int
    quarter: int
    token: str
    directory: Path
    files: dict[str, Path]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_from_rebuild(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REBUILD_DIR / path).resolve()


def normalize_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.upper()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def discover_quarters(raw_root: Path, first_year: int, last_year: int) -> list[QuarterFiles]:
    quarters: list[QuarterFiles] = []
    for year in range(first_year, last_year + 1):
        for quarter in range(1, 5):
            base = raw_root / str(year) / f"Q{quarter}"
            ascii_dir = next((path for path in (base / "ASCII", base / "ascii") if path.exists()), None)
            if ascii_dir is None:
                raise FileNotFoundError(f"Missing ASCII directory: {base}")
            files: dict[str, Path] = {}
            for prefix in TABLE_PREFIXES:
                candidates = [
                    path for path in ascii_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt" and path.stem.upper().startswith(prefix)
                ]
                if len(candidates) != 1:
                    raise FileNotFoundError(f"Expected one {prefix} TXT in {ascii_dir}, found {len(candidates)}")
                files[prefix] = candidates[0]
            quarters.append(QuarterFiles(year, quarter, f"{year}Q{quarter}", ascii_dir, files))
    return quarters


def header_map(path: Path) -> dict[str, str]:
    header = pd.read_csv(path, sep="$", nrows=0, encoding="latin-1")
    return {
        str(column).strip().lower().lstrip("\ufeffï»¿"): column
        for column in header.columns
    }


def raw_reader(
    path: Path,
    aliases: dict[str, tuple[str, ...]],
    *,
    chunksize: int | None = None,
) -> Iterator[pd.DataFrame]:
    columns = header_map(path)
    selected: dict[str, str] = {}
    for canonical, alternatives in aliases.items():
        actual = next((columns[name] for name in alternatives if name in columns), None)
        if actual is not None:
            selected[canonical] = actual
    required = {"primaryid"}
    if not required.issubset(selected):
        raise ValueError(f"{path} lacks required fields: {sorted(required - set(selected))}")
    reader = pd.read_csv(
        path,
        sep="$",
        usecols=list(selected.values()),
        dtype=str,
        encoding="latin-1",
        keep_default_na=False,
        low_memory=False,
        # Some legacy AERS rows contain one more trailing field than the
        # published header. Without this option pandas consumes ISR as an
        # index and shifts every requested field one column to the left.
        index_col=False,
        chunksize=chunksize,
    )
    reverse = {actual: canonical for canonical, actual in selected.items()}
    if chunksize is None:
        reader = [reader]
    for frame in reader:
        frame = frame.rename(columns=reverse)
        for canonical in aliases:
            if canonical not in frame:
                frame[canonical] = ""
        yield frame[list(aliases)]


def load_deleted_caseids(raw_root: Path, last_year: int) -> tuple[set[str], pd.DataFrame]:
    deleted: set[str] = set()
    rows: list[dict[str, Any]] = []
    for directory in raw_root.rglob("*"):
        if not directory.is_dir() or "delete" not in directory.name.lower():
            continue
        match = re.search(r"[\\/](\d{4})[\\/]Q([1-4])[\\/]", str(directory))
        if not match or int(match.group(1)) > last_year:
            continue
        for path in sorted(directory.glob("*.txt")):
            file_ids: set[str] = set()
            with path.open("r", encoding="latin-1", errors="ignore") as handle:
                for line in handle:
                    token = line.strip().split("$", 1)[0].strip()
                    if token.isdigit():
                        file_ids.add(token)
            deleted.update(file_ids)
            rows.append({"path": str(path), "n_unique_caseids": len(file_ids)})
    return deleted, pd.DataFrame(rows)


def stage_demo(quarters: list[QuarterFiles], stage_dir: Path) -> None:
    aliases = {
        "primaryid": ("primaryid", "isr"),
        "caseid": ("caseid", "case"),
        "caseversion": ("caseversion", "foll_seq"),
        "age": ("age",),
        "age_cod": ("age_cod",),
        "age_grp": ("age_grp",),
        "sex": ("sex", "gndr_cod"),
        "fda_dt": ("fda_dt",),
        "rept_cod": ("rept_cod",),
        "e_sub": ("e_sub",),
        "reporter_country": ("reporter_country",),
        "occr_country": ("occr_country",),
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    for index, quarter in enumerate(quarters, start=1):
        output = stage_dir / f"demo_{quarter.token}.parquet"
        if output.exists():
            print(f"[demo {index:02d}/{len(quarters)}] reuse {quarter.token}", flush=True)
            continue
        frame = next(raw_reader(quarter.files["DEMO"], aliases))
        frame["primaryid"] = normalize_id(frame["primaryid"])
        frame["caseid"] = normalize_id(frame["caseid"])
        frame = frame[frame["primaryid"].ne("") & frame["caseid"].ne("")].copy()
        frame["primaryid_num"] = pd.to_numeric(frame["primaryid"], errors="coerce").astype("Int64")
        frame["caseversion_num"] = pd.to_numeric(frame["caseversion"], errors="coerce").astype("Int64")
        frame["source_year"] = quarter.year
        frame["source_quarter"] = quarter.quarter
        frame["source_order"] = quarter.year * 4 + quarter.quarter
        frame["source_period"] = quarter.token
        frame.to_parquet(output, index=False, compression="zstd")
        print(f"[demo {index:02d}/{len(quarters)}] {quarter.token}: {len(frame):,}", flush=True)


def sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def copy_query(connection: duckdb.DuckDBPyConnection, query: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def build_case_bases(stage_demo_dir: Path, deleted: set[str], output_dir: Path) -> dict[str, int]:
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='20GB'")
    connection.execute("SET temp_directory='{}'".format(sql_path(output_dir / "tmp")))
    deleted_frame = pd.DataFrame({"caseid": sorted(deleted)})
    connection.register("deleted_ids", deleted_frame)
    source = sql_path(stage_demo_dir / "*.parquet")
    ranked = f"""
        WITH source AS (
            SELECT * FROM read_parquet('{source}', union_by_name=true)
        ), primary_ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY primaryid
                ORDER BY source_order DESC, caseversion_num DESC NULLS LAST,
                         caseid DESC, primaryid_num DESC NULLS LAST
            ) AS primary_rank
            FROM source
        ), primary_deduplicated AS (
            -- The same legacy ISR can be republished in a later quarterly file
            -- with a corrected CASE number. Keep the later record before
            -- selecting the newest report within CASE.
            SELECT * EXCLUDE(primary_rank)
            FROM primary_ranked
            WHERE primary_rank = 1
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY caseid
                ORDER BY
                    CASE WHEN source_order >= {MODERN_FAERS_FIRST_SOURCE_ORDER} THEN 1 ELSE 0 END DESC,
                    CASE WHEN source_order >= {MODERN_FAERS_FIRST_SOURCE_ORDER}
                         THEN caseversion_num ELSE NULL END DESC NULLS LAST,
                    CASE WHEN source_order < {MODERN_FAERS_FIRST_SOURCE_ORDER}
                         THEN primaryid_num ELSE NULL END DESC NULLS LAST,
                    source_order DESC,
                    primaryid_num DESC NULLS LAST
            ) AS version_rank
            FROM primary_deduplicated
        )
        SELECT * EXCLUDE(version_rank)
        FROM ranked
        WHERE version_rank = 1
          AND caseid NOT IN (SELECT caseid FROM deleted_ids)
    """
    latest_path = output_dir / "01_latest_nondeleted_cases.parquet"
    copy_query(connection, ranked, latest_path)
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
    elderly_query = f"""
        WITH base AS (
            SELECT *, {age_expression} AS age_years
            FROM read_parquet('{sql_path(latest_path)}')
        )
        SELECT
            caseid, primaryid, caseversion_num AS caseversion, source_year AS year,
            source_quarter AS quarter, source_period, fda_dt,
            age_years,
            CASE WHEN age_years < 75 THEN '65-74'
                 WHEN age_years < 85 THEN '75-84' ELSE '>=85' END AS age_group_3,
            upper(trim(sex)) AS sex_clean,
            rept_cod, e_sub, reporter_country, occr_country,
            CASE
                WHEN upper(trim(coalesce(nullif(reporter_country, ''), occr_country))) IN
                     ('US', 'USA', 'UNITED STATES', 'UNITED STATES OF AMERICA') THEN 'US'
                WHEN trim(coalesce(nullif(reporter_country, ''), occr_country)) = '' THEN 'unknown'
                ELSE 'non-US'
            END AS country_group,
            CASE WHEN source_year <= 2012 THEN '2004-2012'
                 WHEN source_year <= 2018 THEN '2013-2018' ELSE '2019-2025' END AS regulatory_period
        FROM base
        WHERE age_years BETWEEN 65 AND 120
    """
    elderly_path = output_dir / "02_elderly_case_base.parquet"
    copy_query(connection, elderly_query, elderly_path)
    metrics = {
        "latest_nondeleted_all_ages": connection.execute(
            f"SELECT count(*) FROM read_parquet('{sql_path(latest_path)}')"
        ).fetchone()[0],
        "elderly_cases": connection.execute(
            f"SELECT count(*) FROM read_parquet('{sql_path(elderly_path)}')"
        ).fetchone()[0],
        "elderly_unique_caseids": connection.execute(
            f"SELECT count(DISTINCT caseid) FROM read_parquet('{sql_path(elderly_path)}')"
        ).fetchone()[0],
        "elderly_unique_primaryids": connection.execute(
            f"SELECT count(DISTINCT primaryid) FROM read_parquet('{sql_path(elderly_path)}')"
        ).fetchone()[0],
    }
    connection.close()
    return metrics


def compile_drug_patterns(dictionary_path: Path) -> tuple[dict[str, re.Pattern[str]], dict[str, re.Pattern[str]]]:
    terms = pd.read_csv(dictionary_path)
    include: dict[str, re.Pattern[str]] = {}
    exclude: dict[str, re.Pattern[str]] = {}
    for action, destination in [("include", include), ("exclude", exclude)]:
        selected = terms[terms["match_action"].str.lower().eq(action)]
        for drug_key, group in selected.groupby("drug_key"):
            pieces = []
            for term in group["term"].dropna().astype(str):
                normalized = " ".join(re.findall(r"[A-Z0-9]+", term.upper()))
                if normalized:
                    pieces.append(r"\b" + r"\s+".join(map(re.escape, normalized.split())) + r"\b")
            if pieces:
                destination[drug_key] = re.compile("|".join(sorted(set(pieces), key=len, reverse=True)))
    return include, exclude


def extract_linked_tables(
    quarters: list[QuarterFiles],
    elderly_path: Path,
    stage_root: Path,
    chunksize: int,
    dictionary_path: Path,
    master_path: Path,
) -> dict[str, int]:
    base = pd.read_parquet(elderly_path, columns=["caseid", "primaryid", "source_period"])
    period_maps = {
        period: dict(zip(group["primaryid"].astype(str), group["caseid"].astype(str)))
        for period, group in base.groupby("source_period", observed=True)
    }
    include_patterns, exclude_patterns = compile_drug_patterns(dictionary_path)
    master = pd.read_csv(master_path).set_index("drug_key")
    table_specs = {
        "DRUG": {
            "primaryid": ("primaryid", "isr"), "role_cod": ("role_cod",),
            "drugname": ("drugname",), "prod_ai": ("prod_ai",), "route": ("route",),
            "dose_form": ("dose_form",),
        },
        "REAC": {"primaryid": ("primaryid", "isr"), "pt": ("pt", "reac_pt")},
        "OUTC": {"primaryid": ("primaryid", "isr"), "outc_cod": ("outc_cod",)},
        "INDI": {"primaryid": ("primaryid", "isr"), "indi_pt": ("indi_pt",)},
    }
    totals = {name.lower() + "_rows": 0 for name in table_specs}
    totals["target_match_rows"] = 0
    for table, aliases in table_specs.items():
        (stage_root / table.lower()).mkdir(parents=True, exist_ok=True)
    (stage_root / "target_matches").mkdir(parents=True, exist_ok=True)

    for index, quarter in enumerate(quarters, start=1):
        id_map = period_maps.get(quarter.token, {})
        if not id_map:
            continue
        selected_ids = set(id_map)
        for table, aliases in table_specs.items():
            output = stage_root / table.lower() / f"{table.lower()}_{quarter.token}.parquet"
            target_output = stage_root / "target_matches" / f"target_{quarter.token}.parquet"
            if output.exists() and (table != "DRUG" or target_output.exists()):
                continue
            frames: list[pd.DataFrame] = []
            target_frames: list[pd.DataFrame] = []
            for chunk in raw_reader(quarter.files[table], aliases, chunksize=chunksize):
                chunk["primaryid"] = normalize_id(chunk["primaryid"])
                chunk = chunk[chunk["primaryid"].isin(selected_ids)].copy()
                if chunk.empty:
                    continue
                chunk["caseid"] = chunk["primaryid"].map(id_map)
                chunk["source_period"] = quarter.token
                if table == "DRUG":
                    chunk["role_cod"] = chunk["role_cod"].fillna("").astype(str).str.upper().str.strip()
                    combined = normalize_text(chunk["drugname"]) + " " + normalize_text(chunk["prod_ai"])
                    route_form = normalize_text(chunk["route"]) + " " + normalize_text(chunk["dose_form"])
                    topical = route_form.str.contains(r"\b(?:TOPICAL|CREAM|OINTMENT)\b", regex=True, na=False)
                    for drug_key, pattern in include_patterns.items():
                        mask = combined.str.contains(pattern, regex=True, na=False)
                        if drug_key in exclude_patterns:
                            mask &= ~combined.str.contains(exclude_patterns[drug_key], regex=True, na=False)
                        if drug_key in {"doxepin", "mirtazapine"}:
                            mask &= ~topical
                        if mask.any():
                            match = chunk.loc[mask, ["caseid", "primaryid", "role_cod"]].copy()
                            match["drug_key"] = drug_key
                            match["drug_group"] = master.loc[drug_key, "drug_group"]
                            target_frames.append(match)
                frames.append(chunk)
            if frames:
                data = pd.concat(frames, ignore_index=True).drop_duplicates()
            else:
                data = pd.DataFrame(columns=[*aliases, "caseid", "source_period"])
            data.to_parquet(output, index=False, compression="zstd")
            totals[table.lower() + "_rows"] += len(data)
            if table == "DRUG":
                target_data = (
                    pd.concat(target_frames, ignore_index=True).drop_duplicates()
                    if target_frames else pd.DataFrame(columns=["caseid", "primaryid", "role_cod", "drug_key", "drug_group"])
                )
                target_data.to_parquet(target_output, index=False, compression="zstd")
                totals["target_match_rows"] += len(target_data)
        print(f"[links {index:02d}/{len(quarters)}] {quarter.token}", flush=True)
    return totals


def merge_stage_files(stage_root: Path, output_dir: Path) -> None:
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='20GB'")
    connection.execute("SET temp_directory='{}'".format(sql_path(output_dir / "tmp")))
    for index, table in enumerate(("drug", "reac", "outc", "indi"), start=3):
        source = stage_root / table / "*.parquet"
        output = output_dir / f"{index:02d}_{'reaction' if table == 'reac' else 'outcome' if table == 'outc' else 'indication' if table == 'indi' else 'drug'}_rows.parquet"
        copy_query(connection, f"SELECT * FROM read_parquet('{sql_path(source)}', union_by_name=true)", output)
    target_source = stage_root / "target_matches" / "*.parquet"
    copy_query(connection, f"SELECT * FROM read_parquet('{sql_path(target_source)}', union_by_name=true)", output_dir / "07_target_drug_matches.parquet")
    connection.close()


def quoted_terms(terms: list[str]) -> str:
    return ",".join("'" + term.replace("'", "''") + "'" for term in terms)


def build_case_matrices(config: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "tmp").mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='20GB'")
    connection.execute("SET temp_directory='{}'".format(sql_path(output_dir / "tmp")))
    reac_path = output_dir / "04_reaction_rows.parquet"
    drug_path = output_dir / "03_drug_rows.parquet"
    outc_path = output_dir / "05_outcome_rows.parquet"
    indi_path = output_dir / "06_indication_rows.parquet"
    elderly_path = output_dir / "02_elderly_case_base.parquet"
    target_path = output_dir / "07_target_drug_matches.parquet"

    strict = quoted_terms(config["strict_fall_terms"])
    broad = quoted_terms(config["broad_fall_terms"])
    phenotype_select = []
    for column, terms in config["phenotype_terms"].items():
        phenotype_select.append(
            f"bool_or(upper(trim(pt)) IN ({quoted_terms(terms)})) AS {column}"
        )
    reaction_query = f"""
        SELECT caseid,
               count(*)::INTEGER AS all_reac_n,
               bool_or(upper(trim(pt)) IN ({strict})) AS strict_fall,
               bool_or(upper(trim(pt)) IN ({broad})) AS broad_fall,
               sum(CASE WHEN upper(trim(pt)) IN ({strict}) THEN 1 ELSE 0 END)::INTEGER AS fall_pt_count,
               {', '.join(phenotype_select)}
        FROM read_parquet('{sql_path(reac_path)}')
        GROUP BY caseid
    """
    copy_query(connection, reaction_query, output_dir / "08_outcome_phenotype_matrix.parquet")

    outc_query = f"""
        SELECT caseid,
            bool_or(upper(trim(outc_cod))='DE') AS serious_death,
            bool_or(upper(trim(outc_cod))='LT') AS serious_life_threatening,
            bool_or(upper(trim(outc_cod))='HO') AS serious_hospitalization,
            bool_or(upper(trim(outc_cod))='DS') AS serious_disability,
            bool_or(upper(trim(outc_cod))='CA') AS serious_congenital_anomaly,
            bool_or(upper(trim(outc_cod))='RI') AS serious_required_intervention,
            bool_or(upper(trim(outc_cod))='OT') AS serious_other,
            true AS serious_any
        FROM read_parquet('{sql_path(outc_path)}') GROUP BY caseid
    """
    copy_query(connection, outc_query, output_dir / "09_serious_outcome_matrix.parquet")

    combined = "upper(trim(coalesce(nullif(prod_ai,''), drugname)))"
    drug_flags = []
    for column, terms in COMEDICATION_TERMS.items():
        expressions = [f"contains({combined}, '{term}')" for term in terms]
        drug_flags.append(f"bool_or({' OR '.join(expressions)}) AS {column}")
    drug_query = f"""
        SELECT caseid,
            count(*)::INTEGER AS drug_n,
            count(DISTINCT nullif({combined}, ''))::INTEGER AS distinct_drug_n,
            count(*) FILTER (WHERE upper(trim(role_cod))='PS')::INTEGER AS ps_drug_n,
            count(*) FILTER (WHERE upper(trim(role_cod))='SS')::INTEGER AS ss_drug_n,
            count(*) FILTER (WHERE upper(trim(role_cod))='C')::INTEGER AS concomitant_drug_n,
            count(*) FILTER (WHERE upper(trim(role_cod))='I')::INTEGER AS interacting_drug_n,
            {', '.join(drug_flags)}
        FROM read_parquet('{sql_path(drug_path)}') GROUP BY caseid
    """
    copy_query(connection, drug_query, output_dir / "10_drug_covariate_matrix.parquet")

    indication_flags = []
    for column, stems in INDICATION_STEMS.items():
        expressions = [f"contains(upper(trim(indi_pt)), '{stem}')" for stem in stems]
        indication_flags.append(f"bool_or({' OR '.join(expressions)}) AS {column}")
    indication_query = f"""
        SELECT caseid, count(*)::INTEGER AS indi_n,
               count(DISTINCT upper(trim(indi_pt)))::INTEGER AS distinct_indi_n,
               {', '.join(indication_flags)}
        FROM read_parquet('{sql_path(indi_path)}') GROUP BY caseid
    """
    copy_query(connection, indication_query, output_dir / "11_indication_covariate_matrix.parquet")

    target = pd.read_parquet(target_path)
    base_ids = pd.read_parquet(elderly_path, columns=["caseid"])
    master = pd.read_csv(resolve_from_rebuild(config["drug_master"]))
    target["role_cod"] = target["role_cod"].fillna("").astype(str).str.upper().str.strip()
    target["ps_ss"] = target["role_cod"].isin(["PS", "SS"])
    target["ps_only"] = target["role_cod"].eq("PS")
    matrix = base_ids.copy()
    for drug in master.itertuples(index=False):
        subset = target[target["drug_key"].eq(drug.drug_key)]
        matrix[f"exposure_{drug.drug_key}_ps_ss"] = matrix["caseid"].isin(
            subset.loc[subset["ps_ss"], "caseid"]
        )
        matrix[f"exposure_{drug.drug_key}_ps_only"] = matrix["caseid"].isin(
            subset.loc[subset["ps_only"], "caseid"]
        )
    for group in master["drug_group"].drop_duplicates():
        keys = master.loc[master["drug_group"].eq(group), "drug_key"]
        matrix[f"exposure_{group}_ps_ss"] = matrix[[f"exposure_{key}_ps_ss" for key in keys]].any(axis=1)
        matrix[f"exposure_{group}_ps_only"] = matrix[[f"exposure_{key}_ps_only" for key in keys]].any(axis=1)
    matrix["exposure_z_drug_ps_ss"] = matrix[
        ["exposure_zolpidem_ps_ss", "exposure_eszopiclone_ps_ss", "exposure_zaleplon_ps_ss", "exposure_zopiclone_ps_ss"]
    ].any(axis=1)
    matrix["exposure_other_z_drug_ps_ss"] = matrix[
        ["exposure_eszopiclone_ps_ss", "exposure_zaleplon_ps_ss", "exposure_zopiclone_ps_ss"]
    ].any(axis=1)
    matrix.to_parquet(output_dir / "12_target_exposure_matrix.parquet", index=False, compression="zstd")
    connection.register("target_matrix", matrix)

    main_query = f"""
        SELECT b.*,
               coalesce(r.strict_fall, false) AS strict_fall,
               coalesce(r.broad_fall, false) AS broad_fall,
               coalesce(r.fall_pt_count, 0) AS fall_pt_count,
               coalesce(r.all_reac_n, 0) AS all_reac_n,
               coalesce(r.pheno_sedation, false) AS pheno_sedation,
               coalesce(r.pheno_neurocognitive, false) AS pheno_neurocognitive,
               coalesce(r.pheno_dizziness_syncope, false) AS pheno_dizziness_syncope,
               coalesce(r.pheno_gait_balance, false) AS pheno_gait_balance,
               coalesce(r.pheno_hypotension, false) AS pheno_hypotension,
               coalesce(r.pheno_visual_disturbance, false) AS pheno_visual_disturbance,
               coalesce(o.serious_any, false) AS serious_any,
               coalesce(o.serious_death, false) AS serious_death,
               coalesce(o.serious_life_threatening, false) AS serious_life_threatening,
               coalesce(o.serious_hospitalization, false) AS serious_hospitalization,
               coalesce(o.serious_disability, false) AS serious_disability,
               coalesce(d.drug_n, 0) AS drug_n,
               coalesce(d.distinct_drug_n, 0) AS distinct_drug_n,
               coalesce(d.distinct_drug_n >= 5, false) AS polypharmacy,
               coalesce(d.distinct_drug_n >= 5, false) AS polypharmacy_5,
               coalesce(d.is_antidepressant, false) AS is_antidepressant,
               coalesce(d.is_antipsychotic, false) AS is_antipsychotic,
               coalesce(d.is_opioid, false) AS is_opioid,
               coalesce(d.is_antiepileptic, false) AS is_antiepileptic,
               coalesce(i.indi_insomnia, false) AS indi_insomnia,
               coalesce(i.indi_anxiety, false) AS indi_anxiety,
               coalesce(i.indi_depression, false) AS indi_depression,
               coalesce(i.indi_pain, false) AS indi_pain,
               coalesce(i.indi_epilepsy, false) AS indi_epilepsy,
               t.* EXCLUDE(caseid)
        FROM read_parquet('{sql_path(elderly_path)}') b
        LEFT JOIN read_parquet('{sql_path(output_dir / '08_outcome_phenotype_matrix.parquet')}') r USING(caseid)
        LEFT JOIN read_parquet('{sql_path(output_dir / '09_serious_outcome_matrix.parquet')}') o USING(caseid)
        LEFT JOIN read_parquet('{sql_path(output_dir / '10_drug_covariate_matrix.parquet')}') d USING(caseid)
        LEFT JOIN read_parquet('{sql_path(output_dir / '11_indication_covariate_matrix.parquet')}') i USING(caseid)
        LEFT JOIN target_matrix t USING(caseid)
    """
    copy_query(connection, main_query, output_dir / "13_corrected_main_dataset.parquet")
    connection.close()


def build_qc(
    config: dict[str, Any], output_dir: Path, deleted_inventory: pd.DataFrame,
    case_metrics: dict[str, int], extraction_metrics: dict[str, int],
) -> dict[str, Any]:
    main_path = output_dir / "13_corrected_main_dataset.parquet"
    main = pd.read_parquet(
        main_path,
        columns=["caseid", "primaryid", "strict_fall", "broad_fall", "year", "age_years"]
    )
    old_path = resolve_from_rebuild(config["old_case_dataset"])
    old = pd.read_parquet(old_path, columns=["caseid", "strict_fall"]) if old_path.exists() else pd.DataFrame()
    old_ids = set(normalize_id(old["caseid"])) if not old.empty else set()
    new_ids = set(main["caseid"].astype(str))
    validation = {
        "caseid_unique": bool(main["caseid"].is_unique),
        "primaryid_unique": bool(main["primaryid"].is_unique),
        "age_range_valid": bool(main["age_years"].between(65, 120).all()),
        "year_range_valid": bool(main["year"].between(int(config["first_year"]), int(config["last_year"])).all()),
    }
    report = {
        "status": "CORRECTED_FOUNDATION_COMPLETE" if all(validation.values()) else "QC_FAILED_DO_NOT_ANALYZE",
        "rules": {
            "post_2012": "highest numeric CASEVERSION; later source quarter breaks ties",
            "legacy": "largest numeric ISR/PRIMARYID; later source quarter breaks ties",
            "repeated_primaryid": "later quarterly record wins before CASE-level version selection",
            "deletions": "cumulative CASEID exclusions through the configured end year",
            "eligibility": "age 65-120 after version selection and deletion exclusion",
        },
        "counts": {
            **{key: int(value) for key, value in case_metrics.items()},
            **{key: int(value) for key, value in extraction_metrics.items()},
            "delete_files": int(len(deleted_inventory)),
            "deleted_caseids_sum_by_file": int(deleted_inventory["n_unique_caseids"].sum()) if not deleted_inventory.empty else 0,
            "corrected_main_rows": int(len(main)),
            "corrected_unique_caseids": int(main["caseid"].nunique()),
            "corrected_unique_primaryids": int(main["primaryid"].nunique()),
            "corrected_strict_fall_cases": int(main["strict_fall"].sum()),
            "corrected_broad_fall_cases": int(main["broad_fall"].sum()),
            "old_case_rows": int(len(old)),
            "old_strict_fall_cases": int(old["strict_fall"].sum()) if not old.empty else 0,
            "caseids_added_vs_old": int(len(new_ids - old_ids)),
            "caseids_removed_vs_old": int(len(old_ids - new_ids)),
        },
        "validation": validation,
        "next_step": "Recompute original ROR/regression/sensitivity analyses and rebuild all-drug temporal labels from 13_corrected_main_dataset.parquet.",
    }
    (output_dir / "qc").mkdir(exist_ok=True)
    deleted_inventory.to_csv(output_dir / "qc" / "deleted_file_inventory.csv", index=False, encoding="utf-8-sig")
    (output_dir / "qc" / "rebuild_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Corrected FAERS rebuild report", "", f"**Status: {report['status']}**", "", "## Counts", "",
    ]
    lines.extend(f"- {key}: {value:,}" for key, value in report["counts"].items())
    lines.extend(["", "## Validation", ""])
    lines.extend(f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in report["validation"].items())
    lines.extend(["", f"Next: {report['next_step']}", ""])
    (output_dir / "qc" / "rebuild_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild corrected latest-version, non-deleted FAERS case data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fresh", action="store_true", help="Remove only this rebuild's staging directory before running.")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_root = Path(config["raw_root"])
    output_dir = args.output_dir.resolve()
    stage_root = output_dir / "staging"
    if args.fresh and stage_root.exists():
        shutil.rmtree(stage_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    quarters = discover_quarters(raw_root, int(config["first_year"]), int(config["last_year"]))
    deleted, deleted_inventory = load_deleted_caseids(raw_root, int(config["last_year"]))
    stage_demo(quarters, stage_root / "demo")
    case_metrics = build_case_bases(stage_root / "demo", deleted, output_dir)
    extraction_metrics = extract_linked_tables(
        quarters,
        output_dir / "02_elderly_case_base.parquet",
        stage_root,
        int(config["chunksize"]),
        resolve_from_rebuild(config["drug_dictionary"]),
        resolve_from_rebuild(config["drug_master"]),
    )
    merge_stage_files(stage_root, output_dir)
    build_case_matrices(config, output_dir)
    report = build_qc(config, output_dir, deleted_inventory, case_metrics, extraction_metrics)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
