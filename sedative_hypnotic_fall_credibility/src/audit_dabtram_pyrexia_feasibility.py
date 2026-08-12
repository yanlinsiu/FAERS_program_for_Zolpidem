"""Strict local FAERS feasibility audit for dabrafenib + trametinib pyrexia.

The script intentionally stops short of disproportionality testing.  It answers
the study-readiness questions first:

* How many latest-version, non-deleted CASEIDs contain both drugs?
* How many of those cases contain the core MedDRA PT ``Pyrexia``?
* Do pyrexia cases contain enough non-fever co-reported PTs for phenotype work?
* Are clinically interesting companion symptom groups actually represented?

Raw FAERS ASCII is used because the existing project parquets are restricted to
older adults with usable age/sex data.  The unit of analysis here is one retained
FAERS CASEID across all ages.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(r"D:\program_FAERS\data")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "outputs" / "dabtram_pyrexia_feasibility"
)

TARGET_PATTERNS = {
    "dabrafenib": re.compile(r"\b(?:dabrafenib|tafinlar)\b", re.IGNORECASE),
    "trametinib": re.compile(r"\b(?:trametinib|mekinist)\b", re.IGNORECASE),
}
ROLE_PS_SS = {"PS", "SS"}

CORE_FEVER_PTS = {"PYREXIA"}
EXTENDED_FEVER_PTS = {
    "PYREXIA",
    "HYPERPYREXIA",
    "HYPERTHERMIA",
    "BODY TEMPERATURE INCREASED",
    "FEVER",
    "FEBRILE",
}

# These are descriptive phenotype buckets, not biological mechanism labels.
PHENOTYPE_GROUPS = {
    "systemic_inflammatory_like": {
        "CHILLS",
        "RIGORS",
        "NIGHT SWEATS",
        "MYALGIA",
        "FATIGUE",
        "ASTHENIA",
        "MALAISE",
        "INFLUENZA LIKE ILLNESS",
    },
    "skin_joint_immune_like": {
        "RASH",
        "RASH MACULO-PAPULAR",
        "RASH GENERALIZED",
        "ERYTHEMA",
        "ARTHRALGIA",
        "ARTHRITIS",
        "PANNICULITIS",
    },
    "gastrointestinal_volume_loss": {
        "NAUSEA",
        "VOMITING",
        "DIARRHOEA",
        "DEHYDRATION",
        "DECREASED APPETITE",
    },
    "haemodynamic_organ_effects": {
        "HYPOTENSION",
        "SYNCOPE",
        "ACUTE KIDNEY INJURY",
        "RENAL IMPAIRMENT",
        "CONFUSIONAL STATE",
        "DIZZINESS",
    },
}


@dataclass(frozen=True)
class FaersFile:
    role: str
    year: int
    quarter: int
    path: Path

    @property
    def period(self) -> str:
        return f"{self.year}Q{self.quarter}"


def normalize_id(series: pd.Series) -> pd.Series:
    """Normalize FAERS identifiers without converting them to floating point."""
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalize_pt(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def discover_files(
    data_root: Path,
    role: str,
    start_year: int,
    end_year: int,
) -> list[FaersFile]:
    # 2018Q1 is distributed locally as DEMO18Q1_new.txt; allow a descriptive
    # suffix while still enforcing one physical file per role and quarter.
    pattern = re.compile(
        rf"^{re.escape(role)}(?P<yy>\d{{2}})q(?P<q>[1-4])(?:_[^.]+)?\.txt$",
        re.IGNORECASE,
    )
    found: dict[tuple[int, int], FaersFile] = {}
    for path in data_root.rglob("*.txt"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        yy = int(match.group("yy"))
        year = 2000 + yy if yy < 90 else 1900 + yy
        quarter = int(match.group("q"))
        if not start_year <= year <= end_year:
            continue
        key = (year, quarter)
        candidate = FaersFile(role=role.lower(), year=year, quarter=quarter, path=path)
        previous = found.get(key)
        if previous is None:
            found[key] = candidate
        elif previous.path.resolve() != path.resolve():
            raise RuntimeError(
                f"Multiple {role} files found for {year}Q{quarter}: "
                f"{previous.path} and {path}"
            )
    return [found[key] for key in sorted(found)]


def validate_quarter_coverage(
    files: list[FaersFile], start_year: int, end_year: int, role: str
) -> None:
    expected = {(year, q) for year in range(start_year, end_year + 1) for q in range(1, 5)}
    observed = {(item.year, item.quarter) for item in files}
    missing = sorted(expected - observed)
    if missing:
        missing_text = ", ".join(f"{year}Q{q}" for year, q in missing)
        raise FileNotFoundError(f"Missing {role} quarters: {missing_text}")


def read_chunks(
    path: Path,
    wanted: Iterable[str],
    *,
    chunksize: int,
    optional: Iterable[str] = (),
) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, sep="$", nrows=0, encoding="latin-1")
    colmap = {str(column).strip().lower(): column for column in header.columns}
    wanted_set = set(wanted)
    optional_set = set(optional)
    missing_required = sorted(wanted_set - optional_set - set(colmap))
    if missing_required:
        raise ValueError(f"{path} is missing required columns: {missing_required}")
    usecols = [colmap[column] for column in wanted_set if column in colmap]
    reader = pd.read_csv(
        path,
        sep="$",
        usecols=usecols,
        dtype=str,
        encoding="latin-1",
        keep_default_na=False,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk.columns = [str(column).strip().lower() for column in chunk.columns]
        for column in optional_set - set(chunk.columns):
            chunk[column] = ""
        yield chunk


def _header_index(path: Path) -> dict[str, int]:
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        header = handle.readline().rstrip("\r\n")
    return {column.strip().lower(): index for index, column in enumerate(header.split("$"))}


def _rg_matching_lines(
    path: Path,
    *,
    literals: Iterable[str] | None = None,
    pattern_file: Path | None = None,
    ignore_case: bool = False,
) -> list[str]:
    """Use ripgrep as a fast raw-line prefilter, then parse exact FAERS fields."""
    rg = shutil.which("rg")
    if rg is None:
        raise RuntimeError("ripgrep (rg) is required for the fast feasibility scan.")
    command = [rg, "--no-filename", "--color", "never", "-F"]
    if ignore_case:
        command.append("-i")
    if literals is not None:
        for literal in literals:
            command.extend(["-e", literal])
    if pattern_file is not None:
        command.extend(["-f", str(pattern_file)])
    command.extend(["--", str(path)])
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode not in {0, 1}:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"rg failed for {path}: {stderr}")
    return [line.decode("latin-1", errors="replace") for line in completed.stdout.splitlines()]


def _records_from_lines(
    path: Path,
    lines: Iterable[str],
    wanted: Iterable[str],
    *,
    optional: Iterable[str] = (),
) -> pd.DataFrame:
    col_index = _header_index(path)
    wanted_list = list(wanted)
    optional_set = set(optional)
    missing_required = sorted(set(wanted_list) - optional_set - set(col_index))
    if missing_required:
        raise ValueError(f"{path} is missing required columns: {missing_required}")
    records: list[dict[str, str]] = []
    for line in lines:
        parts = line.rstrip("\r\n").split("$")
        record: dict[str, str] = {}
        for column in wanted_list:
            index = col_index.get(column)
            record[column] = parts[index] if index is not None and index < len(parts) else ""
        records.append(record)
    return pd.DataFrame.from_records(records, columns=wanted_list)


def _fixed_pattern_file(values: Iterable[str]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="ascii", newline="\n", suffix=".txt", delete=False
    )
    try:
        for value in sorted(set(values)):
            if value:
                handle.write(f"{value}\n")
    finally:
        handle.close()
    return Path(handle.name)


def load_deleted_caseids(data_root: Path) -> tuple[set[str], int]:
    deleted: set[str] = set()
    n_files = 0
    for directory in data_root.rglob("*"):
        if not directory.is_dir() or "delete" not in directory.name.lower():
            continue
        for path in sorted(directory.glob("*.txt")):
            n_files += 1
            with path.open("r", encoding="latin-1", errors="ignore") as handle:
                for raw_line in handle:
                    token = raw_line.strip().split("$", 1)[0].strip()
                    if token.isdigit():
                        deleted.add(token)
    return deleted, n_files


def scan_target_drugs(
    files: list[FaersFile], chunksize: int
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    del chunksize  # retained in the public interface for a pandas fallback if ever needed
    matches: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    literals = ["dabrafenib", "tafinlar", "trametinib", "mekinist"]
    for index, item in enumerate(files, start=1):
        matching_lines = _rg_matching_lines(
            item.path, literals=literals, ignore_case=True
        )
        chunk = _records_from_lines(
            item.path,
            matching_lines,
            ["primaryid", "caseid", "role_cod", "drugname", "prod_ai"],
            optional={"prod_ai"},
        )
        n_match_rows = 0
        if not chunk.empty:
            chunk["primaryid"] = normalize_id(chunk["primaryid"])
            chunk["caseid"] = normalize_id(chunk["caseid"])
            chunk["role_cod"] = chunk["role_cod"].fillna("").str.upper().str.strip()
            combined = normalize_text(chunk["drugname"]) + " " + normalize_text(chunk["prod_ai"])
            for target, pattern in TARGET_PATTERNS.items():
                mask = combined.str.contains(pattern, na=False)
                if not mask.any():
                    continue
                selected = chunk.loc[
                    mask,
                    ["primaryid", "caseid", "role_cod", "drugname", "prod_ai"],
                ].copy()
                selected["target_drug"] = target
                selected["source_period"] = item.period
                matches.append(selected)
                n_match_rows += len(selected)
        inventory.append(
            {
                "role": "drug",
                "period": item.period,
                "path": str(item.path),
                "file_size_bytes": item.path.stat().st_size,
                "rows_scanned": np.nan,
                "prefilter_match_lines": len(matching_lines),
                "target_rows": n_match_rows,
            }
        )
        print(
            f"[DRUG {index:02d}/{len(files)}] {item.period}: "
            f"prefilter_lines={len(matching_lines):,}, target_rows={n_match_rows:,}",
            flush=True,
        )
    if not matches:
        return pd.DataFrame(), inventory
    result = pd.concat(matches, ignore_index=True)
    result = result[(result["primaryid"] != "") & (result["caseid"] != "")].copy()
    result = result.drop_duplicates(
        ["primaryid", "caseid", "role_cod", "target_drug", "drugname", "prod_ai"]
    )
    return result, inventory


def scan_candidate_demo(
    files: list[FaersFile],
    candidate_caseids: set[str],
    chunksize: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    del chunksize
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    wanted = {
        "primaryid",
        "caseid",
        "caseversion",
        "fda_dt",
        "reporter_country",
        "occr_country",
    }
    pattern_file = _fixed_pattern_file(candidate_caseids)
    try:
        for index, item in enumerate(files, start=1):
            matching_lines = _rg_matching_lines(item.path, pattern_file=pattern_file)
            chunk = _records_from_lines(
                item.path,
                matching_lines,
                wanted,
                optional={"reporter_country", "occr_country"},
            )
            n_selected = 0
            if not chunk.empty:
                chunk["caseid"] = normalize_id(chunk["caseid"])
                selected = chunk[chunk["caseid"].isin(candidate_caseids)].copy()
                if not selected.empty:
                    selected["primaryid"] = normalize_id(selected["primaryid"])
                    selected["source_period"] = item.period
                    rows.append(selected)
                    n_selected = len(selected)
            inventory.append(
                {
                    "role": "demo",
                    "period": item.period,
                    "path": str(item.path),
                    "file_size_bytes": item.path.stat().st_size,
                    "rows_scanned": np.nan,
                    "prefilter_match_lines": len(matching_lines),
                    "target_rows": n_selected,
                }
            )
            print(
                f"[DEMO {index:02d}/{len(files)}] {item.period}: "
                f"prefilter_lines={len(matching_lines):,}, candidate_rows={n_selected:,}",
                flush=True,
            )
    finally:
        pattern_file.unlink(missing_ok=True)
    if not rows:
        return pd.DataFrame(), inventory
    return pd.concat(rows, ignore_index=True), inventory


def retain_latest_case_version(demo: pd.DataFrame) -> pd.DataFrame:
    if demo.empty:
        return demo
    work = demo.copy()
    work["caseversion_num"] = pd.to_numeric(work["caseversion"], errors="coerce")
    work["fda_dt_num"] = pd.to_numeric(work["fda_dt"], errors="coerce")
    work["primaryid_num"] = pd.to_numeric(work["primaryid"], errors="coerce")
    work = work[(work["caseid"] != "") & (work["primaryid"] != "")].copy()
    work = work.sort_values(
        ["caseid", "fda_dt_num", "caseversion_num", "primaryid_num", "source_period"],
        kind="mergesort",
        na_position="first",
    )
    work = work.drop_duplicates("caseid", keep="last")
    return work.drop(columns=["caseversion_num", "fda_dt_num", "primaryid_num"])


def build_exposure_cases(
    target_rows: pd.DataFrame,
    retained_demo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained_lookup = retained_demo.set_index("caseid")["primaryid"].to_dict()
    retained_rows = target_rows[
        target_rows.apply(
            lambda row: retained_lookup.get(row["caseid"]) == row["primaryid"], axis=1
        )
    ].copy()
    retained_rows = retained_rows.drop_duplicates(
        ["caseid", "primaryid", "target_drug", "role_cod"]
    )

    caseids = sorted(retained_rows["caseid"].unique())
    cases = pd.DataFrame({"caseid": caseids})
    for target in TARGET_PATTERNS:
        target_any = set(
            retained_rows.loc[retained_rows["target_drug"].eq(target), "caseid"]
        )
        target_ps_ss = set(
            retained_rows.loc[
                retained_rows["target_drug"].eq(target)
                & retained_rows["role_cod"].isin(ROLE_PS_SS),
                "caseid",
            ]
        )
        target_ps = set(
            retained_rows.loc[
                retained_rows["target_drug"].eq(target)
                & retained_rows["role_cod"].eq("PS"),
                "caseid",
            ]
        )
        cases[f"has_{target}"] = cases["caseid"].isin(target_any)
        cases[f"has_{target}_ps_ss"] = cases["caseid"].isin(target_ps_ss)
        cases[f"has_{target}_ps"] = cases["caseid"].isin(target_ps)

    cases["combo_all_roles"] = cases["has_dabrafenib"] & cases["has_trametinib"]
    cases["combo_at_least_one_ps_ss"] = cases["combo_all_roles"] & (
        cases["has_dabrafenib_ps_ss"] | cases["has_trametinib_ps_ss"]
    )
    cases["combo_both_ps_ss"] = cases["combo_all_roles"] & (
        cases["has_dabrafenib_ps_ss"] & cases["has_trametinib_ps_ss"]
    )
    cases["combo_both_ps"] = cases["combo_all_roles"] & (
        cases["has_dabrafenib_ps"] & cases["has_trametinib_ps"]
    )
    cases["dabrafenib_without_trametinib_all_roles"] = (
        cases["has_dabrafenib"] & ~cases["has_trametinib"]
    )
    cases["trametinib_without_dabrafenib_all_roles"] = (
        cases["has_trametinib"] & ~cases["has_dabrafenib"]
    )
    cases["dabrafenib_without_trametinib_ps_ss"] = (
        cases["has_dabrafenib_ps_ss"] & ~cases["has_trametinib"]
    )
    cases["trametinib_without_dabrafenib_ps_ss"] = (
        cases["has_trametinib_ps_ss"] & ~cases["has_dabrafenib"]
    )

    demo_columns = [
        "caseid",
        "primaryid",
        "caseversion",
        "fda_dt",
        "reporter_country",
        "occr_country",
        "source_period",
    ]
    cases = cases.merge(retained_demo[demo_columns], on="caseid", how="left", validate="one_to_one")
    return cases, retained_rows


def scan_reactions(
    files: list[FaersFile],
    retained_pid_to_case: dict[str, str],
    chunksize: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    del chunksize
    rows: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    target_primaryids = set(retained_pid_to_case)
    blank_pt_rows = 0
    pattern_file = _fixed_pattern_file(target_primaryids)
    try:
        for index, item in enumerate(files, start=1):
            matching_lines = _rg_matching_lines(item.path, pattern_file=pattern_file)
            chunk = _records_from_lines(
                item.path,
                matching_lines,
                ["primaryid", "caseid", "pt"],
                optional={"caseid"},
            )
            n_selected = 0
            if not chunk.empty:
                chunk["primaryid"] = normalize_id(chunk["primaryid"])
                selected = chunk[chunk["primaryid"].isin(target_primaryids)].copy()
                if not selected.empty:
                    selected["caseid"] = selected["primaryid"].map(retained_pid_to_case)
                    selected["pt"] = normalize_pt(selected["pt"])
                    blank_pt_rows += int(selected["pt"].eq("").sum())
                    selected = selected[selected["pt"] != ""]
                    rows.append(selected[["caseid", "primaryid", "pt"]])
                    n_selected = len(selected)
            inventory.append(
                {
                    "role": "reac",
                    "period": item.period,
                    "path": str(item.path),
                    "file_size_bytes": item.path.stat().st_size,
                    "rows_scanned": np.nan,
                    "prefilter_match_lines": len(matching_lines),
                    "target_rows": n_selected,
                }
            )
            print(
                f"[REAC {index:02d}/{len(files)}] {item.period}: "
                f"prefilter_lines={len(matching_lines):,}, retained_target_rows={n_selected:,}",
                flush=True,
            )
    finally:
        pattern_file.unlink(missing_ok=True)
    if not rows:
        return pd.DataFrame(columns=["caseid", "primaryid", "pt"]), inventory, blank_pt_rows
    result = pd.concat(rows, ignore_index=True).drop_duplicates(["caseid", "pt"])
    return result, inventory, blank_pt_rows


def add_reaction_flags(cases: pd.DataFrame, reaction_links: pd.DataFrame) -> pd.DataFrame:
    pt_sets = reaction_links.groupby("caseid")["pt"].agg(set).to_dict()
    out = cases.copy()
    out["has_any_reaction_pt"] = out["caseid"].map(lambda caseid: bool(pt_sets.get(caseid, set())))
    out["has_core_pyrexia"] = out["caseid"].map(
        lambda caseid: bool(pt_sets.get(caseid, set()) & CORE_FEVER_PTS)
    )
    out["has_extended_fever"] = out["caseid"].map(
        lambda caseid: bool(pt_sets.get(caseid, set()) & EXTENDED_FEVER_PTS)
    )
    return out


def cohort_summary(cases: pd.DataFrame) -> pd.DataFrame:
    cohorts = [
        "combo_all_roles",
        "combo_at_least_one_ps_ss",
        "combo_both_ps_ss",
        "combo_both_ps",
        "dabrafenib_without_trametinib_all_roles",
        "trametinib_without_dabrafenib_all_roles",
        "dabrafenib_without_trametinib_ps_ss",
        "trametinib_without_dabrafenib_ps_ss",
    ]
    rows: list[dict[str, object]] = []
    for cohort in cohorts:
        subset = cases[cases[cohort]]
        n = len(subset)
        core_n = int(subset["has_core_pyrexia"].sum())
        extended_n = int(subset["has_extended_fever"].sum())
        rows.append(
            {
                "cohort": cohort,
                "exposure_caseids": n,
                "core_pyrexia_caseids": core_n,
                "core_pyrexia_report_pct": round(core_n / n * 100, 2) if n else np.nan,
                "extended_fever_caseids": extended_n,
                "extended_fever_report_pct": round(extended_n / n * 100, 2) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def co_reported_pt_table(
    cases: pd.DataFrame, reaction_links: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fever_caseids = set(
        cases.loc[cases["combo_all_roles"] & cases["has_core_pyrexia"], "caseid"]
    )
    no_fever_caseids = set(
        cases.loc[cases["combo_all_roles"] & ~cases["has_extended_fever"], "caseid"]
    )
    strict_fever_caseids = set(
        cases.loc[cases["combo_both_ps_ss"] & cases["has_core_pyrexia"], "caseid"]
    )
    non_anchor = reaction_links[~reaction_links["pt"].isin(EXTENDED_FEVER_PTS)].copy()

    fever_links = non_anchor[non_anchor["caseid"].isin(fever_caseids)]
    no_fever_links = non_anchor[non_anchor["caseid"].isin(no_fever_caseids)]
    strict_links = non_anchor[non_anchor["caseid"].isin(strict_fever_caseids)]

    fever_counts = fever_links.groupby("pt")["caseid"].nunique()
    no_fever_counts = no_fever_links.groupby("pt")["caseid"].nunique()
    strict_counts = strict_links.groupby("pt")["caseid"].nunique()
    pts = sorted(set(fever_counts.index) | set(no_fever_counts.index))
    table = pd.DataFrame({"co_reported_pt": pts})
    table["n_core_pyrexia_cases"] = table["co_reported_pt"].map(fever_counts).fillna(0).astype(int)
    table["pct_core_pyrexia_cases"] = (
        table["n_core_pyrexia_cases"] / max(len(fever_caseids), 1) * 100
    ).round(2)
    table["n_no_fever_cases"] = table["co_reported_pt"].map(no_fever_counts).fillna(0).astype(int)
    table["pct_no_fever_cases"] = (
        table["n_no_fever_cases"] / max(len(no_fever_caseids), 1) * 100
    ).round(2)
    table["prevalence_difference_pp"] = (
        table["pct_core_pyrexia_cases"] - table["pct_no_fever_cases"]
    ).round(2)
    table["n_strict_both_ps_ss_core_pyrexia"] = (
        table["co_reported_pt"].map(strict_counts).fillna(0).astype(int)
    )
    table = table.sort_values(
        ["n_core_pyrexia_cases", "prevalence_difference_pp", "co_reported_pt"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    richness = pd.DataFrame({"caseid": sorted(fever_caseids)})
    non_anchor_counts = fever_links.groupby("caseid")["pt"].nunique()
    richness["n_unique_non_fever_pt"] = (
        richness["caseid"].map(non_anchor_counts).fillna(0).astype(int)
    )

    group_rows: list[dict[str, object]] = []
    fever_pt_sets = fever_links.groupby("caseid")["pt"].agg(set).to_dict()
    for group, group_pts in PHENOTYPE_GROUPS.items():
        matched_cases = {
            caseid for caseid in fever_caseids if fever_pt_sets.get(caseid, set()) & group_pts
        }
        group_rows.append(
            {
                "phenotype_group": group,
                "n_core_pyrexia_cases": len(matched_cases),
                "pct_core_pyrexia_cases": round(
                    len(matched_cases) / max(len(fever_caseids), 1) * 100, 2
                ),
                "predefined_pts": "|".join(sorted(group_pts)),
                "observed_pts": "|".join(
                    sorted(set().union(*(fever_pt_sets.get(caseid, set()) for caseid in matched_cases)) & group_pts)
                ),
            }
        )
    return table, richness, pd.DataFrame(group_rows)


def richness_summary(
    richness: pd.DataFrame,
    co_pt: pd.DataFrame,
) -> pd.DataFrame:
    values = richness["n_unique_non_fever_pt"] if not richness.empty else pd.Series(dtype=float)
    total_links = int(values.sum()) if len(values) else 0
    top_counts = co_pt["n_core_pyrexia_cases"].tolist() if not co_pt.empty else []
    metrics = [
        ("core_pyrexia_cases", len(values)),
        ("median_unique_non_fever_pt", float(values.median()) if len(values) else np.nan),
        ("q1_unique_non_fever_pt", float(values.quantile(0.25)) if len(values) else np.nan),
        ("q3_unique_non_fever_pt", float(values.quantile(0.75)) if len(values) else np.nan),
        ("mean_unique_non_fever_pt", round(float(values.mean()), 2) if len(values) else np.nan),
        ("anchor_only_cases", int((values == 0).sum()) if len(values) else 0),
        ("pct_with_at_least_1_non_fever_pt", round(float((values >= 1).mean() * 100), 2) if len(values) else np.nan),
        ("pct_with_at_least_2_non_fever_pt", round(float((values >= 2).mean() * 100), 2) if len(values) else np.nan),
        ("pct_with_at_least_3_non_fever_pt", round(float((values >= 3).mean() * 100), 2) if len(values) else np.nan),
        ("pct_with_at_least_5_non_fever_pt", round(float((values >= 5).mean() * 100), 2) if len(values) else np.nan),
        ("distinct_non_fever_pts", int((co_pt["n_core_pyrexia_cases"] > 0).sum()) if not co_pt.empty else 0),
        ("total_case_pt_links", total_links),
        ("top1_share_of_case_pt_links_pct", round(top_counts[0] / total_links * 100, 2) if total_links and top_counts else np.nan),
        ("top5_share_of_case_pt_links_pct", round(sum(top_counts[:5]) / total_links * 100, 2) if total_links else np.nan),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def annual_summary(cases: pd.DataFrame) -> pd.DataFrame:
    work = cases.copy()
    work["report_year"] = pd.to_numeric(work["fda_dt"], errors="coerce").floordiv(10000)
    work["report_year"] = work["report_year"].fillna(
        work["source_period"].str[:4].astype(float)
    )
    rows: list[dict[str, object]] = []
    years = sorted(int(year) for year in work["report_year"].dropna().unique())
    for year in years:
        subset = work[work["report_year"].eq(year)]
        combo = subset[subset["combo_all_roles"]]
        strict = subset[subset["combo_both_ps_ss"]]
        rows.append(
            {
                "year": year,
                "combo_all_roles_caseids": len(combo),
                "combo_core_pyrexia_caseids": int(combo["has_core_pyrexia"].sum()),
                "combo_core_pyrexia_report_pct": round(
                    float(combo["has_core_pyrexia"].mean() * 100), 2
                )
                if len(combo)
                else np.nan,
                "combo_both_ps_ss_caseids": len(strict),
                "combo_both_ps_ss_core_pyrexia_caseids": int(
                    strict["has_core_pyrexia"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return strict-JSON records, converting pandas missing values to null."""
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=300_000)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered: dict[str, list[FaersFile]] = {}
    for role in ("DRUG", "DEMO", "REAC"):
        files = discover_files(data_root, role, args.start_year, args.end_year)
        validate_quarter_coverage(files, args.start_year, args.end_year, role)
        discovered[role.lower()] = files
        print(f"{role}: {len(files)} complete quarters discovered", flush=True)

    deleted_caseids, deleted_file_count = load_deleted_caseids(data_root)
    print(
        f"Deleted-case lists: files={deleted_file_count:,}, unique_caseids={len(deleted_caseids):,}",
        flush=True,
    )

    target_rows, drug_inventory = scan_target_drugs(
        discovered["drug"], chunksize=args.chunksize
    )
    if target_rows.empty:
        raise RuntimeError("No dabrafenib or trametinib rows were found.")

    raw_candidate_caseids = set(target_rows["caseid"])
    deleted_target_caseids = raw_candidate_caseids & deleted_caseids
    candidate_caseids = raw_candidate_caseids - deleted_caseids
    target_rows = target_rows[target_rows["caseid"].isin(candidate_caseids)].copy()

    demo_rows, demo_inventory = scan_candidate_demo(
        discovered["demo"], candidate_caseids, chunksize=args.chunksize
    )
    retained_demo = retain_latest_case_version(demo_rows)
    demo_caseids = set(retained_demo["caseid"])
    missing_demo_caseids = candidate_caseids - demo_caseids
    if missing_demo_caseids:
        target_rows = target_rows[~target_rows["caseid"].isin(missing_demo_caseids)].copy()

    cases, retained_target_rows = build_exposure_cases(target_rows, retained_demo)
    retained_pid_to_case = dict(zip(cases["primaryid"], cases["caseid"]))
    reaction_links, reac_inventory, blank_pt_rows = scan_reactions(
        discovered["reac"], retained_pid_to_case, chunksize=args.chunksize
    )
    cases = add_reaction_flags(cases, reaction_links)

    cohorts = cohort_summary(cases)
    co_pt, richness_cases, phenotype_groups = co_reported_pt_table(cases, reaction_links)
    richness = richness_summary(richness_cases, co_pt)
    annual = annual_summary(cases)

    inventory = pd.DataFrame(drug_inventory + demo_inventory + reac_inventory)
    qc_rows = [
        ("analysis_start_year", args.start_year),
        ("analysis_end_year", args.end_year),
        ("quarters_expected_per_table", (args.end_year - args.start_year + 1) * 4),
        ("drug_files", len(discovered["drug"])),
        ("demo_files", len(discovered["demo"])),
        ("reac_files", len(discovered["reac"])),
        ("deleted_list_files", deleted_file_count),
        ("deleted_caseids_global", len(deleted_caseids)),
        ("raw_target_drug_rows", len(target_rows)),
        ("raw_candidate_caseids_before_deleted_filter", len(raw_candidate_caseids)),
        ("candidate_caseids_removed_by_deleted_lists", len(deleted_target_caseids)),
        ("candidate_caseids_missing_demo", len(missing_demo_caseids)),
        ("retained_latest_candidate_caseids", len(retained_demo)),
        ("retained_caseids_with_target_drug_in_latest_version", len(cases)),
        ("retained_target_drug_rows", len(retained_target_rows)),
        ("duplicate_caseids_final", int(cases["caseid"].duplicated().sum())),
        ("target_cases_without_reaction_pt", int((~cases["has_any_reaction_pt"]).sum())),
        ("blank_reaction_pt_rows", blank_pt_rows),
    ]
    qc = pd.DataFrame(qc_rows, columns=["metric", "value"])

    outputs = {
        "cohort_summary": output_dir / "cohort_summary.csv",
        "co_reported_pt": output_dir / "co_reported_pt.csv",
        "richness_summary": output_dir / "richness_summary.csv",
        "phenotype_groups": output_dir / "phenotype_groups.csv",
        "annual_summary": output_dir / "annual_summary.csv",
        "qc_metrics": output_dir / "qc_metrics.csv",
        "source_inventory": output_dir / "source_inventory.csv",
        "case_summary": output_dir / "case_summary.parquet",
        "retained_target_drug_rows": output_dir / "retained_target_drug_rows.parquet",
        "reaction_links": output_dir / "reaction_links.parquet",
        "main_pyrexia_richness_cases": output_dir / "main_pyrexia_richness_cases.parquet",
    }
    cohorts.to_csv(outputs["cohort_summary"], index=False, encoding="utf-8-sig")
    co_pt.to_csv(outputs["co_reported_pt"], index=False, encoding="utf-8-sig")
    richness.to_csv(outputs["richness_summary"], index=False, encoding="utf-8-sig")
    phenotype_groups.to_csv(outputs["phenotype_groups"], index=False, encoding="utf-8-sig")
    annual.to_csv(outputs["annual_summary"], index=False, encoding="utf-8-sig")
    qc.to_csv(outputs["qc_metrics"], index=False, encoding="utf-8-sig")
    inventory.to_csv(outputs["source_inventory"], index=False, encoding="utf-8-sig")
    cases.to_parquet(outputs["case_summary"], index=False)
    retained_target_rows.to_parquet(outputs["retained_target_drug_rows"], index=False)
    reaction_links.to_parquet(outputs["reaction_links"], index=False)
    richness_cases.to_parquet(outputs["main_pyrexia_richness_cases"], index=False)

    report_payload = {
        "generated_from": "raw FAERS ASCII",
        "period": f"{args.start_year}Q1-{args.end_year}Q4",
        "unit": "latest non-deleted CASEID",
        "cohorts": json_records(cohorts),
        "richness": json_records(richness),
        "phenotype_groups": json_records(phenotype_groups),
        "top_co_reported_pts": json_records(co_pt.head(30)),
        "annual": json_records(annual),
        "qc": json_records(qc),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(output_dir / "report_payload.json", report_payload)

    print("\n=== COHORT SUMMARY ===", flush=True)
    print(cohorts.to_string(index=False), flush=True)
    print("\n=== RICHNESS SUMMARY ===", flush=True)
    print(richness.to_string(index=False), flush=True)
    print("\n=== PHENOTYPE GROUPS ===", flush=True)
    print(phenotype_groups.to_string(index=False), flush=True)
    print("\n=== TOP 20 CO-REPORTED PTs ===", flush=True)
    print(co_pt.head(20).to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
