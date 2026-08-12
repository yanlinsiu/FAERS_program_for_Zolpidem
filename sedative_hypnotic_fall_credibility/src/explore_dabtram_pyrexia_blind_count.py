"""Blind feasibility count for dabrafenib + trametinib pyrexia syndrome.

Only produces counts, missing rates, and co-reported PT tallies. No ROR / OR /
significance is computed, so the topic is not selected by effect direction.

Scans raw FAERS ASCII (DRUG for co-exposure, REAC for reactions). Deduplication
is approximate: we report both primaryid-level and unique-caseid-level counts.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"D:\program_FAERS\data")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "qc"

DAB = re.compile(r"\b(dabrafenib|tafinlar)\b")
TRAM = re.compile(r"\b(trametinib|mekinist)\b")

# Core fever PTs (MedDRA). Chills/night sweats/myalgia are companions, not fever.
PYREXIA_PT = {
    "pyrexia",
    "body temperature increased",
    "hyperthermia",
    "hyperpyrexia",
    "febrile",
    "fever",
}

ROLE_PS_SS = {"PS", "SS"}


def norm(value: object) -> str:
    if value is None or (isinstance(value, float)):
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def load(path: Path, targets: set[str]) -> pd.DataFrame:
    header = pd.read_csv(path, sep="$", nrows=0, encoding="latin-1")
    colmap = {c.lower().strip(): c for c in header.columns}
    use = [colmap[t] for t in targets if t in colmap]
    if not use:
        return pd.DataFrame(columns=list(targets))
    df = pd.read_csv(path, sep="$", usecols=use, dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def drug_files() -> list[Path]:
    files = sorted(DATA_DIR.rglob("DRUG*.txt")) + sorted(DATA_DIR.rglob("drug*.txt"))
    out = []
    for p in files:
        m = re.search(r"drug(\d{2})q[1-4]", p.name.lower())
        if m and int(m.group(1)) >= 13:  # dabrafenib/trametinib launched 2013
            out.append(p)
    return sorted(set(out))


def reac_files() -> list[Path]:
    files = sorted(DATA_DIR.rglob("REAC*.txt")) + sorted(DATA_DIR.rglob("reac*.txt"))
    out = []
    for p in files:
        m = re.search(r"reac(\d{2})q[1-4]", p.name.lower())
        if m and int(m.group(1)) >= 13:
            out.append(p)
    return sorted(set(out))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # role per primaryid for each drug: keep strongest role seen (PS>SS>other)
    dab_role: dict[str, str] = {}
    tram_role: dict[str, str] = {}
    pid_to_case: dict[str, str] = {}

    def upgrade(store: dict[str, str], pid: str, role: str) -> None:
        rank = {"PS": 3, "SS": 2}
        cur = store.get(pid, "")
        if rank.get(role, 1) > rank.get(cur, 0):
            store[pid] = role

    for path in drug_files():
        df = load(path, {"primaryid", "caseid", "role_cod", "drugname", "prod_ai"})
        if df.empty or "primaryid" not in df.columns:
            continue
        text = df.get("drugname", "").map(norm) + " " + df.get("prod_ai", pd.Series("", index=df.index)).map(norm)
        role = df.get("role_cod", pd.Series("", index=df.index)).fillna("").str.upper().str.strip()
        pid = df["primaryid"].fillna("")
        case = df.get("caseid", pd.Series("", index=df.index)).fillna("")
        for p, c in zip(pid, case):
            if p and c:
                pid_to_case.setdefault(p, c)
        dab_mask = text.str.contains(DAB, na=False)
        tram_mask = text.str.contains(TRAM, na=False)
        for p, r in zip(pid[dab_mask], role[dab_mask]):
            if p:
                upgrade(dab_role, p, r)
        for p, r in zip(pid[tram_mask], role[tram_mask]):
            if p:
                upgrade(tram_role, p, r)
        print(f"DRUG {path.name}: dab_pids={len(dab_role):,} tram_pids={len(tram_role):,}")

    combo_pids = set(dab_role) & set(tram_role)
    combo_cases = {pid_to_case.get(p, p) for p in combo_pids}
    combo_ps_ss = {
        p for p in combo_pids
        if dab_role.get(p) in ROLE_PS_SS and tram_role.get(p) in ROLE_PS_SS
    }
    combo_ps_ss_cases = {pid_to_case.get(p, p) for p in combo_ps_ss}

    # REAC pass restricted to combo primaryids
    pt_by_pid: dict[str, set[str]] = {p: set() for p in combo_pids}
    for path in reac_files():
        df = load(path, {"primaryid", "pt"})
        if df.empty or "primaryid" not in df.columns or "pt" not in df.columns:
            continue
        df = df[df["primaryid"].isin(combo_pids)]
        for p, pt in zip(df["primaryid"], df["pt"]):
            if p in pt_by_pid:
                pt_by_pid[p].add(norm(pt))
        print(f"REAC {path.name}: matched rows accumulated")

    def has_pyrexia(pts: set[str]) -> bool:
        return any(pt in PYREXIA_PT for pt in pts)

    fever_pids = {p for p, pts in pt_by_pid.items() if has_pyrexia(pts)}
    fever_cases = {pid_to_case.get(p, p) for p in fever_pids}
    fever_ps_ss = fever_pids & combo_ps_ss

    # co-reported PT tally among fever cases (exclude the fever PTs themselves)
    co_pt = Counter()
    for p in fever_pids:
        for pt in pt_by_pid[p]:
            if pt and pt not in PYREXIA_PT:
                co_pt[pt] += 1

    summary = pd.DataFrame(
        [
            {"metric": "combo_coreport_primaryids", "value": len(combo_pids)},
            {"metric": "combo_coreport_unique_caseids", "value": len(combo_cases)},
            {"metric": "combo_both_ps_ss_primaryids", "value": len(combo_ps_ss)},
            {"metric": "combo_both_ps_ss_unique_caseids", "value": len(combo_ps_ss_cases)},
            {"metric": "combo_with_pyrexia_primaryids", "value": len(fever_pids)},
            {"metric": "combo_with_pyrexia_unique_caseids", "value": len(fever_cases)},
            {"metric": "combo_with_pyrexia_both_ps_ss_primaryids", "value": len(fever_ps_ss)},
        ]
    )
    summary.to_csv(OUT_DIR / "explore_dabtram_summary.csv", index=False, encoding="utf-8-sig")

    co_df = (
        pd.DataFrame(co_pt.most_common(60), columns=["co_reported_pt", "n_fever_cases_with_pt"])
        .assign(pct_of_fever_cases=lambda d: (d["n_fever_cases_with_pt"] / max(len(fever_pids), 1) * 100).round(1))
    )
    co_df.to_csv(OUT_DIR / "explore_dabtram_fever_copt.csv", index=False, encoding="utf-8-sig")

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nFever cases (primaryid): {len(fever_pids):,}")
    print("Top co-reported PTs among fever cases:")
    print(co_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
