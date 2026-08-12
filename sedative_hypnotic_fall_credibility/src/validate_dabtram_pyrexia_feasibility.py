"""Independent consistency checks for the dabrafenib/trametinib audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "outputs" / "dabtram_pyrexia_feasibility"
)
FEVER_EXCLUSION = {
    "PYREXIA",
    "HYPERPYREXIA",
    "HYPERTHERMIA",
    "BODY TEMPERATURE INCREASED",
    "FEVER",
    "FEBRILE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    cases = pd.read_parquet(output_dir / "case_summary.parquet")
    reactions = pd.read_parquet(output_dir / "reaction_links.parquet")
    target_rows = pd.read_parquet(output_dir / "retained_target_drug_rows.parquet")
    cohorts = pd.read_csv(output_dir / "cohort_summary.csv")
    annual = pd.read_csv(output_dir / "annual_summary.csv")
    reported_richness = pd.read_csv(output_dir / "richness_summary.csv").set_index("metric")["value"]
    inventory = pd.read_csv(output_dir / "source_inventory.csv")

    combo = cases["combo_all_roles"]
    strict = cases["combo_both_ps_ss"]
    core = cases["has_core_pyrexia"]
    extended = cases["has_extended_fever"]
    main_caseids = set(cases.loc[combo & core, "caseid"])
    non_fever_links = reactions[
        reactions["caseid"].isin(main_caseids) & ~reactions["pt"].isin(FEVER_EXCLUSION)
    ]
    counts = non_fever_links.groupby("caseid")["pt"].nunique().reindex(main_caseids, fill_value=0)

    cohort_lookup = cohorts.set_index("cohort")
    checks: list[dict[str, object]] = []

    def check(name: str, observed: object, expected: object) -> None:
        if isinstance(observed, float) or isinstance(expected, float):
            passed = abs(float(observed) - float(expected)) < 0.011
        else:
            passed = observed == expected
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "passed": bool(passed),
            }
        )

    check("caseid_unique", int(cases["caseid"].duplicated().sum()), 0)
    check("primaryid_unique", int(cases["primaryid"].duplicated().sum()), 0)
    check(
        "reaction_case_pt_unique",
        int(reactions.duplicated(["caseid", "pt"]).sum()),
        0,
    )
    check(
        "all_reaction_caseids_in_case_summary",
        int((~reactions["caseid"].isin(cases["caseid"])).sum()),
        0,
    )
    check(
        "target_rows_use_retained_primaryid",
        int(
            (
                target_rows.merge(
                    cases[["caseid", "primaryid"]],
                    on="caseid",
                    how="left",
                    suffixes=("_row", "_case"),
                    validate="many_to_one",
                )["primaryid_row"]
                != target_rows.merge(
                    cases[["caseid", "primaryid"]],
                    on="caseid",
                    how="left",
                    suffixes=("_row", "_case"),
                    validate="many_to_one",
                )["primaryid_case"]
            ).sum()
        ),
        0,
    )
    check(
        "combo_all_roles_count",
        int(combo.sum()),
        int(cohort_lookup.loc["combo_all_roles", "exposure_caseids"]),
    )
    check(
        "combo_core_pyrexia_count",
        int((combo & core).sum()),
        int(cohort_lookup.loc["combo_all_roles", "core_pyrexia_caseids"]),
    )
    check(
        "combo_extended_fever_count",
        int((combo & extended).sum()),
        int(cohort_lookup.loc["combo_all_roles", "extended_fever_caseids"]),
    )
    check(
        "strict_combo_count",
        int(strict.sum()),
        int(cohort_lookup.loc["combo_both_ps_ss", "exposure_caseids"]),
    )
    check(
        "strict_combo_core_pyrexia_count",
        int((strict & core).sum()),
        int(cohort_lookup.loc["combo_both_ps_ss", "core_pyrexia_caseids"]),
    )
    check(
        "annual_combo_sum",
        int(annual["combo_all_roles_caseids"].sum()),
        int(combo.sum()),
    )
    check(
        "annual_combo_core_pyrexia_sum",
        int(annual["combo_core_pyrexia_caseids"].sum()),
        int((combo & core).sum()),
    )
    check(
        "richness_median",
        float(counts.median()),
        float(reported_richness["median_unique_non_fever_pt"]),
    )
    check(
        "richness_q1",
        float(counts.quantile(0.25)),
        float(reported_richness["q1_unique_non_fever_pt"]),
    )
    check(
        "richness_q3",
        float(counts.quantile(0.75)),
        float(reported_richness["q3_unique_non_fever_pt"]),
    )
    check(
        "richness_pct_at_least_2",
        round(float((counts >= 2).mean() * 100), 2),
        float(reported_richness["pct_with_at_least_2_non_fever_pt"]),
    )
    for role in ("drug", "demo", "reac"):
        check(
            f"source_inventory_{role}_quarters",
            int(inventory.loc[inventory["role"].eq(role), "period"].nunique()),
            52,
        )

    check(
        "role_nesting_both_ps_ss_within_at_least_one",
        int((cases["combo_both_ps_ss"] & ~cases["combo_at_least_one_ps_ss"]).sum()),
        0,
    )
    check(
        "role_nesting_at_least_one_within_all",
        int((cases["combo_at_least_one_ps_ss"] & ~cases["combo_all_roles"]).sum()),
        0,
    )

    result = pd.DataFrame(checks)
    result.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    failures = result.loc[~result["passed"]]
    status = {
        "status": "passed" if failures.empty else "failed",
        "checks": len(result),
        "failures": int(len(failures)),
        "failed_checks": failures["check"].tolist(),
    }
    (output_dir / "validation_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print(json.dumps(status, ensure_ascii=False))
    if not failures.empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
