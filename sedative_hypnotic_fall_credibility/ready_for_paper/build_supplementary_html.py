from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
OUTPUT = Path(__file__).with_name("DrugSafety_Supplementary_Materials_20260713.html")

SUPPLEMENTS = [
    (1, "Target-drug dictionary and drug-name matching terms", [
        ("Panel A. Target-drug dictionary", "table_s1_drug_dictionary.csv"),
        ("Panel B. Drug-name matching terms", "table_s1_drug_name_matching_terms.csv"),
    ]),
    (2, "Definitions of the strict-fall outcome and phenotype domains", [(None, "table_s2_outcome_phenotype_definitions.csv")]),
    (3, "Primary-suspect-only sensitivity analysis", [(None, "table_s3_ps_only_sensitivity.csv")]),
    (4, "Sensitivity analysis excluding mixed sedative-hypnotic exposure", [(None, "table_s4_excluding_mixed_exposure_sensitivity.csv")]),
    (5, "Reporting-source-stratified sensitivity analyses", [(None, "table_s5_reporting_source_stratified_sensitivity.csv")]),
    (6, "Primary phenotype distribution among strict-fall reports", [(None, "table_s6_primary_phenotype_distribution.csv")]),
    (7, "Chi-square tests of phenotype distributions", [(None, "table_s7_phenotype_chi_square_tests.csv")]),
    (8, "Adjusted logistic models for phenotype components", [(None, "table_s8_phenotype_adjusted_logistic_models.csv")]),
    (9, "Drug-level phenotype fingerprints", [(None, "table_s9_drug_level_phenotype_fingerprint.csv")]),
    (10, "Drug-level crude phenotype contrasts", [(None, "table_s10_drug_level_phenotype_crude_contrasts.csv")]),
    (11, "Phenotype-cluster model selection", [(None, "table_s11_cluster_model_selection.csv")]),
    (12, "Summary of fall-phenotype clusters", [(None, "table_s12_fall_phenotype_cluster_summary.csv")]),
    (13, "Distribution of phenotype clusters by drug group", [(None, "table_s13_drug_group_by_cluster_distribution.csv")]),
    (14, "Internal-support evidence-component details", [(None, "table_s14_evidence_component_details.csv")]),
    (15, "Full characteristics of mutually exclusive comparator groups", [(None, "table_s15_mutually_exclusive_comparator_characteristics.csv")]),
    (16, "BCPNN signal-sensitivity results", [(None, "table_s16_bcpnn_signal_sensitivity.csv")]),
    (17, "Leave-one-domain-out analysis of internal-support classification", [(None, "table_s17_credibility_leave_one_domain_out.csv")]),
    (18, "Threshold-sensitivity analysis of internal-support classification", [(None, "table_s18_credibility_threshold_sensitivity.csv")]),
]


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return (rows[0], rows[1:]) if rows else ([], [])


def render_table(path: Path, panel_title: str | None = None) -> str:
    headers, rows = read_csv(path)
    heading = f"<h3>{html.escape(panel_title)}</h3>" if panel_title else ""
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body_html = []
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        body_html.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in padded[: len(headers)]) + "</tr>")
    return f"""
      {heading}
      <p class="source">Source data: <code>outputs/tables/{html.escape(path.name)}</code> · {len(rows):,} data rows</p>
      <div class="table-scroll"><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table></div>
    """


def model_specification() -> str:
    rows = [
        ("Crude model", "Exposure group only"),
        ("Model 1", "Exposure group, age group, sex, and reporting year"),
        ("Model 2", "Model 1 plus country/region group, FAERS report type code, and electronic submission status"),
        ("Model 3 (fully adjusted)", "Model 2 plus polypharmacy, antidepressant, antipsychotic, opioid and antiepileptic co-reporting, and indication markers for insomnia, anxiety, depression, pain and epilepsy"),
    ]
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in rows)
    return f"""
    <section id="model-specification" class="supplement">
      <h2>Supplementary Methods 1. Covariates and active-comparator model specifications</h2>
      <p>Covariates were derived at case level. Missing binary indicators were set to false after case-level merging, categorical variables were treated as factors, and variables with insufficient variation or causing instability could be omitted from an individual comparison model.</p>
      <div class="table-scroll"><table><thead><tr><th>Model</th><th>Variables included</th></tr></thead><tbody>{body}</tbody></table></div>
      <p class="note"><strong>Interpretation:</strong> adjusted odds ratios describe differences in FAERS reporting patterns. They do not estimate incidence, absolute risk, or causal effects.</p>
    </section>
    """


def build() -> str:
    toc = ['<li><a href="#model-specification">Supplementary Methods 1. Model specifications</a></li>']
    sections = [model_specification()]
    for number, title, panels in SUPPLEMENTS:
        toc.append(f'<li><a href="#table-s{number}">Table S{number}. {html.escape(title)}</a></li>')
        rendered = "".join(render_table(TABLE_DIR / filename, panel) for panel, filename in panels)
        open_attr = " open" if number <= 5 else ""
        sections.append(f"""
        <section id="table-s{number}" class="supplement">
          <details{open_attr}>
            <summary>Supplementary Table S{number}. {html.escape(title)}</summary>
            <div class="details-body">{rendered}</div>
          </details>
        </section>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Electronic Supplementary Material - Sedative-Hypnotics and Fall-Related FAERS Signals</title>
<style>
:root {{ --ink:#1f2933; --muted:#64748b; --line:#cbd5e1; --head:#eaf1f6; --accent:#235f7a; --paper:#fff; --bg:#edf2f5; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--bg); font:15px/1.55 Arial, Helvetica, sans-serif; }}
main {{ width:min(1500px, calc(100% - 32px)); margin:24px auto 60px; background:var(--paper); padding:42px; box-shadow:0 8px 28px #1e293b1c; }}
h1 {{ margin:0 0 8px; font:700 31px/1.2 Georgia, serif; }}
h2 {{ font:700 23px/1.3 Georgia, serif; color:#173f52; }}
h3 {{ margin-top:24px; }}
.subtitle {{ font-size:19px; color:#334155; }}
.meta,.source,.note {{ color:var(--muted); }}
.notice {{ padding:14px 16px; border-left:4px solid var(--accent); background:#eff7fa; }}
nav {{ columns:2; column-gap:40px; margin:28px 0 34px; padding:18px 24px; background:#f8fafc; border:1px solid var(--line); }}
nav h2 {{ column-span:all; margin-top:0; }}
nav li {{ break-inside:avoid; margin:0 0 7px; }}
a {{ color:#145d78; text-decoration:none; }}
.supplement {{ scroll-margin-top:16px; margin:34px 0; border-top:2px solid #b9cbd5; padding-top:18px; }}
summary {{ cursor:pointer; color:#173f52; font:700 22px/1.35 Georgia, serif; }}
.details-body {{ padding-top:12px; }}
.table-scroll {{ overflow:auto; max-height:72vh; border:1px solid var(--line); background:white; }}
table {{ border-collapse:collapse; width:max-content; min-width:100%; font-size:12px; }}
th,td {{ border:1px solid #d7e0e6; padding:6px 8px; text-align:left; vertical-align:top; max-width:340px; overflow-wrap:anywhere; }}
th {{ position:sticky; top:0; z-index:1; background:var(--head); color:#173f52; white-space:normal; }}
tbody tr:nth-child(even) {{ background:#f8fafc; }}
code {{ font-size:12px; }}
.top {{ display:inline-block; margin-top:10px; }}
@media (max-width:800px) {{ main {{ width:100%; margin:0; padding:20px; }} nav {{ columns:1; }} }}
@media print {{ body {{ background:white; }} main {{ width:100%; margin:0; padding:0; box-shadow:none; }} nav {{ columns:1; }} details {{ display:block; }} details > summary {{ list-style:none; }} .table-scroll {{ overflow:visible; max-height:none; border:0; }} table {{ width:100%; font-size:8pt; }} th {{ position:static; }} .supplement {{ break-before:page; }} }}
</style>
</head>
<body><main id="top">
<header>
  <p class="meta">Electronic Supplementary Material · Prepared 13 July 2026</p>
  <h1>Phenotype-Guided Dissection of Fall-Related Pharmacovigilance Signals for Sedative-Hypnotics in Older Adults</h1>
  <p class="subtitle">Supplementary Methods and Tables</p>
  <p class="notice">This file consolidates the project’s existing supplementary result tables. Results describe reporting patterns in FAERS and must not be interpreted as incidence, absolute risk, or causal effects.</p>
</header>
<nav><h2>Contents</h2><ol>{''.join(toc)}</ol></nav>
{''.join(sections)}
<p><a class="top" href="#top">Back to top</a></p>
</main></body></html>"""


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(OUTPUT)
