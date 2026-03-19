from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from config import DEFAULT_OUTPUT_ROOT, RAW_ROOT


KNOWN_DATASETS = ("DEMO", "DRUG", "INDI", "OUTC", "REAC", "RPSR", "THER", "STAT", "SIZE")
DELIMITER_CANDIDATES = ("$", "\t", "|", ",")


def clean_column_name(column_name: str) -> str:
    cleaned = column_name.strip()
    cleaned = cleaned.lstrip("\ufeff")
    if cleaned.startswith("ï»¿"):
        cleaned = cleaned[3:]
    return cleaned


def detect_delimiter(header_line: str) -> str | None:
    counts = {delimiter: header_line.count(delimiter) for delimiter in DELIMITER_CANDIDATES}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    return delimiter if count > 0 else None


def infer_dataset_family(file_name: str) -> str:
    upper_name = file_name.upper()
    for dataset in KNOWN_DATASETS:
        if upper_name.startswith(dataset):
            return dataset
    if "DELETE" in upper_name or "DELETED" in upper_name:
        return "DELETE"
    return "OTHER"


def classify_folder_variant(file_path: Path) -> str:
    parts_upper = [part.upper() for part in file_path.parts]
    if "ASCII" in parts_upper:
        for part in file_path.parts:
            if part.upper() == "ASCII":
                return part
    if "DELETED" in parts_upper:
        for part in file_path.parts:
            if part.upper() == "DELETED":
                return part
    return "ROOT"


def parse_header(file_path: Path) -> tuple[str | None, list[str], str | None]:
    try:
        with file_path.open("r", encoding="latin1", newline="") as handle:
            header_line = handle.readline().rstrip("\r\n")
    except OSError:
        return None, [], "read_error"

    if not header_line:
        return None, [], "empty_header"

    delimiter = detect_delimiter(header_line)
    if delimiter is None:
        return None, [], "delimiter_not_detected"

    columns = [clean_column_name(column) for column in header_line.split(delimiter)]
    return delimiter, columns, None


def scan_structure(raw_root: Path) -> tuple[list[dict], list[dict], dict]:
    file_rows: list[dict] = []
    column_rows: list[dict] = []

    years = sorted(
        path for path in raw_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{4}", path.name)
    )

    for year_path in years:
        for quarter_path in sorted(
            path for path in year_path.iterdir() if path.is_dir() and re.fullmatch(r"Q[1-4]", path.name, re.I)
        ):
            txt_files = sorted(quarter_path.rglob("*.txt"))
            for file_path in txt_files:
                delimiter, columns, issue = parse_header(file_path)
                dataset_family = infer_dataset_family(file_path.name)
                relative_path = file_path.relative_to(raw_root)
                folder_variant = classify_folder_variant(file_path)

                file_row = {
                    "year": int(year_path.name),
                    "quarter": quarter_path.name.upper(),
                    "relative_path": str(relative_path),
                    "file_name": file_path.name,
                    "dataset_family": dataset_family,
                    "folder_variant": folder_variant,
                    "size_bytes": file_path.stat().st_size,
                    "delimiter": delimiter or "",
                    "column_count": len(columns),
                    "header_issue": issue or "",
                    "header_signature": "|".join(column.lower() for column in columns),
                }
                file_rows.append(file_row)

                for position, column in enumerate(columns, start=1):
                    column_rows.append(
                        {
                            "year": int(year_path.name),
                            "quarter": quarter_path.name.upper(),
                            "relative_path": str(relative_path),
                            "file_name": file_path.name,
                            "dataset_family": dataset_family,
                            "column_position": position,
                            "column_name": column,
                            "column_name_normalized": column.lower(),
                        }
                    )

    summary = build_summary(file_rows, column_rows)
    return file_rows, column_rows, summary


def build_summary(file_rows: list[dict], column_rows: list[dict]) -> dict:
    folder_variants = Counter(row["folder_variant"] for row in file_rows)
    delimiter_counts = Counter(row["delimiter"] or "UNKNOWN" for row in file_rows)
    header_issues = [row for row in file_rows if row["header_issue"]]

    dataset_files_by_year: dict[int, list[str]] = defaultdict(list)
    for row in file_rows:
        year = row["year"]
        family = row["dataset_family"]
        if family not in dataset_files_by_year[year]:
            dataset_files_by_year[year].append(family)
    dataset_files_by_year = {
        year: sorted(families) for year, families in sorted(dataset_files_by_year.items())
    }

    schema_variants: dict[str, list[dict]] = {}
    for dataset_family in sorted({row["dataset_family"] for row in file_rows}):
        if dataset_family == "OTHER":
            continue
        related_rows = [row for row in file_rows if row["dataset_family"] == dataset_family and row["column_count"] > 0]
        signatures: dict[str, dict] = {}
        for row in related_rows:
            signature = row["header_signature"]
            if signature not in signatures:
                signatures[signature] = {
                    "column_count": row["column_count"],
                    "example_file": row["relative_path"],
                    "years": set(),
                    "quarters": set(),
                }
            signatures[signature]["years"].add(row["year"])
            signatures[signature]["quarters"].add(f"{row['year']}{row['quarter']}")

        schema_variants[dataset_family] = [
            {
                "column_count": value["column_count"],
                "example_file": value["example_file"],
                "years": sorted(value["years"]),
                "quarters": sorted(value["quarters"]),
            }
            for value in signatures.values()
        ]

    column_changes: dict[str, dict] = {}
    grouped_columns: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in column_rows:
        grouped_columns[row["dataset_family"]][row["year"]].add(row["column_name_normalized"])

    for dataset_family, year_map in grouped_columns.items():
        years = sorted(year_map)
        if not years:
            continue
        baseline_year = years[0]
        baseline_columns = year_map[baseline_year]
        changes = []
        for year in years[1:]:
            added = sorted(year_map[year] - baseline_columns)
            removed = sorted(baseline_columns - year_map[year])
            if added or removed:
                changes.append({"year": year, "added": added, "removed": removed})
        column_changes[dataset_family] = {
            "baseline_year": baseline_year,
            "baseline_columns": sorted(baseline_columns),
            "changes_vs_baseline": changes,
        }

    return {
        "folder_variants": dict(folder_variants),
        "delimiter_counts": dict(delimiter_counts),
        "header_issue_count": len(header_issues),
        "header_issues": header_issues,
        "dataset_files_by_year": dataset_files_by_year,
        "schema_variants": schema_variants,
        "column_changes": column_changes,
        "total_files": len(file_rows),
    }


def write_outputs(
    output_dir: Path,
    file_rows: list[dict],
    column_rows: list[dict],
    summary: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    file_df = pd.DataFrame(file_rows).sort_values(["year", "quarter", "relative_path"])
    column_df = pd.DataFrame(column_rows).sort_values(
        ["dataset_family", "year", "quarter", "relative_path", "column_position"]
    )

    file_csv = output_dir / "faers_structure_files.csv"
    columns_csv = output_dir / "faers_structure_columns.csv"
    summary_json = output_dir / "faers_structure_summary.json"
    report_md = output_dir / "faers_structure_report.md"

    file_df.to_csv(file_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    column_df.to_csv(columns_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(render_markdown_report(summary), encoding="utf-8")

    return {
        "file_csv": file_csv,
        "columns_csv": columns_csv,
        "summary_json": summary_json,
        "report_md": report_md,
    }


def render_markdown_report(summary: dict) -> str:
    lines = [
        "# FAERS Structure Scan Report",
        "",
        f"- Total TXT files scanned: {summary['total_files']}",
        f"- Header issues: {summary['header_issue_count']}",
        "",
        "## Folder Variants",
        "",
    ]

    for folder_variant, count in sorted(summary["folder_variants"].items()):
        lines.append(f"- {folder_variant}: {count}")

    lines.extend(["", "## Delimiters", ""])
    for delimiter, count in sorted(summary["delimiter_counts"].items()):
        label = repr(delimiter) if delimiter != "UNKNOWN" else "UNKNOWN"
        lines.append(f"- {label}: {count}")

    lines.extend(["", "## Dataset Families By Year", ""])
    for year, families in summary["dataset_files_by_year"].items():
        lines.append(f"- {year}: {', '.join(families)}")

    lines.extend(["", "## Schema Variants", ""])
    for dataset_family, variants in sorted(summary["schema_variants"].items()):
        lines.append(f"### {dataset_family}")
        if not variants:
            lines.append("- No parseable TXT headers found.")
            lines.append("")
            continue
        for index, variant in enumerate(variants, start=1):
            year_span = f"{variant['years'][0]}-{variant['years'][-1]}" if variant["years"] else "NA"
            lines.append(
                f"- Variant {index}: {variant['column_count']} columns, years {year_span}, "
                f"example `{variant['example_file']}`"
            )
        lines.append("")

    lines.extend(["## Column Changes Vs First Observed Year", ""])
    for dataset_family, details in sorted(summary["column_changes"].items()):
        lines.append(f"### {dataset_family}")
        changes = details["changes_vs_baseline"]
        if not changes:
            lines.append(f"- No year-level changes detected relative to {details['baseline_year']}.")
            lines.append("")
            continue
        for change in changes:
            added = ", ".join(change["added"]) if change["added"] else "None"
            removed = ", ".join(change["removed"]) if change["removed"] else "None"
            lines.append(
                f"- {change['year']} vs {details['baseline_year']}: added [{added}] ; removed [{removed}]"
            )
        lines.append("")

    if summary["header_issues"]:
        lines.extend(["## Header Issues", ""])
        for row in summary["header_issues"]:
            lines.append(f"- {row['relative_path']}: {row['header_issue']}")

    return "\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Scan FAERS raw data structure across years and quarters.")
    parser.add_argument(
        "--raw-root",
        default=RAW_ROOT,
        type=str,
        help="Raw FAERS root directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_OUTPUT_ROOT) / "structure_scan"),
        type=str,
        help="Directory for scan outputs",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    file_rows, column_rows, summary = scan_structure(raw_root)
    outputs = write_outputs(output_dir, file_rows, column_rows, summary)

    print("FAERS structure scan completed.")
    print(f"Scanned TXT files: {summary['total_files']}")
    print(f"Header issues: {summary['header_issue_count']}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
