from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from case_dataset_processor import process_case_dataset
from case_linked_table_processor import process_indi, process_rpsr, process_ther
from demo_processor import process_demo
from drug_exposure_processor import process_drug_exposure
from drug_feature_processor import process_drug_feature
from drug_processor import process_drug
from outc_processor import process_outc
from reac_processor import process_reac
from signal_dataset_processor import process_signal_dataset


Processor = Callable[[int, str, str | Path], object]


@dataclass(frozen=True)
class ProcessingStep:
    name: str
    description: str
    processor: Processor


PROCESSING_STEPS: tuple[ProcessingStep, ...] = (
    ProcessingStep("demo", "Clean DEMO and build case base data", process_demo),
    ProcessingStep("drug", "Clean DRUG records", process_drug),
    ProcessingStep("drug_feature", "Build case-level drug features", process_drug_feature),
    ProcessingStep("drug_exposure", "Build target drug exposure groups", process_drug_exposure),
    ProcessingStep("reac", "Build case-level reaction outcomes", process_reac),
    ProcessingStep("outc", "Build case-level serious outcome flags", process_outc),
    ProcessingStep("indi", "Clean case-linked indication records", process_indi),
    ProcessingStep("rpsr", "Clean case-linked report source records", process_rpsr),
    ProcessingStep("ther", "Clean case-linked therapy records", process_ther),
    ProcessingStep("case", "Build the main case-level dataset", process_case_dataset),
    ProcessingStep("signal", "Build the signal analysis dataset", process_signal_dataset),
)

STEP_BY_NAME = {step.name: step for step in PROCESSING_STEPS}
TABLE_CHOICES = tuple(STEP_BY_NAME) + ("all",)


def normalize_quarter(quarter: str) -> str:
    quarter = str(quarter).strip().upper()
    if quarter not in {"Q1", "Q2", "Q3", "Q4"}:
        raise ValueError(f"quarter must be one of Q1, Q2, Q3, Q4: {quarter!r}")
    return quarter


def normalize_table(table: str) -> str:
    table = str(table).strip().lower()
    if table not in TABLE_CHOICES:
        valid = ", ".join(TABLE_CHOICES)
        raise ValueError(f"table must be one of: {valid}")
    return table


def selected_steps(table: str) -> tuple[ProcessingStep, ...]:
    table = normalize_table(table)
    if table == "all":
        return PROCESSING_STEPS
    return (STEP_BY_NAME[table],)


def run_quarter_step(
    year: int,
    quarter: str,
    table: str,
    output_root: str | Path,
) -> list[str]:
    quarter = normalize_quarter(quarter)
    steps = selected_steps(table)
    for step in steps:
        step.processor(int(year), quarter, output_root)
    return [step.name for step in steps]


def run_full_quarter(year: int, quarter: str, output_root: str | Path) -> list[str]:
    return run_quarter_step(year, quarter, "all", output_root)
