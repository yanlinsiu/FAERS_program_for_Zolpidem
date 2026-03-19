from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
GLOBAL_OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT_GLOBAL"

GLOBAL_DATASET_DIR = GLOBAL_OUTPUT_ROOT / "datasets"
GLOBAL_QC_DIR = GLOBAL_OUTPUT_ROOT / "qc"

