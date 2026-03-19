from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _looks_like_faers_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any((path / str(year)).is_dir() for year in range(2004, 2026))


def _resolve_raw_root() -> Path:
    env_value = os.environ.get("FAERS_RAW_ROOT", "").strip()
    if env_value:
        candidate = Path(env_value).expanduser()
        if _looks_like_faers_root(candidate):
            return candidate
        if _looks_like_faers_root(candidate / "data"):
            return candidate / "data"
        return candidate

    data_dir = PROJECT_ROOT / "data"
    if _looks_like_faers_root(data_dir):
        return data_dir

    if _looks_like_faers_root(PROJECT_ROOT):
        return PROJECT_ROOT

    return data_dir


RAW_ROOT_PATH = _resolve_raw_root()
DEFAULT_OUTPUT_ROOT_PATH = PROJECT_ROOT / "OUTPUT"

# Keep string aliases for compatibility with existing code.
RAW_ROOT = str(RAW_ROOT_PATH)
DEFAULT_OUTPUT_ROOT = str(DEFAULT_OUTPUT_ROOT_PATH)

# Aliases (backward compatibility)
RAW_PATH = RAW_ROOT
OUTPUT_PATH = DEFAULT_OUTPUT_ROOT
