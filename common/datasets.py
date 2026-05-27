from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetBundle:
    period_token: str
    signal_file: Path
    feature_file: Path
    feature_version: str = "v1"


def extract_token(path: Path, prefix: str, suffix: str = "") -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected dataset file name: {path.name}")
    token = stem[len(prefix) :]
    if suffix:
        if not token.endswith(suffix):
            raise ValueError(f"Unexpected dataset file name: {path.name}")
        token = token[: -len(suffix)]
    return token


def token_sort_key(token: str) -> tuple[int, int, int, str]:
    parts = token.split("_")
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        start_year = int(parts[0])
        end_year = int(parts[1])
        return (end_year - start_year, end_year, -start_year, token)
    return (0, 0, 0, token)


def resolve_signal_feature_bundle(
    dataset_dir: str | Path,
    period_token: str | None = None,
) -> DatasetBundle:
    dataset_path = Path(dataset_dir)
    signal_files = sorted(dataset_path.glob("signal_dataset_*.parquet"))
    feature_files = sorted(dataset_path.glob("drug_feature_*_case.parquet"))

    if not signal_files:
        raise FileNotFoundError(f"No signal_dataset_*.parquet files found in {dataset_path}")
    if not feature_files:
        raise FileNotFoundError(f"No drug_feature_*_case.parquet files found in {dataset_path}")

    signal_by_token = {extract_token(path, "signal_dataset_"): path for path in signal_files}
    feature_by_token = {
        extract_token(path, "drug_feature_", "_case"): path for path in feature_files
    }
    shared_tokens = sorted(set(signal_by_token) & set(feature_by_token))
    if not shared_tokens:
        raise RuntimeError(f"No matching signal/feature bundle found in {dataset_path}")

    selected_token = period_token or max(shared_tokens, key=token_sort_key)
    if selected_token not in signal_by_token or selected_token not in feature_by_token:
        raise FileNotFoundError(f"Period token not found in {dataset_path}: {selected_token}")

    return DatasetBundle(
        period_token=selected_token,
        signal_file=signal_by_token[selected_token],
        feature_file=feature_by_token[selected_token],
        feature_version="v1",
    )

