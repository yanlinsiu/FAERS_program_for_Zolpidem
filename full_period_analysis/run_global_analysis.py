from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from config import GLOBAL_OUTPUT_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis_projiect"

if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))


def _load_function(module_file: str, function_name: str):
    module_path = ANALYSIS_ROOT / module_file
    spec = importlib.util.spec_from_file_location(function_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


build_signal_analysis = _load_function("01_signal_analysis.py", "build_signal_analysis")
build_comparative_analysis = _load_function("02_comparative_analysis.py", "build_comparative_analysis")
build_feature_analysis = _load_function("03_feature_analysis.py", "build_feature_analysis")


def main() -> None:
    signal_root = GLOBAL_OUTPUT_ROOT / "datasets"
    output_dir = GLOBAL_OUTPUT_ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_signal_analysis(signal_root=signal_root, output_dir=output_dir)
    build_comparative_analysis(signal_root=signal_root, output_dir=output_dir)
    build_feature_analysis(signal_root=signal_root, output_dir=output_dir)

    print(f"Global analysis completed. Output directory: {output_dir}")


if __name__ == "__main__":
    main()

