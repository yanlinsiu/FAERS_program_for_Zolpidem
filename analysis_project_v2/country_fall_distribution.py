from __future__ import annotations

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from country_analyze import build_country_fall_distribution, main  # noqa: E402,F401


if __name__ == "__main__":
    main()
