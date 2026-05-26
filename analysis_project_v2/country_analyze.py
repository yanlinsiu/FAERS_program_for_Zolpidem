from __future__ import annotations

try:
    from .country.country_analyze import *  # noqa: F403
    from .country.country_analyze import main
except ImportError:
    from country.country_analyze import *  # noqa: F403
    from country.country_analyze import main


if __name__ == "__main__":
    main()
