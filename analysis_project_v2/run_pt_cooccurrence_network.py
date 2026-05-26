from __future__ import annotations

try:
    from .phenotypes.run_pt_cooccurrence_network import *  # noqa: F403
    from .phenotypes.run_pt_cooccurrence_network import main
except ImportError:
    from phenotypes.run_pt_cooccurrence_network import *  # noqa: F403
    from phenotypes.run_pt_cooccurrence_network import main


if __name__ == "__main__":
    main()
