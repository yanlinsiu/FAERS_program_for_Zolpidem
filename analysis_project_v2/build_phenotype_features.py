from __future__ import annotations

try:
    from .phenotypes.build_phenotype_features import *  # noqa: F403
    from .phenotypes.build_phenotype_features import main
except ImportError:
    from phenotypes.build_phenotype_features import *  # noqa: F403
    from phenotypes.build_phenotype_features import main


if __name__ == "__main__":
    main()
