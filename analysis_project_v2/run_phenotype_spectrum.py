from __future__ import annotations

try:
    from .phenotypes.run_phenotype_spectrum import *  # noqa: F403
    from .phenotypes.run_phenotype_spectrum import main
except ImportError:
    from phenotypes.run_phenotype_spectrum import *  # noqa: F403
    from phenotypes.run_phenotype_spectrum import main


if __name__ == "__main__":
    main()
