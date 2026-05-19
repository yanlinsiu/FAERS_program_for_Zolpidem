# FAERS country-field rebuild

- Created at: 2026-05-19 18:57:40
- Year range: 2004-2025
- Cleaned output root: `D:\program_FAERS\OUTPUT_COUNTRY`
- Global output root: `D:\program_FAERS\OUTPUT_GLOBAL_COUNTRY`
- Force rebuild: `False`

## Purpose

This rebuild keeps DEMO country fields in the cleaned case-level datasets.

Newly retained fields:

- `reporter_country`: country of report source
- `occr_country`: country where the adverse event occurred

The original `OUTPUT` and `OUTPUT_GLOBAL` directories are not used as write targets by this script.
