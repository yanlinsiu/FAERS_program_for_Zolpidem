# Z-drug Dictionary

This folder stores the study drug dictionary used to normalize messy FAERS DRUG
names into active ingredients.

Main file:

- `zdrug_dictionary.csv`

Important columns:

- `raw_name`: name as seen in FAERS or external drug sources
- `clean_name`: normalized matching phrase used by the code
- `standard_name`: active ingredient used in analysis
- `role_in_study`: `target` for zolpidem and `comparator` for other Z-drugs
- `include`: `true` means the term can be used for automatic matching
- `evidence_level`: strength of the mapping evidence
- `evidence_source`: authority used to support the mapping
- `exclude_reason`: reason to keep a term out of automatic matching
- `dosage_form`: optional route/form note for formulations such as CR, sublingual, or spray

Maintenance rule:

For the main analysis, keep `include=true` only for terms supported by FDA/DailyMed
labels or NLM RxNorm/RxNav. FAERS-only spelling variants, regional brands, and
ambiguous mixed phrases should not be added to the main dictionary unless the
study protocol is updated to include a separate manually reviewed expanded
dictionary.
