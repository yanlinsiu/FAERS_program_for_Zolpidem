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
- `exclude_reason`: reason to keep a term out of automatic matching
- `dosage_form`: optional route/form note for formulations such as CR, sublingual, or spray

Maintenance rule:

Add new FAERS spellings here first, then rerun the processing pipeline. Avoid
adding ambiguous mixed phrases as `include=true`; keep them in the table with an
`exclude_reason` so the review decision is visible.
