# Full Period Analysis (2004-2025)

This folder is isolated from the existing yearly pipeline.

## What this workflow does

1. Reuse the already cleaned quarterly outputs in `OUTPUT/*/quarterly`.
2. Build a global case index across `2004-2025`.
3. Keep one latest record per `CASEID` using:
   - latest `FDA_DT`
   - if tied, larger `PRIMARYID`
4. Export global datasets into `OUTPUT_GLOBAL/datasets`:
   - `global_case_index_2004_2025.parquet`
   - `signal_dataset_2004_2025.parquet`
   - `drug_feature_2004_2025_case.parquet`

The global step does not reread raw `DELETE` files. Any quarter-level deletion/nullification handling is expected to have been completed upstream during quarterly cleaning.

## Run

```powershell
.\.venv\Scripts\python .\full_period_analysis\build_global_datasets.py --start-year 2004 --end-year 2025
.\.venv\Scripts\python .\full_period_analysis\run_global_analysis.py
```

## Output layout

- `OUTPUT_GLOBAL/datasets`: global parquet datasets
- `OUTPUT_GLOBAL/qc`: row-count and QC summary
- `OUTPUT_GLOBAL/analysis`: analysis csv outputs
