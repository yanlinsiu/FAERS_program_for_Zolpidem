# mainline_2026-05-20 结果目录

这个目录用于保存本轮重构后重新生成的结果。这样做的好处是：旧的 `OUTPUT*` 和
`analysis_reports` 不会被覆盖，方便新旧结果互相比对。

## 目录说明

```text
runs/mainline_2026-05-20/
|-- OUTPUT/             # 新的清洗结果：年度、季度 parquet
|-- OUTPUT_GLOBAL/      # 新的全周期合并结果和统计分析结果
|-- OUTPUT_ML/          # 新的 ML-v2 特征和模型结果
|-- analysis_reports/   # 新的报告结果，例如国家分布分析
|-- logs/               # 建议保存命令日志
|-- qc/                 # 预留给人工质控汇总
`-- notes/              # 预留给过程记录
```

## 推荐运行命令

在项目根目录 `D:\program_FAERS` 运行：

```powershell
$RUN_ROOT = "D:\program_FAERS\runs\mainline_2026-05-20"

python faers_project\year_batch_runner.py `
  --start-year 2004 `
  --end-year 2025 `
  --output-root "$RUN_ROOT\OUTPUT" `
  --force

python full_period_analysis\build_global_datasets.py `
  --start-year 2004 `
  --end-year 2025 `
  --input-output-root "$RUN_ROOT\OUTPUT" `
  --global-output-root "$RUN_ROOT\OUTPUT_GLOBAL"

python analysis_project_v2\run_analysis.py `
  --period-token 2004_2025 `
  --dataset-dir "$RUN_ROOT\OUTPUT_GLOBAL\datasets" `
  --output-dir "$RUN_ROOT\OUTPUT_GLOBAL\analysis_v2"

python analysis_project_v2\country_analyze.py `
  --start-year 2004 `
  --end-year 2025 `
  --input-root "$RUN_ROOT\OUTPUT" `
  --global-index-file "$RUN_ROOT\OUTPUT_GLOBAL\datasets\global_case_index_2004_2025.parquet" `
  --output-dir "$RUN_ROOT\analysis_reports\country_analysis"

python ml_project\features_v2\run_pipeline.py `
  --start-year 2004 `
  --end-year 2025 `
  --cleaned-output-root "$RUN_ROOT\OUTPUT" `
  --global-dataset-dir "$RUN_ROOT\OUTPUT_GLOBAL\datasets" `
  --ml-output-root "$RUN_ROOT\OUTPUT_ML"
```

跑 ML 模型前，先在同一个 PowerShell 窗口设置输出路径：

```powershell
$env:FAERS_GLOBAL_DATASET_DIR = "$RUN_ROOT\OUTPUT_GLOBAL\datasets"
$env:FAERS_ML_OUTPUT_ROOT = "$RUN_ROOT\OUTPUT_ML"

python ml_project\01_logistic_regression.py --feature-version v2
python ml_project\02_random_forest.py --feature-version v2
python ml_project\03_xgboost.py --feature-version v2
```

## 回滚方式

如果这轮结果不满意，不需要删除旧结果。继续使用项目根目录下原来的 `OUTPUT`、
`OUTPUT_GLOBAL`、`OUTPUT_ML` 和 `analysis_reports` 即可。
