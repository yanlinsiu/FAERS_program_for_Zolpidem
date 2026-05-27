# analysis_project_v2

这个目录是现在推荐使用的正式分析入口。

## 主要入口

年度分析（由 `faers_project/year_batch_runner.py` 自动调用）：

```powershell
cd D:\program_FAERS
uv run python analysis_project_v2\annual_analysis.py --signal-root OUTPUT\2024\quarterly --output-dir OUTPUT\2024\analysis
```

Global 全周期分析：

```powershell
cd D:\program_FAERS\analysis_project_v2
uv run python run_all.py --period-token 2004_2025
```

只跑主分析：

```powershell
uv run python run_all.py --period-token 2004_2025 --skip-trend
```

只跑监管趋势分析：

```powershell
uv run python run_all.py --period-token 2004_2025 --skip-main
```

## 现在的代码分工

- `run_all.py`：总入口，负责串起主分析和趋势分析。
- `annual_analysis.py`：年度分析入口，承接旧版年度信号分析、对比分析和特征分析。
- `age_trend_analysis.py`：年龄趋势分析，承接旧版 `04_age_trend_analysis.py`。
- `run_analysis.py`：主分析，包括信号分析、敏感性分析、探索性分析和校正模型。
- `regulatory_trend_analysis.py`：监管事件前后和年度趋势分析。
- `config.py`：分析定义、结局定义、分层定义和阈值。
- `data.py`：读取和合并分析数据。
- `signal_metrics.py`：ROR、PRR、IC、EBGM、p 值、FDR 等公共指标。
- `adjusted_models.py`：校正模型。
- `report_tables.py`：论文表格整理。
- `country/`：国家/地区相关分析。
- `phenotypes/`：表型谱和 PT 共现网络相关分析。

## 兼容说明

根目录下仍保留这些旧入口：

- `country_analyze.py`
- `country_fall_distribution.py`
- `build_phenotype_features.py`
- `run_phenotype_spectrum.py`
- `run_pt_cooccurrence_network.py`

它们现在只是薄薄的一层转发，真正代码已经放进 `country/` 或 `phenotypes/`。这样以前的命令还能跑，同时代码也按功能放在一起。

## 输出位置

- 主分析：`D:\program_FAERS\OUTPUT_GLOBAL\analysis_v2`
- 监管趋势：`D:\program_FAERS\OUTPUT_GLOBAL_COUNTRY\regulatory_trend`
- 表型相关：`D:\program_FAERS\OUTPUT_GLOBAL\phenotypes` 和 `D:\program_FAERS\OUTPUT_GLOBAL\phenotype_analysis`
