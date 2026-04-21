# FAERS Program for Zolpidem

基于 FAERS 原始季度数据的清洗、病例级数据集构建、年度信号分析、全周期汇总分析与机器学习建模项目。

当前代码围绕 `zolpidem` 及其他 `Z-drug` 暴露展开，重点用于老年病例中的跌倒相关结局识别、信号检测和分层分析。

## 项目内容

项目大致分为 4 个层次：

1. `faers_project/`
   按季度处理原始 `DEMO / DRUG / REAC / OUTC` 文本，生成病例级分析数据。
2. `analysis_project/`
   对季度或年度数据执行信号分析、比较分析和特征分析。
3. `full_period_analysis/`
   将多个年份的季度产物汇总成全周期去重数据集，并运行全周期分析。
4. `ml_project/`
   基于全周期数据集进行逻辑回归、随机森林和 XGBoost 建模。

## 目录结构

```text
program_FAERS/
|-- data/                      # 原始 FAERS 数据，按年份/季度存放
|-- faers_project/             # 季度级清洗与数据集构建
|-- analysis_project/          # 年度/季度分析脚本
|-- full_period_analysis/      # 全周期汇总与分析
|-- ml_project/                # 机器学习建模
|-- OUTPUT/                    # 季度与年度输出
|-- OUTPUT_GLOBAL/             # 全周期输出
|-- OUTPUT_ML/                 # 机器学习输出
|-- pyproject.toml
`-- README.md
```

建议的原始数据目录形式如下：

```text
data/
`-- 2024/
    `-- Q1/
        `-- ASCII/
            |-- DEMO24Q1.txt
            |-- DRUG24Q1.txt
            |-- REAC24Q1.txt
            `-- OUTC24Q1.txt
```

## 环境要求

- Python `>= 3.11`
- 建议使用 `uv`

项目当前依赖见 [pyproject.toml](/D:/program_FAERS/pyproject.toml)：

- `duckdb`
- `pandas`
- `pyarrow`
- `scipy`
- `scikit-learn`
- `xgboost`

安装方式：

```powershell
cd D:\program_FAERS
uv sync
```

如果不用 `uv`，至少需要安装：

```powershell
pip install duckdb pandas pyarrow scipy scikit-learn xgboost
```

## 数据路径配置

[faers_project/config.py](/D:/program_FAERS/faers_project/config.py) 会按以下顺序寻找原始数据目录：

1. 环境变量 `FAERS_RAW_ROOT`
2. 项目根目录下的 `data/`
3. 项目根目录本身

默认输出目录为：

- 季度/年度输出：`D:\program_FAERS\OUTPUT`
- 全周期输出：`D:\program_FAERS\OUTPUT_GLOBAL`
- 机器学习输出：`D:\program_FAERS\OUTPUT_ML`

如需显式指定原始数据目录：

```powershell
$env:FAERS_RAW_ROOT="D:\program_FAERS\data"
```

## 快速开始

### 1. 处理单个季度

进入季度处理目录后运行：

```powershell
cd D:\program_FAERS\faers_project
python main.py --year 2024 --quarter Q1 --table all
```

支持的 `--table` 取值：

- `demo`
- `drug`
- `drug_feature`
- `drug_exposure`
- `reac`
- `outc`
- `case`
- `signal`
- `all`

常见示例：

```powershell
python main.py --year 2024 --quarter Q1 --table demo
python main.py --year 2024 --quarter Q1 --table case
python main.py --year 2024 --quarter Q1 --table signal
python main.py --year 2024 --quarter Q1 --table all --output D:\program_FAERS\OUTPUT
```

### 2. 按年批处理

[faers_project/year_batch_runner.py](/D:/program_FAERS/faers_project/year_batch_runner.py) 会自动识别该年份可用季度，依次完成季度处理、年度合并和年度分析。

```powershell
cd D:\program_FAERS\faers_project
python year_batch_runner.py --year 2024
python year_batch_runner.py --start-year 2019 --end-year 2024
```

常用参数：

- `--verbose`：显示每一步详细日志
- `--force`：即使年度结果已存在也强制重建
- `--output-root`：自定义年度输出根目录

### 3. 构建全周期数据集

[full_period_analysis/build_global_datasets.py](/D:/program_FAERS/full_period_analysis/build_global_datasets.py) 会从 `OUTPUT/*/quarterly/` 中读取清洗后的季度产物，构建去重后的全周期病例索引、全周期信号数据集和全周期特征数据集。

```powershell
cd D:\program_FAERS\full_period_analysis
python build_global_datasets.py --start-year 2004 --end-year 2025
```

### 4. 运行全周期分析

```powershell
cd D:\program_FAERS\full_period_analysis
python run_global_analysis.py --period-token 2004_2025
```

如果 `OUTPUT_GLOBAL/datasets/` 下只有一套数据，也可以省略 `--period-token`。

### 5. 运行机器学习建模

`ml_project/` 使用全周期 `signal_dataset_*.parquet` 与 `drug_feature_*_case.parquet` 进行建模，输出单独写入 `OUTPUT_ML/`。

可用脚本包括：

- [01_logistic_regression.py](/D:/program_FAERS/ml_project/01_logistic_regression.py)
- [02_random_forest.py](/D:/program_FAERS/ml_project/02_random_forest.py)
- [03_xgboost.py](/D:/program_FAERS/ml_project/03_xgboost.py)

## 季度处理流程

[faers_project/main.py](/D:/program_FAERS/faers_project/main.py) 的主处理链路如下：

1. `demo_processor.py`
   从 `DEMO` 提取病例基础信息，去除删除病例，按 `caseid` 去重，清洗年龄和性别。
2. `drug_processor.py`
   清洗 `DRUG` 表，仅保留研究病例对应记录。
3. `drug_feature_processor.py`
   构建 `zolpidem`、其他 `Z-drug`、苯二氮卓类、抗抑郁药、阿片类等特征，以及多重用药指标。
4. `drug_exposure_processor.py`
   基于 `role_cod` 定义研究暴露，区分 `PS + SS` 与 `PS only`。
5. `reac_processor.py`
   从 `REAC` 构建狭义和广义跌倒相关结局。
6. `outc_processor.py`
   从 `OUTC` 构建死亡、住院、危及生命等严重结局变量。
7. `case_dataset_processor.py`
   合并季度病例级主分析表。
8. `signal_dataset_processor.py`
   生成信号分析使用的数据集。

## 主要输出

### 季度级输出

常见文件包括：

- `case_base_dataset_YYYYqN.parquet`
- `demo_YYYYqN.parquet`
- `drug_YYYYqN.parquet`
- `drug_feature_dataset_YYYYqN.parquet`
- `drug_feature_YYYYqN_case.parquet`
- `drug_exposure_YYYYqN_case.parquet`
- `reac_YYYYqN_case.parquet`
- `outcome_dataset_YYYYqN.parquet`
- `outc_YYYYqN_case.parquet`
- `case_dataset_YYYYqN.parquet`
- `signal_dataset_YYYYqN.parquet`

### 年度级输出

按年批处理后，通常会在 `OUTPUT/<year>/` 下看到：

- `quarterly/`：该年各季度产物
- `analysis/01_signal_analysis_results.csv`
- `analysis/02_comparative_analysis_results.csv`
- `analysis/03_feature_analysis_results.csv`
- `analysis/quarter_summary_<year>.csv`
- `analysis/summary_<year>.txt`

项目根 `OUTPUT/` 下还可能出现跨年趋势文件，例如：

- `trend_2019_2025.csv`
- `trend_2004_2025.csv`

### 全周期输出

`OUTPUT_GLOBAL/` 主要包括：

- `datasets/global_case_index_<start>_<end>.parquet`
- `datasets/signal_dataset_<start>_<end>.parquet`
- `datasets/drug_feature_<start>_<end>_case.parquet`
- `qc/global_dataset_qc_<start>_<end>.csv`
- `qc/global_signal_summary_<start>_<end>.csv`
- `analysis/` 下的全周期分析结果

### 机器学习输出

`OUTPUT_ML/` 下按模型分类保存运行结果，例如：

- `logistic_regression/`
- `random_forest/`
- `xgboost/`

## 分析内容

[analysis_project/](/D:/program_FAERS/analysis_project) 当前主要包含：

- [01_signal_analysis.py](/D:/program_FAERS/analysis_project/01_signal_analysis.py)
  计算 ROR、PRR，并补充 IC、EBGM 等信号指标。
- [02_comparative_analysis.py](/D:/program_FAERS/analysis_project/02_comparative_analysis.py)
  比较 `zolpidem_only` 与 `other_zdrug_only`。
- [03_feature_analysis.py](/D:/program_FAERS/analysis_project/03_feature_analysis.py)
  在暴露病例中做分层或特征层面的比较分析。
- [04_age_trend_analysis.py](/D:/program_FAERS/analysis_project/04_age_trend_analysis.py)
  进行年龄趋势相关分析。

## 当前研究口径

从代码现状看，当前默认研究设置大致包括：

- 重点关注老年病例，默认年龄范围为 `65-120`
- 默认保留性别为 `M / F` 的病例
- 主分析暴露定义为 `PS + SS`
- 敏感性分析暴露定义为 `PS only`
- 跌倒结局同时包含狭义和广义定义
- 多重用药常用阈值为 `distinct_drug_n >= 5`

这些属于研究口径的一部分，若后续调整代码逻辑，建议同步更新文档与分析解释。

## 辅助脚本

`faers_project/` 下还有一些实用脚本：

- [structure_scan.py](/D:/program_FAERS/faers_project/structure_scan.py)
  扫描不同年份季度的原始文件结构与字段差异。
- [explore_zolpidem_pt.py](/D:/program_FAERS/faers_project/explore_zolpidem_pt.py)
  统计 `zolpidem` 相关病例的 PT 分布。
- [descriptive_total_report.py](/D:/program_FAERS/faers_project/descriptive_total_report.py)
  生成描述性统计报告。
- [check.py](/D:/program_FAERS/faers_project/check.py)
  快速检查处理结果。

## 常见问题

### 找不到原始数据

先检查目录是否满足 `data/<year>/<quarter>/.../*.txt` 形式；如果原始数据不在默认位置，请设置 `FAERS_RAW_ROOT`。

### 某些历史季度字段不一致

FAERS 历史文本结构并不完全统一。项目已在 [faers_project/utils.py](/D:/program_FAERS/faers_project/utils.py) 中兼容部分差异，但遇到更早年份或特殊季度时，建议先运行 `structure_scan.py` 检查字段结构。

### 年度脚本跑不起来

`year_batch_runner.py` 依赖：

- 原始数据目录可读
- `analysis_project/` 中的分析脚本存在
- 季度产物可以正常写入 `OUTPUT/<year>/quarterly/`

如果此前已有损坏的 parquet 文件，也可能在年度合并时报错。

## 推荐阅读顺序

如果需要快速理解代码，建议按下面顺序阅读：

1. [faers_project/main.py](/D:/program_FAERS/faers_project/main.py)
2. [faers_project/config.py](/D:/program_FAERS/faers_project/config.py)
3. [faers_project/utils.py](/D:/program_FAERS/faers_project/utils.py)
4. `faers_project/*_processor.py`
5. [faers_project/year_batch_runner.py](/D:/program_FAERS/faers_project/year_batch_runner.py)
6. [analysis_project/analysis_common.py](/D:/program_FAERS/analysis_project/analysis_common.py)
7. [full_period_analysis/build_global_datasets.py](/D:/program_FAERS/full_period_analysis/build_global_datasets.py)
8. `ml_project/*.py`
