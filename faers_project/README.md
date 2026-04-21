# FAERS 项目说明

这个目录是一套面向 FAERS 原始季度数据的处理脚本。它的目标是把分表文本数据整理成病例级分析数据集，并进一步生成用于信号分析和年度汇总分析的结果文件。

当前代码围绕以下研究目标展开：

- 识别老年病例
- 识别 `zolpidem` 及其他 `Z-drug` 暴露
- 构建跌倒相关结局
- 构建病例级主分析表
- 构建信号分析表
- 生成年度汇总与跨年度趋势结果

## 项目结构

建议配套目录结构如下：

```text
program_FAERS/
|-- data/
|   |-- 2024/
|   |   |-- Q1/
|   |   |   |-- ASCII/
|   |   |   |   |-- DEMO24Q1.txt
|   |   |   |   |-- DRUG24Q1.txt
|   |   |   |   |-- REAC24Q1.txt
|   |   |   |   `-- OUTC24Q1.txt
|-- OUTPUT/
|-- analysis_project/
|   |-- 01_signal_analysis.py
|   |-- 02_comparative_analysis.py
|   `-- 03_feature_analysis.py
`-- faers_project/
    |-- main.py
    |-- year_batch_runner.py
    `-- README.md
```

当前代码中的默认路径规则：

- 原始数据根目录：优先读取环境变量 `FAERS_RAW_ROOT`
- 如果未设置环境变量，则优先尝试项目根目录下的 `data/`
- 默认输出目录：项目根目录下的 `OUTPUT/`
- 年度分析脚本目录：项目根目录下的 `analysis_project/`

如需显式指定原始数据目录，可以在 PowerShell 中设置：

```powershell
$env:FAERS_RAW_ROOT="D:\program_FAERS\data"
```

## 原始数据要求

脚本假定 FAERS 原始数据按“年份/季度”组织，并在季度目录下递归查找对应文本文件。

当前主要处理以下表：

- `DEMO`
- `DRUG`
- `REAC`
- `OUTC`

代码里已经兼容了一些历史差异，包括：

- `ASCII` / `ascii` 目录名差异
- 文件名大小写差异
- 某些早期季度 `REAC` 或 `OUTC` 缺少 `caseid` 时，自动从 `DEMO` 回填
- 删除病例目录存在时，自动剔除对应 `caseid`
- 文本分隔或列名存在历史差异时，由 `utils.py` 中的读取逻辑处理

## 整体处理流程

主入口是 [main.py](D:/program_FAERS/faers_project/main.py)。

季度级处理链路大致如下：

1. `demo_processor.py`
   - 读取 `DEMO`
   - 剔除 deleted reports
   - 按 `caseid` 去重，仅保留最新版本病例
   - 标准化年龄和性别
   - 默认仅保留 `65-120` 岁、性别为 `M/F` 的病例
   - 生成 `case_base_dataset_YYYYqN.parquet`

2. `drug_processor.py`
   - 读取 `DRUG`
   - 仅保留 DEMO 保留病例对应记录
   - 清洗 `drugname`、`prod_ai`、`role_cod`
   - 生成 `drug_YYYYqN.parquet`

3. `drug_feature_processor.py`
   - 在病例级识别 `zolpidem`、其他 `Z-drug`、苯二氮卓、抗抑郁药、抗精神病药、阿片类、抗癫痫药
   - 生成 `drug_n`、`distinct_drug_n`、`polypharmacy_5`
   - 生成 `drug_feature_dataset_YYYYqN.parquet`

4. `drug_exposure_processor.py`
   - 基于 `role_cod` 定义研究暴露
   - 主分析口径：`PS + SS`
   - 敏感性分析口径：`PS only`
   - 输出 `target_drug_group` 与 `target_drug_group_ps`
   - 生成 `drug_exposure_YYYYqN_case.parquet`

5. `reac_processor.py`
   - 从 `REAC` 构建病例级跌倒结局
   - 包含狭义跌倒和广义跌倒相关定义
   - 生成 `reac_YYYYqN_case.parquet`

6. `outc_processor.py`
   - 从 `OUTC` 构建死亡、住院、危及生命等严重结局标记
   - 生成 `outcome_dataset_YYYYqN.parquet`
   - 同时保留兼容文件名 `outc_YYYYqN_case.parquet`

7. `case_dataset_processor.py`
   - 合并 `DEMO`、`REAC`、`DRUG feature`、`DRUG exposure`、`OUTC`
   - 形成病例级主分析表
   - 生成 `case_dataset_YYYYqN.parquet`

8. `signal_dataset_processor.py`
   - 提取信号分析所需字段
   - 形成用于后续 ROR / PRR / 2x2 计算的病例级数据集
   - 生成 `signal_dataset_YYYYqN.parquet`

说明：

- `case` 和 `signal` 两个步骤支持自动补跑前置依赖
- `year_batch_runner.py` 会按年顺序执行季度处理，并调用 `analysis_project` 中的年度分析脚本

## 主要产物

按季度运行时，常见输出包括：

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

按年运行时，还会额外生成：

- 年度合并后的 `parquet`
- `analysis/01_signal_analysis_results.csv`
- `analysis/01_signal_analysis_qc.csv`
- `analysis/02_comparative_analysis_results.csv`
- `analysis/02_comparative_analysis_qc.csv`
- `analysis/03_feature_analysis_results.csv`
- `analysis/03_feature_analysis_qc.csv`
- `analysis/quarter_summary_YYYY.csv`
- `analysis/summary_YYYY.txt`
- 跨年份趋势文件 `trend_START_END.csv`

## 环境依赖

上层 [pyproject.toml](D:/program_FAERS/pyproject.toml) 当前声明的依赖为：

- `duckdb`
- `pandas`
- `pyarrow`
- `scipy`
- `scikit-learn`
- `xgboost`

Python 版本要求：

- `Python >= 3.11`

如果使用 `uv`：

```powershell
cd D:\program_FAERS
uv sync
```

如果使用 `pip`，最少建议先安装：

```powershell
pip install pandas pyarrow scipy duckdb scikit-learn xgboost
```

## 使用方法

### 1. 处理单张表

```powershell
cd D:\program_FAERS\faers_project
python main.py --year 2024 --quarter Q1 --table demo
python main.py --year 2024 --quarter Q1 --table drug
python main.py --year 2024 --quarter Q1 --table reac
python main.py --year 2024 --quarter Q1 --table outc
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

### 2. 处理单个季度的完整流程

```powershell
python main.py --year 2024 --quarter Q1 --table all
```

这个命令会依次执行：

- `demo`
- `drug`
- `drug_feature`
- `drug_exposure`
- `reac`
- `outc`
- `case`
- `signal`

### 3. 只构建病例级主分析表

```powershell
python main.py --year 2024 --quarter Q1 --table case
```

如果依赖文件缺失，脚本会自动补跑前置步骤。

### 4. 只构建信号分析表

```powershell
python main.py --year 2024 --quarter Q1 --table signal
```

如果依赖文件缺失，脚本也会自动补跑前置步骤。

### 5. 自定义输出目录

```powershell
python main.py --year 2024 --quarter Q1 --table all --output D:\program_FAERS\OUTPUT
```

## 按年批处理

处理单年：

```powershell
python year_batch_runner.py --year 2024
```

处理年份区间：

```powershell
python year_batch_runner.py --start-year 2019 --end-year 2024
```

常用参数：

- `--verbose`：显示季度内每一步的详细日志
- `--force`：即使年度结果已存在，也强制重建
- `--output-root`：指定年度输出根目录

年度批处理会做三件事：

1. 自动识别该年可用季度
2. 为每个季度跑完整处理流程
3. 汇总季度结果，并调用 `analysis_project` 中的年度分析脚本

## 年度分析内容

[year_batch_runner.py](D:/program_FAERS/faers_project/year_batch_runner.py) 会在季度产物完成后继续调用 [analysis_project](D:/program_FAERS/analysis_project) 中的分析脚本：

- [01_signal_analysis.py](D:/program_FAERS/analysis_project/01_signal_analysis.py)
  - 比较 `zolpidem suspect` 与其他病例之间的信号强度
- [02_comparative_analysis.py](D:/program_FAERS/analysis_project/02_comparative_analysis.py)
  - 比较 `zolpidem_only` 与 `other_zdrug_only`
- [03_feature_analysis.py](D:/program_FAERS/analysis_project/03_feature_analysis.py)
  - 在 `zolpidem` 暴露病例内部做分层特征分析

当前代码里，信号分析结果会写出 ROR、PRR，以及部分 IC / EBGM 相关字段。

## 关键研究设定

当前版本里比较重要的默认设定如下：

- 研究对象默认仅保留 `65-120` 岁病例
- 性别默认仅保留 `M` 和 `F`
- 主分析暴露口径为 `PS + SS`
- 敏感性分析暴露口径为 `PS only`
- 多药并用定义为 `distinct_drug_n >= 5`
- `REAC` 中同时构建 `is_fall_narrow` 和 `is_fall_broad`
- `signal_dataset` 中保留年龄组、性别、严重结局和暴露分组字段

这些设定都属于研究口径的一部分，后续如果改代码，建议优先同步更新文档。

## 辅助脚本

- [structure_scan.py](D:/program_FAERS/faers_project/structure_scan.py)
  - 扫描原始 FAERS 文件结构、字段变化和分隔情况
- [explore_zolpidem_pt.py](D:/program_FAERS/faers_project/explore_zolpidem_pt.py)
  - 统计 `zolpidem` 病例中的 PT 分布
- [descriptive_total_report.py](D:/program_FAERS/faers_project/descriptive_total_report.py)
  - 生成描述性汇总报告
- [check.py](D:/program_FAERS/faers_project/check.py)
  - 做快速人工检查或结果抽查

## 常见问题

### 1. 找不到原始文件

先确认目录结构是否满足：

```text
data/年份/季度/.../*.txt
```

如果不在默认目录，请设置 `FAERS_RAW_ROOT`。

### 2. 某些季度缺少字段或 `caseid`

FAERS 历史文件结构并不完全一致。这个项目已经在 [utils.py](D:/program_FAERS/faers_project/utils.py) 中兼容了一部分历史差异，但遇到更老或更特殊的季度时，建议先运行 [structure_scan.py](D:/program_FAERS/faers_project/structure_scan.py) 检查实际结构。

### 3. 年度批处理跑不起来

请确认以下目录都存在且内容完整：

- `D:\program_FAERS\data`
- `D:\program_FAERS\OUTPUT`
- `D:\program_FAERS\analysis_project`

尤其是 `year_batch_runner.py` 依赖 `analysis_project` 中的 3 个年度分析脚本，不能只复制 `faers_project` 单独运行。

## 建议阅读顺序

如果后面需要快速重新理解这套代码，推荐按这个顺序看：

1. [main.py](D:/program_FAERS/faers_project/main.py)
2. [config.py](D:/program_FAERS/faers_project/config.py)
3. [utils.py](D:/program_FAERS/faers_project/utils.py)
4. [demo_processor.py](D:/program_FAERS/faers_project/demo_processor.py)
5. [drug_processor.py](D:/program_FAERS/faers_project/drug_processor.py)
6. [drug_feature_processor.py](D:/program_FAERS/faers_project/drug_feature_processor.py)
7. [drug_exposure_processor.py](D:/program_FAERS/faers_project/drug_exposure_processor.py)
8. [reac_processor.py](D:/program_FAERS/faers_project/reac_processor.py) 和 [outc_processor.py](D:/program_FAERS/faers_project/outc_processor.py)
9. [case_dataset_processor.py](D:/program_FAERS/faers_project/case_dataset_processor.py)
10. [signal_dataset_processor.py](D:/program_FAERS/faers_project/signal_dataset_processor.py)
11. [year_batch_runner.py](D:/program_FAERS/faers_project/year_batch_runner.py)

这样基本能最快把数据处理链路和分析入口重新串起来。
