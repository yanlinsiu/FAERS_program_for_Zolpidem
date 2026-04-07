# FAERS 数据处理说明

这个目录是一套面向 FAERS 原始季度数据的处理脚本，作用是把分表文本数据整理成病例级分析数据，并进一步生成药物警戒信号分析所需的数据集和年度汇总结果。

这套代码当前围绕以下研究目标展开：

- 识别老年病例
- 识别 zolpidem 与其他 Z-drug 暴露
- 构建跌倒相关结局
- 构建病例级主分析表
- 构建信号分析表
- 生成年度汇总与年度分析结果

## 项目结构

这份代码默认自己位于整个项目仓库中的一个子目录里。按照当前代码的写法，建议目录结构如下：

```text
program_FAERS/
├─ data/
│  ├─ 2024/
│  │  ├─ Q1/
│  │  │  └─ ASCII/
│  │  │     ├─ DEMO24Q1.txt
│  │  │     ├─ DRUG24Q1.txt
│  │  │     ├─ REAC24Q1.txt
│  │  │     └─ OUTC24Q1.txt
├─ OUTPUT/
├─ analysis_project/
│  ├─ 01_signal_analysis.py
│  ├─ 02_comparative_analysis.py
│  └─ 03_feature_analysis.py
└─ faers_project/
   ├─ main.py
   ├─ year_batch_runner.py
   └─ README.md
```

代码中的默认路径规则如下：

- 原始数据目录：`D:\program_FAERS\data`
- 输出目录：`D:\program_FAERS\OUTPUT`
- 年度分析脚本目录：`D:\program_FAERS\analysis_project`

如果原始数据不在默认位置，可以设置环境变量：

```powershell
$env:FAERS_RAW_ROOT="D:\program_FAERS\data"
```

## 原始数据要求

代码假定原始数据按“年份 / 季度”组织，并在季度目录中递归查找 `.txt` 文件。

目前主要处理以下表：

- `DEMO`
- `DRUG`
- `REAC`
- `OUTC`

代码里已经对一些历史差异做了兼容，包括：

- `ASCII` / `ascii` 目录名差异
- 文件名大小写差异
- 早期季度部分表缺少 `caseid` 时，从 `DEMO` 回填
- 删除病例目录存在时，自动剔除对应 `caseid`
- 某些文本分隔异常时，使用 `utils.py` 中的备用解析逻辑

## 整体处理流程

主入口是 [main.py](D:/program_FAERS/faers_project/main.py)。

整个处理链路大致如下：

1. `demo_processor.py`
   - 读取 `DEMO`
   - 去除 deleted case
   - 以 `caseid` 去重，保留最新版本
   - 标准化年龄和性别
   - 只保留 `65-120` 岁且性别为 `M/F` 的病例
2. `drug_processor.py`
   - 读取 `DRUG`
   - 只保留保留后的 DEMO 病例对应记录
   - 清洗 `drugname`、`prod_ai`、`role_cod`
3. `drug_feature_processor.py`
   - 在病例级识别 `zolpidem`、其他 `Z-drug`、苯二氮卓、抗抑郁药、抗精神病药、阿片类等
   - 生成 `drug_n`、`distinct_drug_n`、`polypharmacy_5`
4. `drug_exposure_processor.py`
   - 根据 `role_cod` 定义研究暴露
   - 主分析口径：`PS + SS`
     - PS = Primary Suspect，主要怀疑药物
     - SS = Secondary Suspect，次要怀疑药物
   - 敏感性分析口径：`PS only`
     - 只有主要怀疑药物才算进暴露，SS 不算
5. `reac_processor.py`
   - 从 `REAC` 生成病例级跌倒结局
   - 包括狭义跌倒和广义跌倒相关定义
6. `outc_processor.py`
   - 从 `OUTC` 生成死亡、住院、危及生命等严重结局标记
7. `case_dataset_processor.py`
   - 合并 DEMO、REAC、DRUG 特征、DRUG 暴露、OUTC
   - 形成病例级主分析表
8. `signal_dataset_processor.py`
   - 形成后续 ROR / PRR 分析使用的病例级信号数据集

## 主要产物

按季度运行时，主要会生成这些文件：

- `case_base_dataset_YYYYqN.parquet`
  - DEMO 处理后的病例主表
- `demo_YYYYqN.parquet`
  - 旧命名保留文件，内容与病例主表兼容
- `drug_YYYYqN.parquet`
  - 清洗后的 DRUG 明细表
- `drug_feature_dataset_YYYYqN.parquet`
  - 病例级药物特征表
- `drug_feature_YYYYqN_case.parquet`
  - 旧命名保留文件
- `drug_exposure_YYYYqN_case.parquet`
  - 病例级研究暴露定义表
- `reac_YYYYqN_case.parquet`
  - 病例级跌倒结局表
- `outcome_dataset_YYYYqN.parquet`
  - 病例级严重结局表
- `outc_YYYYqN_case.parquet`
  - 旧命名保留文件
- `case_dataset_YYYYqN.parquet`
  - 病例级主分析表
- `signal_dataset_YYYYqN.parquet`
  - 信号分析表

按年度运行时，还会生成：

- 年度合并后的 `parquet`
- `analysis/01_signal_analysis_results.csv`
- `analysis/01_signal_analysis_qc.csv`
- `analysis/02_comparative_analysis_results.csv`
- `analysis/02_comparative_analysis_qc.csv`
- `analysis/03_feature_analysis_results.csv`
- `analysis/03_feature_analysis_qc.csv`
- `analysis/quarter_summary_YYYY.csv`
- `analysis/summary_YYYY.txt`
- 跨年份运行时的 `trend_START_END.csv`

## 安装依赖

上层 [pyproject.toml](D:/program_FAERS/pyproject.toml) 里当前声明了以下依赖：

- `pandas`
- `pyarrow`
- `scipy`
- `duckdb`
- `scikit-learn`
- `xgboost`

如果使用 `uv`：

```powershell
cd D:\program_FAERS
uv sync
```

如果使用 `pip`，至少建议安装：

```powershell
pip install pandas pyarrow scipy
```

## 运行方法

### 处理单张表

```powershell
cd D:\program_FAERS\faers_project
python main.py --year 2024 --quarter Q1 --table demo
python main.py --year 2024 --quarter Q1 --table drug
python main.py --year 2024 --quarter Q1 --table reac
python main.py --year 2024 --quarter Q1 --table outc
```

### 处理单个季度的完整流程

```powershell
python main.py --year 2024 --quarter Q1 --table all
```

这个命令会顺序执行：

- `demo`
- `drug`
- `drug_feature`
- `drug_exposure`
- `reac`
- `outc`
- `case`
- `signal`

### 只构建病例级主分析表

```powershell
python main.py --year 2024 --quarter Q1 --table case
```

如果依赖文件不存在，脚本会自动补跑前置步骤。

### 只构建信号分析表

```powershell
python main.py --year 2024 --quarter Q1 --table signal
```

如果依赖文件不存在，也会自动补跑前置步骤。

### 按年批处理

处理单年：

```powershell
python year_batch_runner.py --year 2024
```

处理年份区间：

```powershell
python year_batch_runner.py --start-year 2019 --end-year 2024
```

常用参数：

- `--verbose`：显示每一步详细输出
- `--force`：即使已有年度结果也强制重建
- `--output-root`：指定输出根目录

## 年度分析内容

[year_batch_runner.py](D:/program_FAERS/faers_project/year_batch_runner.py) 在季度级产物完成后，会继续调用 [analysis_project](D:/program_FAERS/analysis_project) 中的分析脚本：

- [01_signal_analysis.py](D:/program_FAERS/analysis_project/01_signal_analysis.py)
  - 比较 `zolpidem suspect` 与其他 suspect drug 的不成比例信号
- [02_comparative_analysis.py](D:/program_FAERS/analysis_project/02_comparative_analysis.py)
  - 比较 `zolpidem_only` 与 `other_zdrug_only`
- [03_feature_analysis.py](D:/program_FAERS/analysis_project/03_feature_analysis.py)
  - 在 `zolpidem` 暴露人群内部做分层特征分析

当前主要分析的结局包括：

- `strict_fall`：狭义跌倒定义
- `broad_fall`：广义跌倒相关定义

## 辅助脚本

- [structure_scan.py](D:/program_FAERS/faers_project/structure_scan.py)
  - 扫描原始 FAERS 文件结构、字段变化和分隔符情况
- [explore_zolpidem_pt.py](D:/program_FAERS/faers_project/explore_zolpidem_pt.py)
  - 统计 `zolpidem` 病例中的 PT 词频
- [check.py](D:/program_FAERS/faers_project/check.py)
  - 对某个固定输出文件做快速人工检查

## 关键默认假设

这套代码目前有几个很重要的研究设定，后续回看时最好优先确认：

- 研究对象默认只保留 `65-120` 岁病例
- 性别只保留 `M` 和 `F`
- `zolpidem` 主分析暴露口径为 `PS + SS`
- 敏感性分析暴露口径为 `PS only`
- 多药并用定义为 `distinct_drug_n >= 5`
- 同时命中 `zolpidem` 和其他 `Z-drug` 的病例，在部分比较分析里会被排除

## 常见问题

### 1. 找不到原始文件

先确认目录结构是否符合：

```text
data/年份/Q季度/.../*.txt
```

如果不在默认目录，请设置 `FAERS_RAW_ROOT`。

### 2. 处理时提示缺字段

不同年份的 FAERS 表头可能不完全一致。代码已经在 [utils.py](D:/program_FAERS/faers_project/utils.py) 中兼容了一部分历史字段名。如果遇到更老或更特殊的季度，建议先运行 [structure_scan.py](D:/program_FAERS/faers_project/structure_scan.py) 查看原始结构。

### 3. 年度批处理跑不起来

先确认以下目录存在：

- `D:\program_FAERS\data`
- `D:\program_FAERS\OUTPUT`
- `D:\program_FAERS\analysis_project`

`year_batch_runner.py` 依赖 `analysis_project` 中的年度分析脚本，不能只拷贝 `faers_project` 单独运行。

## 建议的回顾顺序

如果之后很久不看代码，建议按下面顺序快速找回上下文：

1. 先看 [main.py](D:/program_FAERS/faers_project/main.py)
2. 再看 [utils.py](D:/program_FAERS/faers_project/utils.py)
3. 再看 `demo_processor.py`
4. 再看 `drug_processor.py`
5. 再看 `drug_feature_processor.py`
6. 再看 `drug_exposure_processor.py`
7. 再看 `reac_processor.py` 和 `outc_processor.py`
8. 最后看 [case_dataset_processor.py](D:/program_FAERS/faers_project/case_dataset_processor.py) 和 [signal_dataset_processor.py](D:/program_FAERS/faers_project/signal_dataset_processor.py)

这样最快能重新理解这条数据处理链路。
