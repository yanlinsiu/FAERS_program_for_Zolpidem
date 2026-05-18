# FAERS machine learning module

这个目录是独立的离线机器学习层，用来给现有的 FAERS 统计分析再加一层“风险排序、模型对比、解释增强”。  
它不是主流程，不替代 `ROR / PRR / 分层分析`，也不输出因果结论。

## 设计思路

- 一个模型一个文件，方便展示和横向比较。
- 公共逻辑全部收敛到 `ml_common.py`，避免重复代码。
- 调参只在训练期数据里做，严格使用时间外验证。
- 主模型是 Logistic Regression，Random Forest 和 XGBoost 是 benchmark。
- 当前不引入深度学习，因为现在的输入是低维结构化特征，可解释模型更合适。

## 文件结构

- `01_logistic_regression.py`
- `02_random_forest.py`
- `03_xgboost.py`
- `ml_common.py`

## 输入数据

模型默认读取：

- `OUTPUT_GLOBAL/datasets/signal_dataset_<period>.parquet`
- `OUTPUT_GLOBAL/datasets/drug_feature_<period>_case.parquet`

按 `caseid` 做病例级 join，并只保留建模需要的字段。

## 特征

v1 使用固定的结构化特征：

- `age_group`
- `sex_clean`
- `quarter`
- `year`
- `drug_n`
- `distinct_drug_n`
- `is_zolpidem`
- `is_zaleplon`
- `is_zopiclone`
- `is_eszopiclone`
- `is_benzo`
- `is_antidepressant`
- `is_antipsychotic`
- `is_opioid`
- `is_antiepileptic`
- `polypharmacy_5`

明确不纳入：

- `fall_pt_list`
- `fall_narrow_pt_count`
- 任何直接 outcome 派生字段

## 时间切分

- 训练集：`year <= 2023`
- 验证集：`year = 2024`
- 测试集：`year = 2025`

调参、交叉验证都只能发生在训练期数据内。  
验证集只用于概率校准和阈值选择。  
测试集只用于最终报告。

## 调参策略

### Logistic Regression

- 默认 `search-mode=fast`
- 小范围 `GridSearchCV`
- 目标是稳、清楚、可解释

### Random Forest

- 默认 `search-mode=full`
- 受控 `GridSearchCV`
- 作为非线性对照模型

### XGBoost

- 默认 `search-mode=full`
- 小规模 `RandomizedSearchCV`
- 避免组合爆炸

所有搜索主指标都是 `average_precision`，同时记录 `roc_auc` 和 `brier_score`。

## 运行方式

在项目根目录执行：

```powershell
python ml_project\01_logistic_regression.py
python ml_project\02_random_forest.py
python ml_project\03_xgboost.py
```

常用参数：

```powershell
python ml_project\01_logistic_regression.py --search-mode fast --train-sample-n 0
python ml_project\02_random_forest.py --search-mode full --train-sample-n 400000
python ml_project\03_xgboost.py --search-mode full --train-sample-n 400000
```

快速烟雾测试：

```powershell
python ml_project\01_logistic_regression.py --search-mode none --train-sample-n 2000 --cv-folds 2 --bootstrap-iterations 10
```

## 输出

每次运行都会写入：

- `OUTPUT_ML/<model>/<target>_<period>_<timestamp>/`

统一产物包括：

- `metrics.json`
- `cv_metrics.csv`
- `best_params.json`
- `split_summary.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `validation_roc_curve.csv`
- `test_roc_curve.csv`
- `validation_calibration_curve.csv`
- `test_calibration_curve.csv`
- `test_bootstrap_metrics.csv`
- `model_card.md`

模型专属产物：

- Logistic Regression：`coefficients.csv`、`odds_ratios.csv`
- Random Forest / XGBoost：`feature_importance.csv`

## 解释口径

- 这些模型做的是预测排序，不是因果推断。
- 系数、重要性、概率分数都只能解释“和报告模式相关”，不能解释“药物导致了结果”。
- 真正的主叙事仍然应该回到你现有的 FAERS 统计分析框架里。
