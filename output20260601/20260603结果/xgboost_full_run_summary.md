# XGBoost 全量调参结果汇总（2026-06-03）

本结果来自学院集群全量训练，不是抽样训练。

## 运行设置

- 模型：XGBoost
- 特征版本：v2
- 目标变量：is_fall
- 研究人群：all
- 数据时期：2004-2025
- 训练集：2004-2023，3,511,708 行
- 验证集：2024，313,722 行
- 测试集：2025，329,593 行
- train_sample_n：0，即全量训练，不抽样
- 调参模式：full
- 交叉验证：3 折
- 搜索方式：RandomizedSearchCV
- 搜索参数组合数：60
- 调参优化目标：average_precision（AP）
- Bootstrap iterations：100

## 主要结果

| 特征集 | Validation ROC-AUC | Validation AP | Validation Brier | Test ROC-AUC | Test AP | Test Brier | Test Recall | Test Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core | 0.7921 | 0.1297 | 0.0249 | 0.7827 | 0.1376 | 0.0290 | 0.7437 | 0.0699 |
| enhanced | 0.8223 | 0.1753 | 0.0243 | 0.8150 | 0.1883 | 0.0282 | 0.7589 | 0.0791 |

## 结论

enhanced 特征集明显优于 core 特征集。测试集 AP 从 0.1376 提升到 0.1883，ROC-AUC 从 0.7827 提升到 0.8150，说明加入 phenotype 相关增强特征后，模型对跌倒病例的排序能力更好。

在测试集 top 5% 高风险病例中，enhanced 模型的阳性率为 0.1968，对应 lift 为 6.2588；top 10% 高风险病例中，阳性率为 0.1453，对应 lift 为 4.6188。考虑到测试集总体阳性率仅 0.0315，这说明模型能把更高风险病例排到前面。

## enhanced 最佳参数

```text
subsample = 0.7
scale_pos_weight = 5.0
reg_lambda = 1.0
reg_alpha = 1.0
n_estimators = 600
min_child_weight = 3
max_depth = 6
learning_rate = 0.08
gamma = 0.0
colsample_bytree = 0.9
```

## 输出文件位置

完整远端结果已复制到本地：

```text
D:\program_FAERS\output20260601\20260603结果\xgboost_full_run_remote_outputs\
```

其中：

```text
is_fall_all_2004_2025_v2_core_20260603_025119
is_fall_all_2004_2025_v2_enhanced_20260603_123825
```

每个目录内包含 `metrics.json`、`best_params.json`、`feature_importance.csv`、`validation_predictions.csv`、`test_predictions.csv`、ROC 曲线、校准曲线和 bootstrap 指标表等。
