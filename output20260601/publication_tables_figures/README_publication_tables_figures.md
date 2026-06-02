# Publication tables and figures

已根据 `output20260601` 中现有结果补齐普通中文药学期刊常用的 3 个表和 4 张图。

## Tables
- Table1_signal_main_sensitivity.csv / XLSX sheet `Table1`: 主分析和敏感性分析信号结果。
- Table2_adjusted_ROR.csv / XLSX sheet `Table2`: 暴露项的调整后 ROR 结果。
- Table3_regulatory_trend.csv / XLSX sheet `Table3`: 监管节点前后趋势。

## Figures
- Figure1_study_flow.png/pdf: 研究流程图。
- Figure2_annual_ROR_trend.png/pdf: 年度 ROR 趋势图。
- Figure3_top15_PT_bar.png/pdf: 前 15 个共现 PT 条形图。
- Figure3_alternative_PT_network.png/pdf: 已有 PT 共现网络图，可作为图 3 的替代版本。
- Figure4_ML_ROC.png/pdf: 机器学习模型 ROC 曲线。
- Figure4_alternative_XGBoost_feature_importance.png/pdf: XGBoost 特征重要性图，可作为图 4 的替代版本。

## Suggested captions
表 1  唑吡坦相关跌倒报告的主分析和敏感性分析信号结果

表 2  唑吡坦相关跌倒报告的多因素调整后 ROR 结果

表 3  主要监管节点前后唑吡坦相关跌倒报告趋势

图 1  研究对象筛选和分析流程。

图 2  2004-2025 年唑吡坦相关跌倒报告的年度 ROR 变化趋势。

图 3  唑吡坦相关跌倒报告中排名前 15 位的共现 PT。

图 4  机器学习模型在测试集中的 ROC 曲线。

## Source files
- Signal tables: `signal_analysis_ror_prr_ic_ebgm/analysis_v2_20260528_archived/`
- Trend tables: `regulatory_trend_ror/regulatory_trend_current/`
- PT network: `pt_cooccurrence_network/pt_cooccurrence_network_20260526_20260527/`
- ML runs: enhanced logistic regression, random forest, and XGBoost runs under `machine_learning/`