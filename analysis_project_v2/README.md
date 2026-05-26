# analysis_project_v2

这一层负责 2004-2025 年全周期 FAERS 分析。主分析和趋势分析分开写，但共用同一套数据读取、暴露定义、结局定义和信号计算接口。

## 推荐运行顺序

```powershell
cd D:\program_FAERS\analysis_project_v2
uv run python run_all.py --period-token 2004_2025
```

如只需要重跑趋势分析：

```powershell
uv run python run_all.py --period-token 2004_2025 --skip-main
```

## 输出

主分析仍输出到 `D:\program_FAERS\OUTPUT_GLOBAL\analysis_v2`。

趋势分析输出到 `D:\program_FAERS\OUTPUT_GLOBAL_COUNTRY\regulatory_trend`，其中：

- `annual_trend.csv`：逐年唑吡坦 suspect 报告数、跌倒报告数、跌倒报告比例和年度 ROR。
- `rolling_trend.csv`：3 年滚动窗口 ROR，例如 2004-2006、2005-2007。
- `event_period_comparison.csv`：2013、2019 和 2023 前后分层结果。
- `paper_regulatory_trend_summary.csv`：论文正文优先使用的精简表，只保留全周期、2013 前后、2019 前后的主分析狭义跌倒结果。
- `annual_trend_qc.csv`：趋势分析质控结果。

## 口径说明

趋势分析不是独立研究，而是主分析的时间延伸。论文正文建议重点报告 `primary_ps_ss` + `strict_fall`，也就是 PS+SS 口径下唑吡坦 suspect 暴露与狭义跌倒结局。PS only、广义跌倒和 2023 AGS Beers Criteria 相关结果保留在 CSV 中作为补充。
