# 以唑吡坦为暴露，和“其他可疑药物”做不成比例分析
# 指标：ROR、PRR

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from analysis_common import (
    OUTCOME_SPECS,
    build_stratified_rows,
    describe_signal,
    ensure_output_dir,
    load_signal_dataset,
    make_overall_qc,
    ror_prr_from_counts,
    save_tables,
    two_by_two_counts,
)


# 定义主要暴露变量和敏感性分析暴露变量
PRIMARY_EXPOSURE = "is_zolpidem_suspect"  # 主要分析：唑吡坦作为可疑药物
SENSITIVITY_EXPOSURE = "is_zolpidem_suspect_ps"  # 敏感性分析：唑吡坦作为主要可疑药物


def build_signal_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建信号分析数据集，计算唑吡坦与跌倒不良事件的关联信号
    
    该函数加载信号数据集，针对不同的结局指标和暴露定义，
    计算报告比值比（ROR）和比例报告比值比（PRR）等信号检测指标，
    并生成质量控制报告和分层分析结果。
    
    参数:
        signal_root: 信号数据集根目录路径，默认为 None 使用内置默认路径
        output_dir: 输出目录路径，默认为 None 使用内置默认路径
        signal_file: 指定的信号数据文件路径，默认为 None 自动查找
        
    返回:
        tuple[pd.DataFrame, pd.DataFrame]: 
            - 第一个 DataFrame: 信号分析结果，包含 ROR、PRR 等指标
            - 第二个 DataFrame: 质量控制数据，用于验证分析过程
            
    异常:
        FileNotFoundError: 当信号数据文件不存在时
        KeyError: 当必要字段缺失时
        
    示例:
        >>> result_df, qc_df = build_signal_analysis(
        ...     signal_root="D:/data/OUTPUT",
        ...     output_dir="D:/data/analysis"
        ... )
    """
    # 加载信号数据集
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)

    result_rows = []  # 存储信号分析结果行
    qc_frames = []  # 存储质量控制数据框列表
    stratified_frames = []  # 存储分层分析数据框列表

    # 遍历所有结局指标配置（如跌倒、骨折等）
    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]  # 结局指标名称
        outcome_col = outcome_spec["outcome_col"]  # 结局指标列名
        outcome_label = outcome_spec["outcome_label"]  # 结局指标标签描述
        if outcome_col not in signal_df.columns:
            continue  # 跳过不存在的结局列

        # 遍历两种分析类型：主要分析和敏感性分析
        for analysis_name, exposure_col, suspect_col, group_col in [
            ("primary_ps_ss", PRIMARY_EXPOSURE, "suspect_role_any", "target_drug_group"),  # 主要分析：任何可疑角色
            ("sensitivity_ps_only", SENSITIVITY_EXPOSURE, "suspect_role_any_ps", "target_drug_group_ps"),  # 敏感性分析：仅主要可疑
        ]:
            # 筛选出目标药物作为可疑药物的病例
            subset = signal_df[signal_df[suspect_col].fillna(False).astype(bool)].copy()
            # 排除唑吡坦和其他 Z 药物同时存在的情况，避免混杂
            subset = subset[subset[group_col] != "both_zolpidem_and_other_zdrug"].copy()

            # 计算 2x2 列联表计数：暴露（唑吡坦）vs 结局（跌倒）
            counts = two_by_two_counts(subset[exposure_col], subset[outcome_col])
            # 基于计数计算信号检测指标（ROR、PRR 及其置信区间）
            metrics = ror_prr_from_counts(**counts)
            # 构建结果行，包含分析信息和信号结论
            result_rows.append(
                {
                    "analysis": analysis_name,  # 分析类型名称
                    "exposure_definition": exposure_col,  # 暴露定义列名
                    "outcome_name": outcome_name,  # 结局指标名称
                    "outcome_definition": outcome_label,  # 结局指标定义描述
                    "comparison_group": "all_other_suspect_drugs_excluding_mixed_zdrug_cases",  # 对照组定义
                    "conclusion": describe_signal(metrics),  # 基于指标的结论描述
                    **metrics,  # 展开信号指标字典（ROR、PRR 等）
                }
            )

            # 生成整体质量控制报告
            qc_df = make_overall_qc(subset, analysis_name, exposure_col, outcome_col)
            qc_df["outcome_name"] = outcome_name  # 添加结局名称
            qc_df["outcome_definition"] = outcome_label  # 添加结局定义
            # 按药物分组统计病例数和结局发生数
            group_counts = (
                subset.groupby(group_col, dropna=False)
                .agg(
                    n_cases=("caseid", "count"),  # 病例数
                    n_outcome=(outcome_col, "sum"),  # 结局发生数
                )
                .reset_index()
                .rename(columns={group_col: "drug_group"})  # 重命名为药物组
            )
            group_counts["analysis"] = analysis_name  # 添加分析类型
            group_counts["outcome_name"] = outcome_name  # 添加结局名称
            group_counts["outcome_definition"] = outcome_label  # 添加结局定义
            qc_frames.append(qc_df)  # 添加整体 QC 数据
            qc_frames.append(group_counts)  # 添加分组计数数据

            # 构建分层分析结果（按年龄、性别等分层）
            stratified_df = build_stratified_rows(
                subset,
                analysis_name=analysis_name,
                exposure_col=exposure_col,
                outcome_col=outcome_col,
                outcome_name=outcome_name,
                outcome_label=outcome_label,
            )
            if not stratified_df.empty:
                stratified_frames.append(stratified_df)  # 收集非空的分层结果

    # 将结果列表转换为 DataFrame
    result_df = pd.DataFrame(result_rows)
    # 合并所有质量控制数据
    qc_result_df = pd.concat(qc_frames, ignore_index=True, sort=False)
    # 合并所有分层分析结果
    stratified_result_df = (
        pd.concat(stratified_frames, ignore_index=True, sort=False)
        if stratified_frames
        else pd.DataFrame()
    )

    # 确保输出目录存在，不存在则创建
    output_root = ensure_output_dir(output_dir)
    # 保存主结果和 QC 结果为 CSV 文件
    save_tables(
        result_df,
        qc_result_df,
        output_root / "01_signal_analysis_results.csv",  # 主结果文件
        output_root / "01_signal_analysis_qc.csv",  # QC 结果文件
    )
    # 保存分层分析结果为独立 CSV 文件
    stratified_result_df.to_csv(output_root / "01_signal_analysis_stratified.csv", index=False, encoding="utf-8-sig")
    return result_df, qc_result_df


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Zolpidem vs fall signal analysis on FAERS case-level data")
    parser.add_argument(
        "--signal-root",
        default=r"D:\program_FAERS\OUTPUT",
        help="Directory containing signal_dataset_*.parquet"
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\program_FAERS\OUTPUT\analysis",
        help="Directory to save result and QC tables"
    )
    args = parser.parse_args()
    
    # 执行信号分析
    results, qc = build_signal_analysis(args.signal_root, args.output_dir)
    # 打印结果到控制台
    print(results.to_string(index=False))
    print("saved QC rows:", len(qc))
