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
    ror_prr_from_counts,
    save_tables,
    two_by_two_counts,
)


def _build_comparator_subset(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    构建对照组分析子集，筛选仅使用唑吡坦或仅使用其他 Z 药物的病例
    
    该函数用于 Comparative Analysis（对比分析），通过限制分析人群为
    "zolpidem_only"（仅唑吡坦）和"other_zdrug_only"（仅其他 Z 药物）两组，
    排除同时使用两类药物的混杂病例，实现更纯净的组间对比。
    
    参数:
        df: 输入的信号数据集 DataFrame，包含药物分组信息
        group_col: 药物分组列名，如"target_drug_group"或"target_drug_group_ps"
        
    返回:
        pd.DataFrame: 筛选后的子集，仅包含"zolpidem_only"和"other_zdrug_only"两组的病例
        
    异常:
        KeyError: 当 group_col 列不存在于 DataFrame 中时
        
    示例:
        >>> subset = _build_comparator_subset(signal_df, "target_drug_group")
        >>> print(subset["target_drug_group"].unique())
        ['zolpidem_only', 'other_zdrug_only']
    """
    allowed = {"zolpidem_only", "other_zdrug_only"}  # 允许的药物分组：仅唑吡坦和仅其他 Z 药物
    subset = df[df[group_col].isin(allowed)].copy()  # 筛选出这两组病例
    return subset


def build_comparative_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    执行对比分析：唑吡坦 vs 其他 Z 药物
    
    该函数比较唑吡坦单药治疗与其他 Z 药物单药治疗的不良事件报告信号，
    计算报告比值比（ROR）和比例报告比值比（PRR）等指标，评估唑吡坦
    相对于其他 Z 药物的安全性差异。
    
    分析场景包括：
    1. 主要分析（primary_ps_ss）：基于任何可疑角色的药物分组
    2. 敏感性分析（sensitivity_ps_only）：基于主要可疑角色的药物分组
    
    参数:
        signal_root: 信号数据集根目录路径，默认为 None 使用内置默认路径
        output_dir: 输出目录路径，默认为 None 使用内置默认路径
        signal_file: 指定的信号数据文件路径，默认为 None 自动查找
        
    返回:
        tuple[pd.DataFrame, pd.DataFrame]:
            - 第一个 DataFrame: 对比分析结果，包含 ROR、PRR 等信号指标
            - 第二个 DataFrame: 质量控制数据，包含各分组的病例数、结局报告率等
            
    异常:
        FileNotFoundError: 当信号数据文件不存在时
        KeyError: 当必要字段缺失时
        
    示例:
        >>> result_df, qc_df = build_comparative_analysis(
        ...     signal_root="D:/data/OUTPUT",
        ...     output_dir="D:/data/analysis"
        ... )
        >>> print(result_df[["analysis", "outcome_name", "ror"]])
    """
    # 加载信号数据集
    signal_df = load_signal_dataset(signal_root=signal_root, signal_file=signal_file)

    result_rows = []  # 存储对比分析结果行
    qc_rows = []  # 存储质量控制数据行
    stratified_frames = []  # 存储分层分析数据框列表
    
    # 定义两种分析配置：分析名称、分组列名、唑吡坦组标识值
    configs = [
        ("primary_ps_ss", "target_drug_group", "zolpidem_only"),  # 主要分析：任何可疑角色
        ("sensitivity_ps_only", "target_drug_group_ps", "zolpidem_only"),  # 敏感性分析：主要可疑角色
    ]

    # 遍历所有结局指标配置（如跌倒、骨折等）
    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]  # 结局指标名称
        outcome_col = outcome_spec["outcome_col"]  # 结局指标列名
        outcome_label = outcome_spec["outcome_label"]  # 结局指标标签描述
        if outcome_col not in signal_df.columns:
            continue  # 跳过不存在的结局列

        # 遍历两种分析配置
        for analysis_name, group_col, zolpidem_value in configs:
            # 构建对照组子集：仅包含唑吡坦单药和其他 Z 药物单药的病例
            subset = _build_comparator_subset(signal_df, group_col)
            # 创建暴露组标识：唑吡坦组为 True，其他 Z 药物组为 False
            exposed = subset[group_col].eq(zolpidem_value)
            # 计算 2x2 列联表计数：暴露（唑吡坦）vs 结局（不良事件）
            counts = two_by_two_counts(exposed, subset[outcome_col])
            # 基于计数计算信号检测指标（ROR、PRR 及其置信区间）
            metrics = ror_prr_from_counts(**counts)
            # 构建结果行，包含分析信息和信号结论
            result_rows.append(
                {
                    "analysis": analysis_name,  # 分析类型名称
                    "comparison": "zolpidem_only_vs_other_zdrug_only",  # 对比组描述
                    "outcome_name": outcome_name,  # 结局指标名称
                    "outcome_definition": outcome_label,  # 结局指标定义描述
                    "conclusion": describe_signal(metrics),  # 基于指标的结论描述
                    **metrics,  # 展开信号指标字典（ROR、PRR 等）
                }
            )

            # 按药物分组生成质量控制统计数据
            for drug_group, frame in subset.groupby(group_col, dropna=False):
                qc_rows.append(
                    {
                        "analysis": analysis_name,  # 分析类型
                        "outcome_name": outcome_name,  # 结局指标名称
                        "drug_group": drug_group,  # 药物分组名称
                        "n_cases": int(len(frame)),  # 病例总数
                        "n_outcome": int(frame[outcome_col].fillna(False).astype(bool).sum()),  # 结局发生数
                        "outcome_reporting_rate": float(frame[outcome_col].fillna(False).astype(bool).mean()) if len(frame) else None,  # 结局报告率
                        "n_female": int(frame["sex_clean"].eq("F").sum()) if "sex_clean" in frame.columns else None,  # 女性病例数
                        "n_age_75_plus": int(frame["age_group"].isin(["75-84", ">=85"]).sum()) if "age_group" in frame.columns else None,  # 75 岁以上病例数
                    }
                )

            # 构建分层分析结果（按年龄、性别等分层）
            stratified_df = build_stratified_rows(
                subset.assign(exposed_group=exposed),  # 添加暴露组列
                analysis_name=analysis_name,
                exposure_col="exposed_group",  # 使用新建的暴露组列
                outcome_col=outcome_col,
                outcome_name=outcome_name,
                outcome_label=outcome_label,
            )
            if not stratified_df.empty:
                stratified_frames.append(stratified_df)  # 收集非空的分层结果

    # 将结果列表转换为 DataFrame
    result_df = pd.DataFrame(result_rows)
    # 将 QC 行列表转换为 DataFrame
    qc_df = pd.DataFrame(qc_rows)
    # 确保输出目录存在，不存在则创建
    output_root = ensure_output_dir(output_dir)
    # 保存主结果和 QC 结果为 CSV 文件
    save_tables(
        result_df,
        qc_df,
        output_root / "02_comparative_analysis_results.csv",  # 主结果文件
        output_root / "02_comparative_analysis_qc.csv",  # QC 结果文件
    )
    # 合并并保存分层分析结果
    (
        pd.concat(stratified_frames, ignore_index=True, sort=False)
        if stratified_frames
        else pd.DataFrame()
    ).to_csv(
        output_root / "02_comparative_analysis_stratified.csv",  # 分层结果文件
        index=False,
        encoding="utf-8-sig",  # 使用 UTF-8 with BOM 编码，兼容 Excel
    )
    return result_df, qc_df


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Comparative analysis: zolpidem vs other Z-drugs")
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
    
    # 执行对比分析
    results, qc = build_comparative_analysis(args.signal_root, args.output_dir)
    # 打印结果到控制台
    print(results.to_string(index=False))
    # 打印保存的 QC 行数
    print("saved QC rows:", len(qc))
