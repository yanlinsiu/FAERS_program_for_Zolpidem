from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_common import (
    OUTCOME_SPECS,
    describe_signal,
    ensure_output_dir,
    feature_mask,
    merge_signal_and_feature,
    ror_prr_from_counts,
    save_tables,
    two_by_two_counts,
)


# 定义暴露（唑吡坦使用）列名
EXPOSURE_COL = "is_zolpidem_suspect"  # 任何可疑角色的唑吡坦使用标识
EXPOSURE_COL_PS = "is_zolpidem_suspect_ps"  # 仅主要可疑角色（PS）的唑吡坦使用标识

# 定义特征规格列表：(特征列名，特征值，特征域)
FEATURE_SPECS = [
    # 人口学特征（demographic）
    ("age_group", "65-74", "demographic"),  # 65-74 岁年龄组
    ("age_group", "75-84", "demographic"),  # 75-84 岁年龄组
    ("age_group", ">=85", "demographic"),   # 85 岁及以上年龄组
    ("sex_clean", "F", "demographic"),      # 女性
    ("sex_clean", "M", "demographic"),      # 男性
    # 严重程度特征（severity）
    ("serious", True, "severity"),          # 严重病例标识
    # 用药负担特征（medication_burden）
    ("polypharmacy_5", True, "medication_burden"),  # 多药治疗（≥5 种药物）
    # 合并用药特征（co_medication）
    ("is_benzo", True, "co_medication"),       # 同时使用苯二氮䓬类药物
    ("is_antidepressant", True, "co_medication"),  # 同时使用抗抑郁药
    ("is_antipsychotic", True, "co_medication"),   # 同时使用抗精神病药
    ("is_opioid", True, "co_medication"),     # 同时使用阿片类药物
    ("is_antiepileptic", True, "co_medication"),  # 同时使用抗癫痫药
]


def _build_feature_rows(
    df: pd.DataFrame,
    analysis_name: str,
    outcome_name: str,
    outcome_col: str,
    outcome_label: str,
) -> pd.DataFrame:
    """
    构建特征分析结果行，计算各特征亚组的信号检测指标
    
    该函数针对唑吡坦暴露人群，按不同特征（如年龄、性别、合并用药等）进行分层，
    计算每个特征亚组中不良事件的报告比值比（ROR）和比例报告比值比（PRR），
    识别高风险亚组。
    
    参数:
        df: 输入 DataFrame，包含唑吡坦暴露人群的数据及特征列
        analysis_name: 分析名称，如"primary_ps_ss"或"sensitivity_ps_only"
        outcome_name: 结局指标名称，如"fall"（跌倒）、"fracture"（骨折）
        outcome_col: 结局指标列名，对应 DataFrame 中的布尔列
        outcome_label: 结局指标的详细描述标签
        
    返回:
        pd.DataFrame: 特征分析结果表，包含各特征亚组的 ROR、PRR 等指标
        
    异常:
        KeyError: 当必要特征列不存在于 DataFrame 中时
        
    示例:
        >>> result_df = _build_feature_rows(
        ...     zolpidem_df,
        ...     analysis_name="primary_ps_ss",
...     outcome_name="fall",
...     outcome_col="is_fall_narrow",
        ...     outcome_label="Fall-related SMQ"
        ... )
        >>> print(result_df[["feature_name", "ror", "prr"]])
    """
    rows = []  # 存储特征分析结果行列表
    
    # 遍历所有预定义的特征规格
    for column, value, domain in FEATURE_SPECS:
        # 跳过不存在的特征列
        if column not in df.columns:
            continue
            
        # 创建特征掩码：筛选出具有该特征的亚组
        # 例如：column="age_group", value="75-84" → 筛选 75-84 岁的病例
        mask = feature_mask(df, column, value)
        
        # 计算 2x2 列联表计数
        # exposed=mask：特征阳性 vs 特征阴性
        # outcome=df[outcome_col]：发生结局 vs 未发生结局
        counts = two_by_two_counts(mask, df[outcome_col])
        
        # 基于计数计算信号检测指标（ROR、PRR、95% CI、χ²）
        metrics = ror_prr_from_counts(**counts)
        
        # 计算特征阳性的病例数
        exposed_n = int(mask.sum())
        
        # 计算特征阳性且发生结局的病例数
        outcome_n = int((mask & df[outcome_col].fillna(False).astype(bool)).sum())
        
        # 构建特征分析结果行
        rows.append(
            {
                "analysis": analysis_name,  # 分析类型名称
                "outcome_name": outcome_name,  # 结局指标名称
                "outcome_definition": outcome_label,  # 结局指标定义描述
                "feature_domain": domain,  # 特征所属域（人口学/严重程度/合并用药等）
                "feature_name": f"{column}={value}",  # 特征名称及取值
                "n_feature_positive": exposed_n,  # 特征阳性的病例数
                "n_feature_positive_outcome": outcome_n,  # 特征阳性且发生结局的病例数
                "outcome_reporting_rate": (outcome_n / exposed_n) if exposed_n else None,  # 结局报告率
                "conclusion": describe_signal(metrics),  # 基于信号指标的结论描述
                **metrics,  # 展开信号指标字典（ROR、PRR、chi2、ci_lower、ci_upper）
            }
        )
    
    # 将结果列表转换为 DataFrame 并返回
    return pd.DataFrame(rows)


def build_feature_analysis(
    signal_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    signal_file: str | Path | None = None,
    feature_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    执行特征分析：识别唑吡坦使用者中的高风险亚组
    
    该函数在唑吡坦暴露人群中，按人口学特征、严重程度、用药负担、合并用药
    等维度进行分层分析，计算各亚组的不良事件信号强度，识别哪些特征与
    更高的不良事件报告风险相关。
    
    分析场景包括：
    1. 主要分析（primary_ps_ss）：基于任何可疑角色的唑吡坦使用
    2. 敏感性分析（sensitivity_ps_only）：基于主要可疑角色的唑吡坦使用
    
    参数:
        signal_root: 信号数据集根目录路径，默认为 None 使用内置默认路径
        output_dir: 输出目录路径，默认为 None 使用内置默认路径
        signal_file: 指定的信号数据文件路径，默认为 None 自动查找
        feature_file: 指定的特征数据文件路径，默认为 None 自动查找
        
    返回:
        tuple[pd.DataFrame, pd.DataFrame]:
            - 第一个 DataFrame: 特征分析结果，包含各特征亚组的 ROR、PRR 等指标
            - 第二个 DataFrame: 质量控制数据，包含样本量、缺失值统计等
            
    异常:
        FileNotFoundError: 当信号数据文件或特征数据文件不存在时
        KeyError: 当必要字段缺失时
        
    示例:
        >>> result_df, qc_df = build_feature_analysis(
        ...     signal_root="D:/data/OUTPUT",
        ...     output_dir="D:/data/analysis"
        ... )
        >>> # 查看哪些特征亚组的跌倒风险最高
        >>> high_risk = result_df[result_df["ror"] > 2].sort_values("ror", ascending=False)
        >>> print(high_risk[["feature_name", "ror", "conclusion"]])
    """
    # 合并信号数据集与特征数据集
    # 该函数加载信号数据（含结局指标）和特征数据（含人口学、合并用药等）
    # 并通过 primaryid 进行关联，返回合并后的 DataFrame
    merged_df = merge_signal_and_feature(
        signal_root=signal_root,
        signal_file=signal_file,
        feature_file=feature_file,
    )

    result_frames = []  # 存储各结局指标的特征分析结果 DataFrame 列表
    qc_rows = []  # 存储质量控制数据行列表
    
    # 遍历所有预定义的结局指标配置（如跌倒、骨折等）
    for outcome_spec in OUTCOME_SPECS:
        outcome_name = outcome_spec["outcome_name"]  # 结局指标名称
        outcome_col = outcome_spec["outcome_col"]  # 结局指标列名
        outcome_label = outcome_spec["outcome_label"]  # 结局指标标签描述
        
        # 跳过不存在的结局列
        if outcome_col not in merged_df.columns:
            continue

        # 遍历两种分析配置：分析名称、暴露列名
        for analysis_name, exposure_col in [
            ("primary_ps_ss", EXPOSURE_COL),      # 主要分析：任何可疑角色的唑吡坦
            ("sensitivity_ps_only", EXPOSURE_COL_PS),  # 敏感性分析：主要可疑角色的唑吡坦
        ]:
            # 筛选唑吡坦暴露人群（即使用唑吡坦的病例）
            subset = merged_df[merged_df[exposure_col].fillna(False).astype(bool)].copy()
            
            # 构建当前结局指标的特征分析结果表
            # 针对唑吡坦暴露人群，按年龄、性别、合并用药等特征分层分析
            result_frames.append(
                _build_feature_rows(
                    subset,
                    analysis_name=analysis_name,
                    outcome_name=outcome_name,
                    outcome_col=outcome_col,
                    outcome_label=outcome_label,
                )
            )
            
            # 构建质量控制数据行，记录样本量和数据质量信息
            qc_rows.append(
                {
                    "analysis": analysis_name,  # 分析类型
                    "outcome_name": outcome_name,  # 结局指标名称
                    "n_zolpidem_exposed": int(len(subset)),  # 唑吡坦暴露的总病例数
                    "n_outcome": int(subset[outcome_col].fillna(False).astype(bool).sum()),  # 发生结局的病例数
                    
                    # 关键变量缺失值统计（用于评估数据质量）
                    "missing_age_group": int(subset["age_group"].isna().sum()) if "age_group" in subset.columns else None,  # 年龄组缺失数
                    "missing_sex_clean": int(subset["sex_clean"].isna().sum()) if "sex_clean" in subset.columns else None,  # 性别缺失数
                    "missing_serious": int(subset["serious"].isna().sum()) if "serious" in subset.columns else None,  # 严重程度缺失数
                    
                    # 重要特征分布统计
                    "n_polypharmacy_5": int(subset["polypharmacy_5"].fillna(False).astype(bool).sum()) if "polypharmacy_5" in subset.columns else None,  # 多药治疗病例数
                    "n_serious": int(subset["serious"].fillna(False).astype(bool).sum()) if "serious" in subset.columns else None,  # 严重病例数
                    "n_benzo": int(subset["is_benzo"].fillna(False).astype(bool).sum()) if "is_benzo" in subset.columns else None,  # 合并使用苯二氮䓬类病例数
                    "n_antidepressant": int(subset["is_antidepressant"].fillna(False).astype(bool).sum()) if "is_antidepressant" in subset.columns else None,  # 合并使用抗抑郁药病例数
                    "n_antipsychotic": int(subset["is_antipsychotic"].fillna(False).astype(bool).sum()) if "is_antipsychotic" in subset.columns else None,  # 合并使用抗精神病药病例数
                    "n_opioid": int(subset["is_opioid"].fillna(False).astype(bool).sum()) if "is_opioid" in subset.columns else None,  # 合并使用阿片类病例数
                    "n_antiepileptic": int(subset["is_antiepileptic"].fillna(False).astype(bool).sum()) if "is_antiepileptic" in subset.columns else None,  # 合并使用抗癫痫药病例数
                }
            )

    # 合并所有结局指标的特征分析结果为单个 DataFrame
    result_df = pd.concat(result_frames, ignore_index=True)
    
    # 按分析类型、结局名称、ROR 值降序排序，便于识别高风险特征
    result_df = result_df.sort_values(
        ["analysis", "outcome_name", "ror"],
        ascending=[True, True, False],  # ROR 降序排列，最大值在前
        na_position="last",  # NaN 值排到最后
    )
    
    # 将 QC 行列表转换为 DataFrame
    qc_df = pd.DataFrame(qc_rows)
    
    # 确保输出目录存在，不存在则创建
    output_root = ensure_output_dir(output_dir)
    
    # 保存主结果表和 QC 表为 CSV 文件
    save_tables(
        result_df,
        qc_df,
        output_root / "03_feature_analysis_results.csv",  # 特征分析主结果文件
        output_root / "03_feature_analysis_qc.csv",  # 质控结果文件
    )
    
    # 返回特征分析结果和质控数据
    return result_df, qc_df


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Feature analysis: who is more likely to have fall reports among zolpidem reports"
    )
    parser.add_argument(
        "--signal-root",
        default=r"D:\program_FAERS\OUTPUT",
        help="Directory containing signal and drug feature parquet files"
    )
    parser.add_argument(
        "--output-dir",
        default=r"D:\program_FAERS\OUTPUT\analysis",
        help="Directory to save result and QC tables"
    )
    args = parser.parse_args()
    
    # 执行特征分析
    results, qc = build_feature_analysis(args.signal_root, args.output_dir)
    
    # 打印特征分析结果到控制台
    print(results.to_string(index=False))
    
    # 打印保存的 QC 行数
    print("saved QC rows:", len(qc))
