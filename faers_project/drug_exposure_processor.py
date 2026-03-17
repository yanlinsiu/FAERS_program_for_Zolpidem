import re
import pandas as pd
from pathlib import Path

from drug_processor import process_drug


ZOLPIDEM_TERMS = [
    "ZOLPIDEM",
    "AMBIEN",
    "STILNOX",
    "EDLUAR",
    "INTERMEZZO",
    "ZOLPIMIST",
]

OTHER_ZDRUG_TERMS = [
    "ZALEPLON",
    "SONATA",
    "ZOPICLONE",
    "IMOVANE",
    "ZIMOVANE",
    "ESZOPICLONE",
    "LUNESTA",
]

SUSPECT_ROLES = {"PS", "SS"}


def _normalize_drug_text(series):
    """
    统一标准化药物文本，便于后续做边界匹配和去重计数。

    处理规则:
        1. 空值转为空字符串
        2. 去除首尾空格
        3. 转为大写
        4. 将连续空白压缩为单个空格

    参数:
        series (pd.Series): 需要标准化的文本序列

    返回:
        pd.Series: 标准化后的文本序列

    示例:
        >>> series = pd.Series(["  Zolpidem  ", "AMBIEN", None])
        >>> _normalize_drug_text(series)
        0    ZOLPIDEM
        1      AMBIEN
        2            
        dtype: object
    """
    return (
        series.where(series.notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def _build_boundary_pattern(terms):
    """
    根据匹配词列表构建正则表达式模式，用于精确匹配药物名称。

    使用单词边界确保不会错误匹配部分字符串。例如，避免将"ZOLPIDEM"匹配到"ZOLPIDEMXR"。
    这里使用 re.escape 对词项转义，避免商品名中若存在特殊字符时影响匹配。

    参数:
        terms (List[str]): 药物术语列表，用于构建匹配模式

    返回:
        str: 正则表达式模式字符串，包含所有术语的边界匹配

    模式说明:
        - (?<![A-Z0-9]): 负向后顾断言，确保前面不是大写字母或数字
        - (?:term1|term2|...): 非捕获组，匹配任意一个术语
        - (?![A-Z0-9]): 负面前瞻断言，确保后面不是大写字母或数字

    示例:
        >>> terms = ["ZOLPIDEM", "AMBIEN"]
        >>> _build_boundary_pattern(terms)
        '(?<![A-Z0-9])(?:ZOLPIDEM|AMBIEN)(?![A-Z0-9])'
    """
    escaped_terms = sorted({re.escape(term) for term in terms}, key=len, reverse=True)
    alternation = "|".join(escaped_terms)
    return rf"(?<![A-Z0-9])(?:{alternation})(?![A-Z0-9])"


def process_drug_exposure(year, quarter, output_root):
    """
    生成病例级研究暴露定义表（PS/SS suspect 口径）。

    基于 FAERS DRUG 数据，识别每个病例中是否涉及唑吡坦或其他 Z-drug 作为可疑药物，
    并生成病例粒度的暴露特征表，用于后续的统计分析和信号挖掘。

    参数:
        year (int): FAERS 数据年份，如 2024
        quarter (str): 季度标识，取值为 'Q1', 'Q2', 'Q3', 'Q4' 之一
        output_root (str | Path): 输出文件的根目录路径

    返回:
        pd.DataFrame: 病例级暴露定义表，包含以下核心字段:
            - caseid: 病例 ID
            - is_zolpidem_suspect: 是否为唑吡坦可疑病例
            - is_other_zdrug_suspect: 是否为其他 Z-drug 可疑病例
            - suspect_role_any: 是否存在任何可疑角色药物
            - target_drug_group: 目标药物分组，取值包括:
                * no_suspect_drug: 无可疑药物
                * no_target_zdrug_suspect: 有可疑药物但非目标 Z-drug
                * other_zdrug_only: 仅其他 Z-drug 可疑
                * zolpidem_only: 仅唑吡坦可疑
                * both_zolpidem_and_other_zdrug: 两者皆可疑

    异常:
        FileNotFoundError: 当 DRUG 数据文件不存在且无法自动生成时抛出
        ValueError: 当 DRUG 数据缺少必要字段时抛出

    处理流程:
        1. 读取或自动生成 DRUG 数据的 Parquet 文件
        2. 验证并标准化关键字段（caseid, drugname, prod_ai, role_cod）
        3. 过滤无效记录（空 caseid 或空药物名称）
        4. 识别可疑角色药物（role_cod 为 PS 或 SS）
        5. 使用正则边界匹配识别唑吡坦和其他 Z-drug
        6. 按 caseid 聚合，生成病例级别的暴露标志
        7. 根据暴露情况划分目标药物组别
        8. 保存结果到 Parquet 文件

    示例:
        >>> df = process_drug_exposure(2024, 'Q1', './output')
        >>> print(df.columns)
        Index(['caseid', 'suspect_role_any', 'is_zolpidem_suspect',
               'is_other_zdrug_suspect', 'target_drug_group'],
              dtype='object')
    """
    # 初始化输出路径对象
    output_root = Path(output_root)
    
    # 构建 DRUG 数据文件路径
    drug_file = output_root / f"drug_{year}{quarter.lower()}.parquet"

    # 如果 DRUG 文件不存在，尝试自动生成
    if not drug_file.exists():
        print(f"未找到 DRUG 结果文件，正在自动生成：{drug_file}")
        process_drug(year, quarter, output_root)

    # 再次检查文件是否存在，不存在则抛出异常
    if not drug_file.exists():
        raise FileNotFoundError(f"找不到 DRUG 文件：{drug_file}")

    # 读取 DRUG 数据
    df = pd.read_parquet(drug_file)

    # 验证必要字段是否存在
    required_cols = ["caseid", "drugname", "prod_ai", "role_cod"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DRUG 结果缺少必要字段：{missing_cols}")

    # 标准化关键字段：去除空格、转大写、处理空值
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    df["drugname"] = _normalize_drug_text(df["drugname"])
    df["prod_ai"] = _normalize_drug_text(df["prod_ai"])
    df["role_cod"] = _normalize_drug_text(df["role_cod"])

    # 过滤掉 caseid 为空的记录
    df = df[df["caseid"] != ""]
    # 过滤掉 drugname 和 prod_ai 同时为空的记录（至少有一个药物信息）
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))]

    # 标记是否为可疑角色药物（PS=主要怀疑，SS=次要怀疑）
    df["is_suspect_role"] = df["role_cod"].isin(SUSPECT_ROLES)

    # 构建唑吡坦和其他 Z-drug 的正则匹配模式
    zolpidem_pattern = _build_boundary_pattern(ZOLPIDEM_TERMS)
    other_zdrug_pattern = _build_boundary_pattern(OTHER_ZDRUG_TERMS)

    # 在 drugname 和 prod_ai 字段中匹配唑吡坦和其他 Z-drug
    df["is_zolpidem_hit"] = (
        df["drugname"].str.contains(zolpidem_pattern, regex=True, na=False)
        | df["prod_ai"].str.contains(zolpidem_pattern, regex=True, na=False)
    )
    df["is_other_zdrug_hit"] = (
        df["drugname"].str.contains(other_zdrug_pattern, regex=True, na=False)
        | df["prod_ai"].str.contains(other_zdrug_pattern, regex=True, na=False)
    )

    # 标记可疑角色中的唑吡坦和其他 Z-drug 记录
    df["is_zolpidem_suspect_row"] = df["is_suspect_role"] & df["is_zolpidem_hit"]
    df["is_other_zdrug_suspect_row"] = df["is_suspect_role"] & df["is_other_zdrug_hit"]

    # 按 caseid 聚合，生成病例级别的暴露标志
    grouped = df.groupby("caseid", as_index=False).agg(
        suspect_role_any=("is_suspect_role", "max"),
        is_zolpidem_suspect=("is_zolpidem_suspect_row", "max"),
        is_other_zdrug_suspect=("is_other_zdrug_suspect_row", "max"),
    )

    # 填充空值并确保布尔类型正确
    grouped["is_zolpidem_suspect"] = grouped["is_zolpidem_suspect"].fillna(False).astype(
        bool
    )
    grouped["is_other_zdrug_suspect"] = grouped["is_other_zdrug_suspect"].fillna(
        False
    ).astype(bool)
    grouped["suspect_role_any"] = grouped["suspect_role_any"].fillna(False).astype(bool)

    # 初始化目标药物分组为"无可疑药物"
    grouped["target_drug_group"] = "no_suspect_drug"
    
    # 如果有可疑角色药物，更新为"无目标 Z-drug 可疑"
    grouped.loc[
        grouped["suspect_role_any"],
        "target_drug_group",
    ] = "no_target_zdrug_suspect"
    
    # 如果有其他 Z-drug 可疑，更新分组
    grouped.loc[
        grouped["is_other_zdrug_suspect"],
        "target_drug_group",
    ] = "other_zdrug_only"
    
    # 如果有唑吡坦可疑，更新分组
    grouped.loc[
        grouped["is_zolpidem_suspect"],
        "target_drug_group",
    ] = "zolpidem_only"
    
    # 如果两者皆有，覆盖为最高优先级分组
    grouped.loc[
        grouped["is_zolpidem_suspect"] & grouped["is_other_zdrug_suspect"],
        "target_drug_group",
    ] = "both_zolpidem_and_other_zdrug"

    # 构建输出文件路径并保存
    output_file = output_root / f"drug_exposure_{year}{quarter.lower()}_case.parquet"
    grouped.to_parquet(output_file, index=False)

    # 打印统计信息
    print("病例级研究暴露定义表行数:", len(grouped))
    print("唑吡坦 suspect 病例数:", int(grouped["is_zolpidem_suspect"].sum()))
    print("其他 Z-drug suspect 病例数:", int(grouped["is_other_zdrug_suspect"].sum()))
    print(f"已保存：{output_file}")

    return grouped
