import pandas as pd
from pathlib import Path

from config import RAW_ROOT
from utils import build_file_path, load_retained_demo_primaryids, read_faers_txt


# OUTC 结果代码到布尔标志列的映射表
# 用于将不良反应结局代码转换为病例级别的布尔标志
OUTC_CODE_TO_FLAG = {
    "DE": "is_death",  # 死亡
    "LT": "is_life_threatening",  # 危及生命
    "HO": "is_hospitalization",  # 住院
    "DS": "is_disability",  # 残疾
    "CA": "is_congenital_anomaly",  # 先天性异常
    "OT": "is_other_serious",  # 其他严重情况
}


def process_outc(year, quarter, output_root):
    """
    处理 FAERS OUTC 数据并生成病例级严重结局表。

    基于 FAERS OUTC（Outcome）数据，识别每个病例的严重结局类型，
    并生成病例粒度的结局特征表，用于后续的严重性分析和风险评估。

    参数:
        year (int): FAERS 数据年份，如 2024
        quarter (str): 季度标识，取值为 'Q1', 'Q2', 'Q3', 'Q4' 之一
        output_root (str | Path): 输出文件的根目录路径

    返回:
        pd.DataFrame: 病例级严重结局表，包含以下核心字段:
            - caseid: 病例 ID
            - is_death: 是否死亡结局
            - is_life_threatening: 是否危及生命
            - is_hospitalization: 是否导致住院
            - is_disability: 是否导致残疾
            - is_congenital_anomaly: 是否先天性异常
            - is_other_serious: 是否其他严重情况
            - is_serious_any: 是否存在任何严重结局（以上任一为 True）

    异常:
        FileNotFoundError: 当 OUTC 原始数据文件不存在时抛出
        ValueError: 当 OUTC 数据缺少必要字段时抛出

    处理流程:
        1. 构建 OUTC 原始数据文件路径
        2. 读取 OUTC 文本数据（使用$分隔符，latin1 编码）
        3. 清洗 primaryid、caseid、outc_cod 字段
        4. 通过与 DEMO 表关联过滤保留有效的 primaryid
        5. 将 outc_cod 代码映射为布尔标志列
        6. 按 caseid 聚合，每个病例保留一条记录
        7. 添加 is_serious_any 汇总标志
        8. 保存为 Parquet 格式文件

    示例:
        >>> df = process_outc(2024, 'Q1', './output')
        >>> print(df.columns)
        Index(['caseid', 'is_death', 'is_life_threatening',
               'is_hospitalization', 'is_disability',
               'is_congenital_anomaly', 'is_other_serious',
               'is_serious_any'], dtype='object')
    """
    # 构建 OUTC 原始数据文件路径
    file_path = build_file_path(RAW_ROOT, year, quarter, "OUTC")
    print(f"processing file: {file_path}")

    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"file not found: {file_path}")

    # 读取 OUTC 数据（使用$分隔符，latin1 编码）
    df = read_faers_txt(file_path)

    # 验证必要字段是否存在
    required_cols = ["primaryid", "caseid", "outc_cod"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"OUTC missing required columns: {missing_cols}")

    # 数据类型转换和清洗
    # primaryid 转为数值型，无效值转为 NaN
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    
    # caseid 去除空格，空值转为空字符串
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    
    # outc_cod 转为大写，去除空格，空值转为空字符串
    df["outc_cod"] = (
        df["outc_cod"]
        .where(df["outc_cod"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # 加载 DEMO 表保留的 primaryid，用于关联过滤
    retained_primaryids = load_retained_demo_primaryids(
        RAW_ROOT, year, quarter, output_root=output_root
    )
    
    # 仅保留与 DEMO 表关联的 primaryid 记录
    df = df[df["primaryid"].isin(retained_primaryids)]
    
    # 过滤掉 caseid 或 outc_cod 为空的记录
    df = df[df["caseid"] != ""]
    df = df[df["outc_cod"] != ""]
    print("OUTC rows after DEMO/caseid/outc_cod filtering:", len(df))

    # 检查未知的 outc_cod 代码并输出统计信息
    known_codes = set(OUTC_CODE_TO_FLAG.keys())
    unknown_codes = df.loc[~df["outc_cod"].isin(known_codes), "outc_cod"]
    if not unknown_codes.empty:
        unknown_counts = unknown_codes.value_counts()
        print(
            "unknown outc_cod found (top 20):",
            unknown_counts.head(20).to_dict(),
        )
        print("unknown outc_cod total rows:", int(unknown_counts.sum()))

    # 将每个 outc_cod 映射为对应的布尔标志列
    for outc_code, flag_col in OUTC_CODE_TO_FLAG.items():
        df[flag_col] = df["outc_cod"].eq(outc_code)

    # 获取所有布尔标志列名
    flag_cols = list(OUTC_CODE_TO_FLAG.values())

    # 如果数据为空，创建空 DataFrame；否则按 caseid 聚合
    if df.empty:
        case_level_df = pd.DataFrame(columns=["caseid", *flag_cols])
    else:
        # 按 caseid 分组，每个病例取最大值（任一记录为 True 即为 True）
        case_level_df = (
            df[["caseid", *flag_cols]].groupby("caseid", as_index=False)[flag_cols].max()
        )

    # 添加 is_serious_any 标志：任一严重结局为 True 即为 True
    case_level_df["is_serious_any"] = case_level_df[flag_cols].any(axis=1)

    # 填充空值并确保所有标志列为布尔类型
    for col in [*flag_cols, "is_serious_any"]:
        case_level_df[col] = case_level_df[col].fillna(False).astype(bool)

    # 创建输出目录并保存结果
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"outc_{year}{quarter.lower()}_case.parquet"
    case_level_df.to_parquet(output_file, index=False)

    # 打印统计信息
    print("OUTC case-level rows:", len(case_level_df))
    print("serious cases (is_serious_any=True):", int(case_level_df["is_serious_any"].sum()))
    print(f"saved: {output_file}")

    return case_level_df
