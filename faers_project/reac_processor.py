import pandas as pd
from pathlib import Path

# 从 utils 模块导入工具函数：路径构建、加载保留的 primaryid、读取 FAERS 文件
from utils import build_file_path, load_retained_demo_primaryids, read_faers_txt

# 从配置文件导入根目录配置
from config import RAW_ROOT


def process_reac(year, quarter, output_root):
    """
    处理 FAERS REAC 数据，并生成病例级别的跌倒标识。

    REAC 表中的每一行代表一个不良反应事件；本函数会先保留与
    DEMO 最新病例版本一致的事件，再将事件级数据聚合为病例级结果。
    """
    # ========== 步骤 1: 构建输入文件路径 ==========
    # 使用 build_file_path 函数根据年份、季度、表名自动生成标准化路径
    # 路径格式：{RAW_ROOT}/{year}/{quarter}/ASCII/REAC{年尾 2 位}{quarter}.txt
    file_path = build_file_path(RAW_ROOT, year, quarter, "REAC")
    print(f"正在处理文件：{file_path}")

    # 检查文件是否存在，如果不存在则抛出异常
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    # ========== 步骤 2: 读取原始数据 ==========
    # 使用统一的读取函数读取 REAC 文本文件
    # 自动处理分隔符 ($)、编码 (latin1) 和列名格式化
    df = read_faers_txt(file_path)

    # ========== 步骤 3: 数据验证和清洗 ==========
    required_cols = ["primaryid", "caseid"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"REAC 缺少必要字段：{missing_cols}")

    if "pt" in df.columns:
        reaction_term_col = "pt"
    elif "reac_pt" in df.columns:
        reaction_term_col = "reac_pt"
    else:
        raise ValueError("REAC 缺少不良反应术语字段：需要 pt 或 reac_pt")

    # 将 primaryid 转换为数值类型，无法转换的值设为 NaN
    # primaryid 用于与 DEMO 表关联，必须是有效的数字 ID
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")

    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    
    df["pt"] = (
        df[reaction_term_col]
        .where(df[reaction_term_col].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ========== 步骤 4: 与 DEMO 表关联过滤 ==========
    retained_primaryids = load_retained_demo_primaryids(RAW_ROOT, year, quarter)
    df = df[df["primaryid"].isin(retained_primaryids)]
    df = df[df["caseid"] != ""]
    print("DEMO 保留 primaryid 过滤后 REAC 事件行数:", len(df))

    # ========== 步骤 5: 创建跌倒标识并聚合到病例级 ==========
    # PT (Preferred Term) 是 MedDRA 标准术语；这里仅将 FALL/FALLS
    # 视为跌倒事件，避免将其他相关但含义不同的术语误判为跌倒。
    fall_terms = ["FALL", "FALLS"]
    df["is_fall"] = df["pt"].isin(fall_terms)

    # REAC 是事件级表，一个病例可对应多条反应记录。
    # 只要任一条记录命中跌倒术语，该病例就标记为发生过跌倒。
    case_level_df = (
        df[["caseid", "is_fall"]].groupby("caseid", as_index=False)["is_fall"].max()
    )

    print("病例级 REAC 行数:", len(case_level_df))
    print("跌倒病例数:", int(case_level_df["is_fall"].sum()))

    # ========== 步骤 6: 保存病例级结果 ==========
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 输出仅保留病例编号和跌倒标识，供后续病例级分析使用。
    output_file = output_root / f"reac_{year}{quarter.lower()}_case.parquet"

    case_level_df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")
