from pathlib import Path

from config import RAW_ROOT
from utils import build_file_path, deduplicate_demo_records, read_faers_txt


def _load_dedup_demo_caseids(year, quarter):
    demo_file = build_file_path(RAW_ROOT, year, quarter, "DEMO")
    if not demo_file.exists():
        raise FileNotFoundError(f"找不到文件：{demo_file}")

    demo_df = read_faers_txt(demo_file)
    demo_df = deduplicate_demo_records(demo_df)
    return set(demo_df["caseid"])


def process_drug(year, quarter, output_root):
    """
    处理 FAERS DRUG 数据（药物信息）

    主要步骤:
    1. 读取原始药物数据
    2. 查看数据基本信息（行数、列名）
    3. 保存为 Parquet 格式

    DRUG 文件包含的药物信息:
    - caseid: 病例 ID
    - drugname: 药物名称
    - drugdose: 药物剂量
    - drugther: 药物治疗作用
    - 等其他药物相关字段
    """
    # 构建输入文件路径（使用 f-string 格式化路径）
    # DRUG 文件包含病例中报告的所有药物信息
    file = build_file_path(RAW_ROOT, year, quarter, "DRUG")
    print(f"正在处理文件：{file}")

    if not file.exists():
        raise FileNotFoundError(f"找不到文件：{file}")

    # 读取 FAERS 药物数据文件
    # read_faers_txt 会自动处理:
    # - 分隔符 ($)
    # - 编码 (latin1)
    # - 列名转小写
    df = read_faers_txt(file)

    required_cols = ["caseid", "drugname", "prod_ai", "role_cod"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DRUG 缺少必要字段: {missing_cols}")

    # 打印数据行数，了解数据规模
    print("DRUG 行数:", len(df))

    # 打印所有列名，查看可用的药物信息字段
    print("DRUG 列名:")
    print(list(df.columns))

    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    df["drugname"] = (
        df["drugname"]
        .where(df["drugname"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["prod_ai"] = (
        df["prod_ai"]
        .where(df["prod_ai"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    df["role_cod"] = (
        df["role_cod"]
        .where(df["role_cod"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    demo_caseids = _load_dedup_demo_caseids(year, quarter)
    df = df[df["caseid"].isin(demo_caseids)]
    print("DEMO 去重 caseid 过滤后行数:", len(df))

    df = df[df["caseid"] != ""]
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))]
    print("去掉空 caseid 及药物字段全空后行数:", len(df))

    # ========== 保存处理后的数据 ==========
    # 构建输出文件路径，保存为 Parquet 格式
    # Parquet 是高效的列式存储格式，适合大规模数据分析
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 保存数据，index=False 表示不保存行索引
    output_file = output_root / f"drug_{year}{quarter.lower()}.parquet"
    df.to_parquet(output_file, index=False)

    # 打印完成提示
    print("DRUG parquet 保存完成:", output_file)
