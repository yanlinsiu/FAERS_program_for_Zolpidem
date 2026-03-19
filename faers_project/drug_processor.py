import pandas as pd
from pathlib import Path

# 从配置文件导入原始数据根目录配置
from config import RAW_ROOT
# 从 utils 模块导入工具函数：路径构建、加载保留的 primaryid、读取 FAERS 文件
from utils import (
    attach_caseid_from_demo,
    build_file_path,
    ensure_required_columns,
    load_retained_demo_primaryids,
    read_faers_txt,
)


def process_drug(year, quarter, output_root):
    """
    处理 FAERS DRUG 数据（药物信息）
    
    DRUG 表说明:
        DRUG (Drugs) 表包含病例中报告的药物详细信息，是 FAERS 数据库的核心表之一。
        每条记录代表一个病例中报告的一种药物，包括药物名称、活性成分、作用角色等。
        一个病例可能报告多种药物，因此 DRUG 表的行数通常远多于 DEMO 表。
    
    参数:
        year (int): 年份，例如 2024
        quarter (str): 季度，例如 'Q1', 'Q2', 'Q3', 'Q4'
        output_root (str or Path): 输出文件的根目录路径
        
    处理步骤:
        1. 构建输入文件路径并检查文件是否存在
        2. 读取原始 DRUG 数据
        3. 数据验证和清洗（检查必要字段、类型转换、文本标准化）
        4. 与 DEMO 表关联过滤（只保留有效病例的记录）
        5. 数据质量过滤（去除关键字段全空的记录）
        6. 保存为 Parquet 格式
        
    DRUG 文件主要字段说明:
        - primaryid: 主要标识符，唯一标识一条药物记录，用于与 DEMO 表关联
        - caseid: 病例 ID，标识属于哪个病例
        - drugname: 药物名称（商品名或通用名）
        - prod_ai: 产品活性成分（Product Active Ingredient）
        - role_cod: 角色代码（Role Code），标识药物在不良反应中的角色
            * PS (Primary Suspect): 主要怀疑药物
            * SS (Secondary Suspect): 次要怀疑药物
            * C (Concomitant): 并用药物
            * I (Interacting): 相互作用药物
        - drugdose: 药物剂量
        - drugther: 药物治疗作用
        - route: 给药途径
        - dose_freq: 给药频率
        - 等其他药物相关字段
        
    数据清洗规则:
        - 将 drugname、prod_ai、role_cod 转换为大写，统一格式
        - 去除首尾空格
        - 处理空值，转换为空字符串
        - 过滤掉 caseid 为空的记录
        - 过滤掉 drugname 和 prod_ai 同时为空的记录（确保至少有药物标识信息）
        
    数据处理逻辑:
        - 保留与 DEMO 表中最新病例版本对应的记录
        - 通过 primaryid 进行关联过滤
        - 转换为 Parquet 格式以优化存储和读取性能
        
    异常:
        FileNotFoundError: 当 DRUG 文件不存在时抛出
        ValueError: 当缺少必要字段时抛出
        
    示例:
        process_drug(2024, 'Q1', 'D:\\processed_data')
    """
    # ========== 步骤 1: 构建输入文件路径 ==========
    # 使用 build_file_path 函数根据年份、季度、表名自动生成标准化路径
    # 路径格式：{RAW_ROOT}/{year}/{quarter}/ASCII/DRUG{年尾 2 位}{quarter}.txt
    file = build_file_path(RAW_ROOT, year, quarter, "DRUG")
    print(f"正在处理文件：{file}")

    # 检查文件是否存在，如果不存在则抛出异常
    if not file.exists():
        raise FileNotFoundError(f"找不到文件：{file}")

    # ========== 步骤 2: 读取原始数据 ==========
    # 使用统一的读取函数读取 DRUG 文本文件
    # read_faers_txt 会自动处理:
    # - 分隔符 ($)：FAERS 标准格式
    # - 编码 (latin1)：兼容特殊字符
    # - 列名转小写：统一命名规范
    df = read_faers_txt(file, dataset_name="DRUG")

    # ========== 步骤 3: 数据验证和清洗 ==========
    # 定义 DRUG 表必需的字段
    required_cols = ["primaryid", "caseid", "drugname", "prod_ai", "role_cod"]

    # 打印数据行数，了解数据规模
    print("DRUG 行数:", len(df))

    # 打印所有列名，查看可用的药物信息字段
    print("DRUG 列名:")
    print(list(df.columns))

    df = attach_caseid_from_demo(df, RAW_ROOT, year, quarter, output_root=output_root)
    ensure_required_columns(df, ["primaryid", "caseid", "drugname", "prod_ai", "role_cod"], "DRUG")

    # 将 primaryid 转换为数值类型，无法转换的值设为 NaN
    # primaryid 用于与 DEMO 表关联，必须是有效的数字 ID
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    
    # 清洗 caseid 字段：
    # 1. 将 NaN 替换为空字符串
    # 2. 转换为字符串类型
    # 3. 去除首尾空格
    df["caseid"] = (
        df["caseid"]
        .where(df["caseid"]  # 1. NaN 变空字符串
        .notna(), "")        
        .astype(str)         # 2. 强制转为字符串
        .str.strip()         # 3. 去除首尾空格
    )
    
    df["drugname"] = (
        df["drugname"]
        .where(df["drugname"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    
    # 清洗 prod_ai（产品活性成分）字段：
    # 处理方式与 drugname 相同，统一转为大写
    df["prod_ai"] = (
        df["prod_ai"]
        .where(df["prod_ai"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    
    # 清洗 role_cod（角色代码）字段：
    # 处理方式与 drugname 相同，统一转为大写
    df["role_cod"] = (
        df["role_cod"]
        .where(df["role_cod"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ========== 步骤 4: 与 DEMO 表关联过滤 ==========
    # 加载 DEMO 表中保留的 primaryid 集合（去重后的最新版本病例）
    # 这样可以确保只处理有效病例的药物记录
    retained_primaryids = load_retained_demo_primaryids(
        RAW_ROOT, year, quarter, output_root=output_root
    )
    
    # 过滤 DRUG 数据，只保留在 DEMO 表中的 primaryid 对应的记录
    # 这一步确保了数据的一致性和完整性
    df = df[df["primaryid"].isin(retained_primaryids)]
    print("DEMO 保留 primaryid 过滤后行数:", len(df))

    # ========== 步骤 5: 数据质量过滤 ==========
    # 过滤掉 caseid 为空的记录
    # caseid 是病例标识，为空则无法关联到具体病例
    df = df[df["caseid"] != ""]
    
    # 过滤掉 drugname 和 prod_ai 同时为空的记录
    # 这两个字段至少有一个要有值，否则该药物记录没有实际意义
    # ~ 表示逻辑取反，即不满足条件的记录被过滤掉
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))]
    print("去掉空 caseid 及药物字段全空后行数:", len(df))

    # ========== 步骤 6: 保存处理后的数据 ==========
    # 构建输出文件路径，保存为 Parquet 格式
    # Parquet 是高效的列式存储格式，适合大规模数据分析
    output_root = Path(output_root)
    # 创建输出目录（如果不存在），parents=True 表示递归创建多级目录
    # exist_ok=True 表示如果目录已存在则不抛出异常
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建输出文件路径
    # 文件名格式：drug_{年份}{季度小写}.parquet，例如：drug_2024q1.parquet
    output_file = output_root / f"drug_{year}{quarter.lower()}.parquet"
    
    # 保存数据为 Parquet 格式
    # index=False 表示不保存索引列
    df.to_parquet(output_file, index=False)

    # 打印完成提示
    print("DRUG parquet 保存完成:", output_file)
