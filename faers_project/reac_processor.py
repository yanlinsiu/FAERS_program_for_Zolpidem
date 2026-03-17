import pandas as pd
from pathlib import Path
# 从 utils 模块导入工具函数：路径构建、加载保留的 primaryid、读取 FAERS 文件
from utils import build_file_path, load_retained_demo_primaryids, read_faers_txt
# 从配置文件导入根目录配置
from config import RAW_ROOT


def process_reac(year, quarter, output_root):
    """
    处理 FAERS REAC 数据（不良反应事件信息）
    
    REAC 表说明:
        REAC (Reactions) 表包含药物不良反应事件的详细信息，是 FAERS 数据库的核心表之一。
        每条记录代表一个病例中报告的一个不良反应事件。
    
    参数:
        year (int): 年份，例如 2024
        quarter (str): 季度，例如 'Q1', 'Q2', 'Q3', 'Q4'
        output_root (str or Path): 输出文件的根目录路径
        
    处理步骤:
        1. 构建输入文件路径并检查文件是否存在
        2. 读取原始 REAC 数据
        3. 数据清洗和验证（检查必要字段、类型转换）
        4. 与 DEMO 表关联过滤（只保留有效病例的记录）
        5. 保存为 Parquet 格式
        
    REAC 文件主要字段说明:
        - primaryid: 主要标识符，唯一标识一条不良反应记录，用于与 DEMO 表关联
        - caseid: 病例 ID，标识属于哪个病例
        - pt: 首选术语（Preferred Term），描述不良反应的标准化医学术语（MedDRA 编码）
        - reac_pt: 不良反应的首选术语（某些版本可能使用此字段名）
        - 等其他不良反应相关字段
        
    数据处理逻辑:
        - 保留与 DEMO 表中最新病例版本对应的记录
        - 通过 primaryid 进行关联过滤
        - 转换为 Parquet 格式以优化存储和读取性能
        
    异常:
        FileNotFoundError: 当 REAC 文件不存在时抛出
        ValueError: 当缺少必要字段时抛出
        
    示例:
        process_reac(2024, 'Q1', 'D:\\processed_data')
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
    # 定义 REAC 表必需的字段
    required_cols = ["primaryid", "caseid"]
    # 检查是否有缺失的必需列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"REAC 缺少必要字段：{missing_cols}")

    # 将 primaryid 转换为数值类型，无法转换的值设为 NaN
    # primaryid 用于与 DEMO 表关联，必须是有效的数字 ID
    df["primaryid"] = pd.to_numeric(df["primaryid"], errors="coerce")
    
    # 清洗 caseid 字段：
    # 1. 将 NaN 替换为空字符串
    # 2. 转换为字符串类型
    # 3. 去除首尾空格
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()

    # ========== 步骤 4: 与 DEMO 表关联过滤 ==========
    # 加载 DEMO 表中保留的 primaryid 集合（去重后的最新版本病例）
    # 这样可以确保只处理有效病例的不良反应记录
    retained_primaryids = load_retained_demo_primaryids(RAW_ROOT, year, quarter)
    
    # 过滤 REAC 数据，只保留在 DEMO 表中的 primaryid 对应的记录
    # 这一步确保了数据的一致性和完整性
    df = df[df["primaryid"].isin(retained_primaryids)]

    # 打印数据基本信息，用于调试和监控
    print("DEMO 保留 primaryid 过滤后行数:", len(df))
    print("列名:")
    print(list(df.columns))

    # ========== 步骤 5: 保存处理后的数据 ==========
    # 将输出根目录转换为 Path 对象，便于路径操作
    output_root = Path(output_root)
    # 创建输出目录（如果不存在），parents=True 表示递归创建多级目录
    # exist_ok=True 表示如果目录已存在则不抛出异常
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建输出文件路径
    # 文件名格式：reac_{年份}{季度小写}.parquet，例如：reac_2024q1.parquet
    output_file = output_root / f"reac_{year}{quarter.lower()}.parquet"

    # 保存数据为 Parquet 格式
    # index=False 表示不保存索引列
    # Parquet 是一种高效的列式存储格式，适合大数据量的读写
    df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")
