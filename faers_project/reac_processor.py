import pandas as pd
from pathlib import Path
from utils import read_faers_txt, build_file_path
from config import RAW_ROOT


def process_reac(year, quarter, output_root):
    """
    处理 FAERS REAC 数据（不良反应事件信息）
    
    参数:
        year: 年份（如 2024）
        quarter: 季度（如 'Q1', 'Q2' 等）
        output_root: 输出文件根目录
    
    处理步骤:
    1. 构建输入文件路径并检查文件是否存在
    2. 读取原始数据
    3. 查看数据基本信息（行数、列名）
    4. 保存为 Parquet 格式
    
    REAC 文件包含的不良反应信息:
    - caseid: 病例 ID
    - pt: 首选术语（Preferred Term），描述不良反应的标准化医学术语
    - 等其他不良反应相关字段
    """
    # ========== 步骤 1: 构建输入文件路径 ==========
    # 使用 build_file_path 函数根据年份、季度、表名自动生成标准化路径
    file_path = build_file_path(RAW_ROOT, year, quarter, "REAC")
    print(f"正在处理文件：{file_path}")

    # 检查文件是否存在，如果不存在则抛出异常
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    # ========== 步骤 2: 读取原始数据 ==========
    df = read_faers_txt(file_path)

    # 打印数据基本信息
    print("原始行数:", len(df))
    print("列名:")
    print(list(df.columns))

    # ========== 步骤 3: 保存处理后的数据 ==========
    # 确保输出目录存在
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建输出文件路径，保存为 Parquet 格式
    output_file = output_root / f"reac_{year}{quarter.lower()}.parquet"

    # 保存数据
    df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")