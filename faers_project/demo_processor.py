import pandas as pd
from pathlib import Path
from config import RAW_ROOT
from utils import read_faers_txt, build_file_path


def process_demo(year, quarter, output_root):
    """
    处理 FAERS DEMO 数据（患者人口统计信息）
    
    参数:
        year: 年份（如 2024）
        quarter: 季度（如 'Q1', 'Q2' 等）
        output_root: 输出文件根目录
    
    处理步骤:
    1. 构建输入文件路径并检查文件是否存在
    2. 读取原始数据
    3. 转换 caseversion 为数值类型
    4. 按病例 ID 和版本号排序，保留最新版本
    5. 去重，确保每个病例只保留一条记录
    6. 保存为 Parquet 格式
    """
    # ========== 步骤 1: 构建输入文件路径 ==========
    # 使用 build_file_path 函数根据年份、季度、表名自动生成标准化路径
    # 例如：D:\program_FAERS\2024\Q1\ASCII\DEMO24Q1.txt
    file_path = build_file_path(RAW_ROOT, year, quarter, "DEMO")
    print(f"正在处理文件：{file_path}")

    # 检查文件是否存在，如果不存在则抛出异常
    # 这是一个安全检查，避免后续操作因文件不存在而报错
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    # ========== 步骤 2: 读取原始数据 ==========
    # read_faers_txt 会自动处理 FAERS 数据格式：
    # - 分隔符：$
    # - 编码：latin1
    # - 列名转小写
    df = read_faers_txt(file_path)

    # 打印数据基本信息，帮助了解数据结构
    print("原始行数:", len(df))
    print("列名:")
    print(list(df.columns))

    # ========== 步骤 3: 数据清洗 - 类型转换 ==========
    # 将 caseversion（病例版本号）转换为数值类型
    # errors="coerce" 会将无法转换的值自动设为 NaN（空值）
    # 这是 pandas 的经典用法：先转类型，再排序，最后去重
    df["caseversion"] = pd.to_numeric(df["caseversion"], errors="coerce")

    # ========== 步骤 4: 数据清洗 - 排序 ==========
    # 先按 caseid（病例 ID）排序，再按 caseversion（版本号）升序排列
    # 目的：让同一个病例的不同版本按从小到大有序排列，为去重做准备
    df = df.sort_values(["caseid", "caseversion"])

    # ========== 步骤 5: 数据清洗 - 去重 ==========
    # 删除重复的 caseid，只保留每个病例的最新版本
    # subset="caseid" 表示根据病例 ID 判断重复
    # keep="last" 表示保留每组中的最后一条记录（即 caseversion 最大的）
    df = df.drop_duplicates(subset="caseid", keep="last")

    # 打印去重效果
    print("去重后行数:", len(df))
    # 验证是否还有重复的 caseid（应该为 0）
    print("去重后重复 caseid 数量:", df["caseid"].duplicated().sum())

    # ========== 步骤 6: 保存处理后的数据 ==========
    # 确保输出目录存在，如果不存在则创建
    # parents=True 表示递归创建所有父目录
    # exist_ok=True 表示如果目录已存在则不报错
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建输出文件路径，保存为 Parquet 格式
    # Parquet 是高效的列式存储格式，读取速度快，文件体积小
    output_file = output_root / f"demo_{year}{quarter.lower()}.parquet"
    
    # 保存数据，index=False 表示不保存行索引
    df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")