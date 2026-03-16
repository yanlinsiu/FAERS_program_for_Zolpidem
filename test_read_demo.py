import pandas as pd
from pathlib import Path

def read_faers_txt(file_path):
    """
    读取 FAERS 数据库的 ASCII 文本文件
    
    参数:
        file_path: 文件路径
        
    返回:
        DataFrame - 读取后的数据，列名已转换为小写
    """
    df = pd.read_csv(
        file_path,              # 文件路径
        sep="$",                # FAERS 数据使用美元符号作为分隔符
        encoding="latin1",      # latin1 编码，兼容特殊字符
        low_memory=False        # 关闭低内存模式，避免类型推断警告
    )
    # 将列名去除空格并转为小写，方便后续引用（如 CaseID → caseid）
    df.columns = df.columns.str.strip().str.lower()
    return df

# ========== 主程序开始 ==========

# 设置 FAERS DEMO 文件路径（患者人口统计信息）
file = r"2024\Q1\ASCII\DEMO24Q1.txt"

# 读取数据
df = read_faers_txt(file)

# 查看数据基本信息
print(df.head())           # 打印前 5 行，预览数据结构
print(df.columns)          # 打印所有列名
print("原始行数：", len(df))  # 打印总行数

# ========== 数据清洗步骤 1: 类型转换 ==========
# 将 caseversion（病例版本号）转换为数值类型
# errors="coerce" 会将无法转换的值自动设为 NaN（空值）
# 这是 pandas 的经典用法：先转类型，再排序，最后去重
df["caseversion"] = pd.to_numeric(df["caseversion"], errors="coerce")

# ========== 数据清洗步骤 2: 排序 ==========
# 按 caseid（病例 ID）和 caseversion（版本号）升序排序
# 目的：让同一个病例的不同版本按从小到大排列，为去重做准备
df = df.sort_values(["caseid", "caseversion"])

# ========== 数据清洗步骤 3: 去重 ==========
# 删除重复的 caseid，只保留每个病例的最新版本
# subset="caseid" 表示根据 caseid 判断是否重复
# keep="last" 表示保留每组重复中的最后一条记录（即 caseversion 最大的）
df_dedup = df.drop_duplicates(subset="caseid", keep="last")

# 检查去重效果
print("\n去重后行数：", len(df_dedup))
print("原始数据中重复 caseid 数量：", df["caseid"].duplicated().sum())
print("去重后重复 caseid 数量：", df_dedup["caseid"].duplicated().sum())

# ========== 保存处理后的数据 ==========
# 保存为 Parquet 格式（高效的列式存储格式，适合大规模数据分析）
output_file = r"D:\program_FAERS\2024\Q1\demo_2024q1.parquet"
df_dedup.to_parquet(output_file, index=False)  # index=False 表示不保存行索引

print("Parquet 文件已保存：", output_file)