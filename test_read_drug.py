import pandas as pd

# DRUG 文件路径
file = r"D:\program_FAERS\2024\Q1\ASCII\DRUG24Q1.txt"

# 读取 FAERS DRUG 表
# sep="$" 表示使用 $ 作为分隔符
# encoding="latin1" 避免编码问题
df = pd.read_csv(
    file,
    sep="$",
    encoding="latin1",
    low_memory=False
)

# 查看前5行数据
print(df.head())

# 查看列名
print(df.columns)

# 查看总行数
print("总行数：", len(df))

# 查看 drugname 列前10个不同值
print("\n示例药物名称：")
print(df["drugname"].dropna().unique()[:10])