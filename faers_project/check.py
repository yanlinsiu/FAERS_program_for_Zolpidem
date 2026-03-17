import pandas as pd

file = r"D:\program_FAERS\OUTPUT\case_dataset_2024q1.parquet"

df = pd.read_parquet(file)

print("数据行数:", len(df))
print("列名:")
print(df.columns)

print("\n前5行数据:")
print(df.head())

print("\n跌倒人数:", df["is_fall"].sum())
print("唑吡坦人数:", df["is_zolpidem"].sum())
print("多药人数:", df["polypharmacy"].sum())