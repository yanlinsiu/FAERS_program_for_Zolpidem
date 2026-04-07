from pathlib import Path

import pandas as pd


file = Path(r"D:\program_FAERS\OUTPUT\case_dataset_2024q1.parquet")
df = pd.read_parquet(file)

print("数据行数:", len(df))
print("列名:")
print(df.columns.tolist())

print("\n前5行数据:")
print(df.head())

narrow_fall_n = int(df["is_fall_narrow"].fillna(False).astype(bool).sum())
broad_fall_n = (
    int(df["is_fall_broad"].fillna(False).astype(bool).sum())
    if "is_fall_broad" in df.columns
    else 0
)
polypharmacy_n = int(df["polypharmacy"].fillna(False).astype(bool).sum())

if "is_zolpidem_suspect" in df.columns:
    zolpidem_n = int(df["is_zolpidem_suspect"].fillna(False).astype(bool).sum())
    zolpidem_scope = "suspect口径(is_zolpidem_suspect)"
else:
    zolpidem_n = int(df["is_zolpidem"].fillna(False).astype(bool).sum())
    zolpidem_scope = "泛暴露口径(is_zolpidem)"

print("\n狭义跌倒人数:", narrow_fall_n)
print("广义跌倒相关人数:", broad_fall_n)
print(f"唑吡坦人数[{zolpidem_scope}]:", zolpidem_n)
if "is_zolpidem" in df.columns:
    print(
        "唑吡坦人数[泛暴露口径(is_zolpidem)]:",
        int(df["is_zolpidem"].fillna(False).astype(bool).sum()),
    )
if "is_zolpidem_suspect" in df.columns:
    print(
        "唑吡坦人数[suspect口径(is_zolpidem_suspect)]:",
        int(df["is_zolpidem_suspect"].fillna(False).astype(bool).sum()),
    )
print("多药并用人数:", polypharmacy_n)
