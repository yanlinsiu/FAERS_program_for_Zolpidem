import pandas as pd
from pathlib import Path

def read_faers_txt(file_path):
    df = pd.read_csv(
        file_path,
        sep="$",
        encoding="latin1",
        low_memory=False
    )
    # 统一列名：转小写，去空格
    df.columns = df.columns.str.strip().str.lower()
    return df