import pandas as pd
from pathlib import Path


def read_faers_txt(file_path):
    """
    读取 FAERS ASCII txt 文件
    """
    df = pd.read_csv(
        file_path,
        sep="$",
        encoding="latin1",
        low_memory=False
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_file_path(raw_root, year, quarter, table_name):
    """
    自动拼接原始文件路径
    例如:
    raw_root=D:\\program_FAERS
    year=2024
    quarter=Q1
    table_name=DRUG

    ->
    D:\\program_FAERS\\2024\\Q1\\ASCII\\DRUG24Q1.txt
    """
    year = str(year)
    quarter = quarter.upper()
    table_name = table_name.upper()

    year_short = year[-2:]
    filename = f"{table_name}{year_short}{quarter}.txt"

    return Path(raw_root) / year / quarter / "ASCII" / filename


def iter_quarters(start_year, start_quarter, end_year=None, end_quarter=None):
    """
    Generate (year, quarter) tuples for a single quarter or an inclusive range.
    """
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    quarter_index = {quarter: index for index, quarter in enumerate(quarter_order)}

    start_year = int(start_year)
    start_quarter = start_quarter.upper()
    end_year = start_year if end_year is None else int(end_year)
    end_quarter = start_quarter if end_quarter is None else end_quarter.upper()

    if start_quarter not in quarter_index or end_quarter not in quarter_index:
        raise ValueError("quarter must be one of Q1, Q2, Q3, Q4")

    start_key = (start_year, quarter_index[start_quarter])
    end_key = (end_year, quarter_index[end_quarter])
    if start_key > end_key:
        raise ValueError("start quarter must be earlier than or equal to end quarter")

    year = start_year
    quarter_pos = quarter_index[start_quarter]
    while (year, quarter_pos) <= end_key:
        yield year, quarter_order[quarter_pos]
        quarter_pos += 1
        if quarter_pos == len(quarter_order):
            year += 1
            quarter_pos = 0
