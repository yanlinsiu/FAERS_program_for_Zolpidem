from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


# 项目根目录路径（当前文件向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认 ML 运行根目录
DEFAULT_ML_RUN_ROOT = PROJECT_ROOT / "runs" / "mainline_2026-05-20"
# 输出根目录，可通过环境变量 FAERS_CLEAN_OUTPUT_ROOT 覆盖
OUTPUT_ROOT = Path(
    os.environ.get("FAERS_CLEAN_OUTPUT_ROOT", DEFAULT_ML_RUN_ROOT / "OUTPUT")
)
# ML 输出根目录，可通过环境变量 FAERS_ML_OUTPUT_ROOT 覆盖
OUTPUT_ML_ROOT = Path(
    os.environ.get("FAERS_ML_OUTPUT_ROOT", DEFAULT_ML_RUN_ROOT / "OUTPUT_ML")
)
# 特征工程 v2 版本的根目录
FEATURE_V2_ROOT = OUTPUT_ML_ROOT / "features_v2"
# 审计信息存放目录
AUDIT_DIR = FEATURE_V2_ROOT / "audit"
# MedDRA 查找表存放目录
LOOKUP_DIR = FEATURE_V2_ROOT / "lookup"
# 季度数据存放目录
QUARTERLY_DIR = FEATURE_V2_ROOT / "quarterly"
# 最终数据集存放目录
DATASET_DIR = FEATURE_V2_ROOT / "datasets"
# 质量控制结果存放目录
QC_DIR = FEATURE_V2_ROOT / "qc"
# 全局数据集目录，可通过环境变量 FAERS_GLOBAL_DATASET_DIR 覆盖
GLOBAL_DATASET_DIR = Path(
    os.environ.get(
        "FAERS_GLOBAL_DATASET_DIR",
        DEFAULT_ML_RUN_ROOT / "OUTPUT_GLOBAL" / "datasets",
    )
)

# FAERS 病例相关表的名称列表
TABLES = ("DEMO", "DRUG", "INDI", "RPSR", "THER")
# 季度标识列表
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


def ensure_feature_v2_dirs() -> None:
    """确保特征工程 v2 版本所需的所有子目录存在"""
    for directory in [AUDIT_DIR, LOOKUP_DIR, QUARTERLY_DIR, DATASET_DIR, QC_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def iter_quarters(start_year: int, end_year: int) -> Iterable[tuple[int, str]]:
    """
    迭代指定年份范围内的所有可用季度
    
    Args:
        start_year: 起始年份
        end_year: 结束年份
        
    Yields:
        (year, quarter) 元组，仅当该季度的 DEMO 表已处理完成时才产出
    """
    for year in range(int(start_year), int(end_year) + 1):
        for quarter in QUARTERS:
            # 只处理已经生成 DEMO 文件的季度
            if not processed_case_file(year, quarter, "DEMO").exists():
                continue
            yield year, quarter


def period_token(start_year: int, end_year: int) -> str:
    """
    生成时间段标识符
    
    Args:
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        格式为 "startYear_endYear" 的字符串，例如 "2019_2024"
    """
    return f"{int(start_year)}_{int(end_year)}"


def quarter_token(year: int, quarter: str) -> str:
    """
    生成季度标识符
    
    Args:
        year: 年份
        quarter: 季度标识（如 "Q1"）
        
    Returns:
        格式为 "yearqX" 的字符串，例如 "2024q1"
    """
    return f"{int(year)}{str(quarter).lower()}"


def clean_caseid(series: pd.Series) -> pd.Series:
    """
    清洗病例 ID 列
    
    Args:
        series: 包含病例 ID 的 Pandas Series
        
    Returns:
        清洗后的 Series：空值替换为空字符串，转换为字符串类型并去除首尾空格
    """
    return series.where(series.notna(), "").astype(str).str.strip()


def normalize_text(series: pd.Series) -> pd.Series:
    """
    标准化文本内容
    
    Args:
        series: 包含文本的 Pandas Series
        
    Returns:
        标准化后的 Series：空值替换为空字符串，转大写，合并多个连续空格为单个空格
    """
    return (
        series.where(series.notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_meddra_term(series: pd.Series) -> pd.Series:
    """
    标准化 MedDRA 术语
    
    Args:
        series: 包含 MedDRA 术语的 Pandas Series
        
    Returns:
        标准化后的 Series：在 normalize_text 基础上，仅保留大写字母和数字，其他字符替换为空格
    """
    return (
        normalize_text(series)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def output_quarter_dir(year: int) -> Path:
    """
    获取指定年份的季度输出目录路径
    
    Args:
        year: 年份
        
    Returns:
        路径对象，格式为 OUTPUT_ROOT/year/quarterly
    """
    return OUTPUT_ROOT / str(int(year)) / "quarterly"


def processed_quarter_file(year: int, quarter: str, stem: str) -> Path:
    """
    构建处理后季度文件的路径
    
    Args:
        year: 年份
        quarter: 季度标识
        stem: 文件名前缀
        
    Returns:
        完整的文件路径，格式为 {stem}_{year}q{X}.parquet
    """
    return output_quarter_dir(year) / f"{stem}_{quarter_token(year, quarter)}.parquet"


def processed_case_file(year: int, quarter: str, table_name: str) -> Path:
    """
    构建病例相关表的处理后文件路径
    
    Args:
        year: 年份
        quarter: 季度标识
        table_name: 表名（DEMO/DRUG/INDI/RPSR/THER）
        
    Returns:
        完整的文件路径
        
    Raises:
        ValueError: 当传入不支持的表名时抛出异常
    """
    table_key = str(table_name).upper()
    # 不同表对应不同的文件名前缀
    stem_by_table = {
        "DEMO": "case_base_dataset",
        "DRUG": "drug",
        "INDI": "indi",
        "RPSR": "rpsr",
        "THER": "ther",
    }
    if table_key not in stem_by_table:
        raise ValueError(f"Unsupported processed case table: {table_name}")
    # INDI、RPSR、THER 表需要在文件名后添加 "_case" 后缀
    suffix = "_case" if table_key in {"INDI", "RPSR", "THER"} else ""
    return output_quarter_dir(year) / f"{stem_by_table[table_key]}_{quarter_token(year, quarter)}{suffix}.parquet"


def require_processed_case_file(year: int, quarter: str, table_name: str) -> Path:
    """
    获取病例表文件路径，如果文件不存在则抛出异常
    
    Args:
        year: 年份
        quarter: 季度标识
        table_name: 表名
        
    Returns:
        文件路径
        
    Raises:
        FileNotFoundError: 当文件不存在时抛出异常，提示需要先运行 faers_project/year_batch_runner.py
    """
    path = processed_case_file(year, quarter, table_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned {table_name} input: {path}. "
            "Run faers_project/year_batch_runner.py first so ML-v2 reads only main cleaned outputs."
        )
    return path


def load_clean_demo(year: int, quarter: str) -> pd.DataFrame:
    """
    加载清洗后的 DEMO 表数据
    
    Args:
        year: 年份
        quarter: 季度标识
        
    Returns:
        DEMO 表的 DataFrame
    """
    return pd.read_parquet(require_processed_case_file(year, quarter, "DEMO"))


def load_clean_case_table(year: int, quarter: str, table_name: str) -> pd.DataFrame:
    """
    加载清洗后的病例相关表数据
    
    Args:
        year: 年份
        quarter: 季度标识
        table_name: 表名（DRUG/INDI/RPSR/THER）
        
    Returns:
        清洗后的 DataFrame，已过滤掉空 caseid 的记录
    """
    df = pd.read_parquet(require_processed_case_file(year, quarter, table_name))
    df["caseid"] = clean_caseid(df["caseid"])
    # 过滤掉 caseid 为空的记录
    return df[df["caseid"] != ""].copy()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """
    将 DataFrame 写入 CSV 文件
    
    Args:
        df: 要保存的 DataFrame
        path: 文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    将 DataFrame 写入 Parquet 文件
    
    Args:
        df: 要保存的 DataFrame
        path: 文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def summarize_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    生成数据集的基本统计摘要
    
    Args:
        df: 要统计的 DataFrame
        dataset: 数据集名称标识
        
    Returns:
        包含以下指标的摘要 DataFrame：
        - n_rows: 总行数
        - unique_caseid: 唯一 caseid 数量（如果存在 caseid 列）
        - duplicate_caseid_rows: 重复 caseid 的行数（如果存在 caseid 列）
    """
    rows = [
        {
            "dataset": dataset,
            "metric": "n_rows",
            "value": int(len(df)),
        },
        {
            "dataset": dataset,
            "metric": "unique_caseid",
            "value": int(df["caseid"].nunique()) if "caseid" in df.columns else None,
        },
        {
            "dataset": dataset,
            "metric": "duplicate_caseid_rows",
            "value": int(df.duplicated("caseid").sum()) if "caseid" in df.columns else None,
        },
    ]
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    生成数据集的缺失值统计表
    
    Args:
        df: 要分析的 DataFrame
        dataset: 数据集名称标识
        
    Returns:
        包含每列缺失情况的 DataFrame，包括：
        - n_rows: 总行数
        - missing_n: 缺失值数量（NaN）
        - blank_n: 空字符串数量（仅针对字符串列）
        - missing_or_blank_rate: 缺失或空值的比例
    """
    rows = []
    for column in df.columns:
        missing = int(df[column].isna().sum())
        blank = 0
        # 仅对字符串类型列统计空字符串数量
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            blank = int(df[column].fillna("").astype(str).str.strip().eq("").sum())
        rows.append(
            {
                "dataset": dataset,
                "column": column,
                "n_rows": int(len(df)),
                "missing_n": missing,
                "blank_n": blank,
                "missing_or_blank_rate": (missing + blank) / len(df) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def concat_existing(paths: list[Path]) -> pd.DataFrame:
    """
    合并存在的 Parquet 文件
    
    Args:
        paths: Parquet 文件路径列表
        
    Returns:
        合并后的 DataFrame，如果没有任何文件存在则返回空 DataFrame
    """
    existing = [path for path in paths if path.exists()]
    if not existing:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in existing), ignore_index=True)


def latest_meddra_excel() -> Path:
    """
    获取最新的 MedDRA Excel 文件路径
    
    Returns:
        按文件名排序后的第一个 MedDRA*.xlsx 文件路径
        
    Raises:
        FileNotFoundError: 当项目根目录下找不到任何 MedDRA*.xlsx 文件时抛出异常
    """
    candidates = sorted(PROJECT_ROOT.glob("MedDRA*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No MedDRA*.xlsx file found in project root.")
    return candidates[0]


def boundary_pattern(terms: list[str]) -> str:
    """
    构建用于匹配术语边界的正则表达式模式
    
    Args:
        terms: 术语列表
        
    Returns:
        正则表达式字符串，使用负向断言确保匹配完整的单词边界
        格式：(?<![A-Z0-9])(?:term1|term2|...)(?![A-Z0-9])
        其中术语按长度降序排列以确保优先匹配长术语
    """
    escaped = sorted({re.escape(term) for term in terms if term}, key=len, reverse=True)
    if not escaped:
        return r"a^"  # 返回不匹配任何内容的模式
    return rf"(?<![A-Z0-9])(?:{'|'.join(escaped)})(?![A-Z0-9])"
