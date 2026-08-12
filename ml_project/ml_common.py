"""
FAERS机器学习项目通用模块

本模块提供了FAERS(FDA不良事件报告系统)数据分析的机器学习通用功能,包括:
- 数据加载与预处理
- 特征工程
- 模型训练与超参数搜索
- 模型评估与校准
- 结果保存与可视化

支持两种特征版本(v1和v2),多种队列筛选策略,以及时间序列划分。
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================== 路径配置 ====================

# 获取项目根目录(当前文件的父目录的父目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import (
    DatasetBundle,
    extract_token,
    resolve_signal_feature_bundle,
    token_sort_key,
)

# 默认ML运行根目录
DEFAULT_ML_RUN_ROOT = PROJECT_ROOT / "runs" / "mainline_2026-05-20"

# 全局数据集目录路径(可通过环境变量FAERS_GLOBAL_DATASET_DIR覆盖)
GLOBAL_DATASET_DIR = Path(
    os.environ.get(
        "FAERS_GLOBAL_DATASET_DIR",
        DEFAULT_ML_RUN_ROOT / "OUTPUT_GLOBAL" / "datasets",
    )
)

# ML输出根目录(可通过环境变量FAERS_ML_OUTPUT_ROOT覆盖)
OUTPUT_ML_ROOT = Path(
    os.environ.get("FAERS_ML_OUTPUT_ROOT", DEFAULT_ML_RUN_ROOT / "OUTPUT_ML")
)

# V2特征数据集目录
FEATURE_V2_DATASET_DIR = OUTPUT_ML_ROOT / "features_v2" / "datasets"

# ==================== 常量定义 ====================

# 可选的目标变量:is_fall(跌倒/坠落事件)或 serious(严重事件)
TARGET_OPTIONS = ("is_fall", "serious")

# 超参数搜索模式:none(不搜索)、fast(快速搜索)、full(完整搜索)
SEARCH_MODES = ("none", "fast", "full")

# 研究队列选项:all(全部)、zolpidem(仅唑吡坦)、zdrug(所有Z类药物)
COHORT_OPTIONS = ("all", "zolpidem", "zdrug")

# 特征版本选项:v1(基础版本)、v2(增强版本)
FEATURE_VERSION_OPTIONS = ("v1", "v2")

# 特征集合选项:core(核心特征)、enhanced(增强特征,包含表型字段)
FEATURE_SET_OPTIONS = ("core", "enhanced")

# ==================== V1版本特征定义 ====================

# V1布尔特征列表:药物类别、多药用药标志等
V1_BOOL_FEATURES = [
    "is_zolpidem",          # 是否使用唑吡坦
    "is_zaleplon",          # 是否使用扎来普隆
    "is_zopiclone",         # 是否使用佐匹克隆
    "is_eszopiclone",       # 是否使用右佐匹克隆
    "is_benzo",             # 是否使用苯二氮䓬类药物
    "is_antidepressant",    # 是否使用抗抑郁药
    "is_antipsychotic",     # 是否使用抗精神病药
    "is_opioid",            # 是否使用阿片类药物
    "is_antiepileptic",     # 是否使用抗癫痫药
    "polypharmacy_5",       # 是否多药用药(≥5种)
    "is_other_zdrug",       # 是否使用其他Z类药物
    "multiple_zdrug",       # 是否使用多种Z类药物
    "any_cns_coprescription",  # 是否有中枢神经系统药物合并用药
    "high_drug_burden_10",  # 高药物负担(≥10种不同药物)
    "very_high_drug_burden_20",  # 极高药物负担(≥20种不同药物)
]

# V1数值特征列表:年份、药物数量等
V1_NUMERIC_FEATURES = [
    "year",                 # 报告年份
    "drug_n",               # 药物总数
    "distinct_drug_n",      # 不同药物数量
    "log_drug_n",           # 药物数量的对数变换
    "log_distinct_drug_n",  # 不同药物数量的对数变换
    "zdrug_count",          # Z类药物数量
    "cns_coprescription_count",  # 中枢神经系统合并用药数量
]

# V1分类特征列表:年龄组、性别、季度等
V1_CATEGORICAL_FEATURES = [
    "age_group",            # 年龄组
    "sex_clean",            # 性别(清洗后)
    "quarter",              # 季度
    "drug_n_bucket",        # 药物数量分桶
    "distinct_drug_n_bucket",  # 不同药物数量分桶
    "cns_coprescription_bucket",  # 中枢神经系统合并用药分桶
]

# ==================== V2版本特征定义 ====================

# V2基础布尔特征:事件日期、药物角色、适应症、表型等
V2_BASE_BOOL_FEATURES = [
    "event_date_known",           # 事件日期是否已知
    "has_ps_drug",                # 是否有主要怀疑药物
    "has_ss_drug",                # 是否有次要怀疑药物
    "zolpidem_as_ps",             # 唑吡坦作为主要怀疑药物
    "zolpidem_as_suspect",        # 唑吡坦作为怀疑药物
    "other_zdrug_as_suspect",     # 其他Z类药物作为怀疑药物
    "indi_insomnia",              # 适应症:失眠
    "indi_anxiety",               # 适应症:焦虑
    "indi_depression",            # 适应症:抑郁
    "indi_pain",                  # 适应症:疼痛
    "indi_epilepsy",              # 适应症:癫痫
    "indi_dizziness_vertigo",     # 适应症:头晕/眩晕
    "has_rpsr",                   # 是否有重新处方/再次使用信息
    "has_start_dt",               # 是否有开始日期
    "has_end_dt",                 # 是否有结束日期
    "duration_known",             # 用药持续时间是否已知
    "pheno_sedation_somnolence",  # 表型:镇静/嗜睡
    "pheno_consciousness_cognition",  # 表型:意识/认知障碍
    "pheno_dizziness_vertigo_syncope",  # 表型:头晕/眩晕/晕厥
    "pheno_gait_balance_motor",   # 表型:步态/平衡/运动障碍
    "pheno_hypotension",          # 表型:低血压
    "pheno_visual_disturbance",   # 表型:视觉障碍
]

# V2基础数值特征:年龄、药物数量、适应症数量等
V2_BASE_NUMERIC_FEATURES = [
    "age_years",              # 年龄(岁)
    "ps_drug_n",              # 主要怀疑药物数量
    "ss_drug_n",              # 次要怀疑药物数量
    "concomitant_drug_n",     # 合并用药数量
    "interacting_drug_n",     # 相互作用药物数量
    "indi_n",                 # 适应症总数
    "distinct_indi_n",        # 不同适应症数量
    "indi_mapped_n",          # 已映射适应症数量
    "indi_unmapped_n",        # 未映射适应症数量
    "therapy_record_n",       # 治疗记录数量
]

# V2基础分类特征:报告类型、国家代码等
V2_BASE_CATEGORICAL_FEATURES = [
    "rept_cod",               # 报告类型代码
    "e_sub",                  # 提交者类型
    "reporter_country",       # 报告者国家
    "occr_country",           # 发生国家
    "rpsr_cod",               # 重新处方代码
]

# ==================== 全局特征变量(运行时动态配置)====================

# 当前使用的布尔特征列表(默认为V1)
BOOL_FEATURES = V1_BOOL_FEATURES.copy()

# 当前使用的数值特征列表(默认为V1)
NUMERIC_FEATURES = V1_NUMERIC_FEATURES.copy()

# 当前使用的分类特征列表(默认为V1)
CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES.copy()

# 模型使用的全部特征列表(分类+数值+布尔)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES

# ==================== 搜索与评估配置 ====================

# 超参数搜索使用的评分指标
SEARCH_SCORING = {
    "average_precision": "average_precision",  # 平均精度
    "roc_auc": "roc_auc",                      # ROC曲线下面积
    "neg_brier_score": "neg_brier_score",      # 负Brier分数(越小越好)
}

# 用于选择最佳模型的指标
REFIT_METRIC = "average_precision"

# 并行搜索的工作进程数(可通过环境变量FAERS_SEARCH_N_JOBS设置)
SEARCH_N_JOBS = int(os.environ.get("FAERS_SEARCH_N_JOBS", "1"))

# 并行后端类型(可通过环境变量FAERS_SEARCH_BACKEND设置,默认为loky)
SEARCH_BACKEND = os.environ.get("FAERS_SEARCH_BACKEND", "loky").strip()

# 最终评估使用的指标列表
EVALUATION_METRICS = [
    "roc_auc",              # ROC-AUC
    "average_precision",    # 平均精度
    "brier_score",          # Brier分数
    "accuracy",             # 准确率
    "precision",            # 精确率
    "recall",               # 召回率
    "f1",                   # F1分数
    "specificity",          # 特异度
    "mcc",                  # Matthews相关系数
]


# ==================== 数据类定义 ====================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    实验配置数据类
    
    存储机器学习实验的所有配置参数,包括数据集、特征、目标变量、
    时间划分、采样、搜索策略等。
    
    Attributes:
        period_token: 数据集时期标记(如"2004_2025")
        feature_version: 特征版本("v1"或"v2")
        feature_set: 特征集合("core"或"enhanced")
        target_col: 目标变量列名
        cohort: 研究队列类型
        train_end_year: 训练集截止年份
        valid_year: 验证集年份
        test_year: 测试集年份
        train_sample_n: 训练样本数量(0表示使用全部数据)
        search_mode: 超参数搜索模式
        cv_folds: 交叉验证折数
        bootstrap_iterations: Bootstrap迭代次数
        random_state: 随机种子
    """
    period_token: str | None
    feature_version: str
    feature_set: str
    target_col: str
    cohort: str
    train_end_year: int
    valid_year: int
    test_year: int
    train_sample_n: int
    search_mode: str
    cv_folds: int
    bootstrap_iterations: int
    random_state: int


@dataclass(frozen=True)
class SearchSpec:
    """
    超参数搜索规格数据类
    
    定义超参数搜索的策略、参数空间和迭代次数。
    
    Attributes:
        strategy: 搜索策略("grid"网格搜索或"random"随机搜索)
        param_space_by_mode: 不同搜索模式下的参数空间字典
        n_iter_by_mode: 不同搜索模式下的迭代次数(仅随机搜索需要)
    """
    strategy: Literal["grid", "random"]
    param_space_by_mode: dict[str, dict[str, list[Any]] | list[dict[str, list[Any]]]]
    n_iter_by_mode: dict[str, int] | None = None


@dataclass
class ExperimentResult:
    """
    实验结果数据类
    
    存储完整的机器学习实验结果,包括数据、模型、评估指标等。
    
    Attributes:
        config: 实验配置
        bundle: 数据集包
        run_dir: 输出目录路径
        train_full_df: 完整训练集DataFrame
        train_df: 实际使用的训练集DataFrame(可能经过采样)
        valid_df: 验证集DataFrame
        test_df: 测试集DataFrame
        pipeline: 训练好的Pipeline对象
        search_summary: 超参数搜索摘要
        search_results_df: 搜索结果DataFrame(可选)
        cv_metrics_df: 交叉验证指标DataFrame
        cv_summary: 交叉验证摘要
        threshold_selection: 阈值选择结果
        validation_metrics: 验证集评估指标
        test_metrics: 测试集评估指标
        validation_metrics_raw: 验证集原始概率评估指标
        test_metrics_raw: 测试集原始概率评估指标
        valid_raw_scores: 验证集原始预测概率
        valid_scores: 验证集校准后预测概率
        test_raw_scores: 测试集原始预测概率
        test_scores: 测试集校准后预测概率
    """
    config: ExperimentConfig
    bundle: DatasetBundle
    run_dir: Path
    train_full_df: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    pipeline: Pipeline
    search_summary: dict[str, Any]
    search_results_df: pd.DataFrame | None
    cv_metrics_df: pd.DataFrame
    cv_summary: dict[str, Any]
    threshold_selection: dict[str, float]
    validation_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    validation_metrics_raw: dict[str, Any]
    test_metrics_raw: dict[str, Any]
    valid_raw_scores: np.ndarray
    valid_scores: np.ndarray
    test_raw_scores: np.ndarray
    test_scores: np.ndarray


# ==================== 工具函数 ====================

def log_step(message: str) -> None:
    """
    打印带前缀的日志消息
    
    Args:
        message: 要打印的消息内容
    """
    print(f"[ml] {message}", flush=True)


def format_duration(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串
    
    Args:
        seconds: 时间长度(秒)
        
    Returns:
        格式化后的时间字符串,如 "1.5s"、"3m 25.3s"、"2h 15m 30.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


@contextmanager
def timed_step(message: str):
    """
    上下文管理器:计时并记录步骤执行时间
    
    用法示例:
        with timed_step("Loading data"):
            data = load_data()
    
    Args:
        message: 步骤描述消息
        
    Yields:
        None
        
    Raises:
        Exception: 如果步骤执行失败,会记录失败时间并重新抛出异常
    """
    start = perf_counter()
    log_step(f"{message} ...")
    try:
        yield
    except Exception:
        log_step(f"{message} failed after {format_duration(perf_counter() - start)}")
        raise
    else:
        log_step(f"{message} done in {format_duration(perf_counter() - start)}")


def format_metric(value: Any, digits: int = 4) -> str:
    """
    格式化指标值为固定小数位数的字符串
    
    Args:
        value: 指标值(可以是None、数字或其他类型)
        digits: 小数位数,默认为4
        
    Returns:
        格式化后的字符串,无效值返回"NA"
    """
    if value is None:
        return "NA"
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric_value):
        return "NA"
    return f"{numeric_value:.{digits}f}"


def _positive_summary(df: pd.DataFrame, target_col: str) -> tuple[int, float]:
    """
    计算DataFrame中正例的数量和比例
    
    Args:
        df: 输入DataFrame
        target_col: 目标变量列名
        
    Returns:
        元组(正例数量,正例比例)
    """
    positives = int(df[target_col].astype(int).sum())
    rate = float(df[target_col].astype(int).mean()) if len(df) else 0.0
    return positives, rate


def log_frame_summary(label: str, df: pd.DataFrame, target_col: str) -> None:
    """
    打印DataFrame的摘要信息(行数、正例数、正例率、年份范围)
    
    Args:
        label: 标签名称(如"Train"、"Validation"、"Test")
        df: 输入DataFrame
        target_col: 目标变量列名
    """
    positives, rate = _positive_summary(df, target_col)
    year_text = ""
    if "year" in df.columns and not df.empty:
        year_text = f", years={int(df['year'].min())}-{int(df['year'].max())}"
    log_step(
        f"{label}: rows={len(df):,}, positives={positives:,}, "
        f"positive_rate={rate:.4%}{year_text}"
    )


def add_common_arguments(
    parser: Any,
    *,
    default_train_sample_n: int,
    default_search_mode: str,
) -> None:
    """
    向ArgumentParser添加通用的命令行参数
    
    这些参数适用于所有ML实验脚本,包括数据集选择、目标变量、
    特征版本、队列筛选、时间划分、采样、搜索策略等。
    
    Args:
        parser: argparse.ArgumentParser对象
        default_train_sample_n: 默认训练样本数量
        default_search_mode: 默认搜索模式
    """
    parser.add_argument(
        "--period-token",
        default=None,
        help="数据集时期标记,如 2004_2025。默认为最新可用的数据集包。",
    )
    parser.add_argument(
        "--target-col",
        default="is_fall",
        choices=TARGET_OPTIONS,
        help=(
            "要预测的目标变量列名。is_fall表示FAERS PT为FALL或DROP ATTACKS的事件。"
        ),
    )
    parser.add_argument(
        "--feature-version",
        default="v1",
        choices=FEATURE_VERSION_OPTIONS,
        help="特征表版本。v1使用当前全局数据集;v2使用OUTPUT_ML/features_v2/datasets。",
    )
    parser.add_argument(
        "--feature-set",
        default="enhanced",
        choices=FEATURE_SET_OPTIONS,
        help=(
            "要使用的特征子集。core排除REAC表型字段;"
            "enhanced包含经过泄漏筛查的表型字段(如果可用)。"
        ),
    )
    parser.add_argument(
        "--cohort",
        default="all",
        choices=COHORT_OPTIONS,
        help=(
            "研究人群。all保留所有符合条件的老年病例;"
            "zolpidem保留唑吡坦暴露病例;zdrug保留任何Z类药物暴露病例。"
        ),
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2023,
        help="使用截止到该年份的所有案例进行模型训练。",
    )
    parser.add_argument(
        "--valid-year",
        type=int,
        default=2024,
        help="用于校准和阈值选择的验证年份。",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        help="仅用于最终评估的保留测试年份。",
    )
    parser.add_argument(
        "--train-sample-n",
        type=int,
        default=default_train_sample_n,
        help="可选的分层训练样本大小。使用0保留完整训练集。",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default=default_search_mode,
        help="超参数搜索深度。none跳过调优,fast是小规模搜索,full是完整配置的搜索。",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="训练期内使用的交叉验证折数。",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="用于最终测试集置信区间的Bootstrap迭代次数。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="用于采样、调优和模型拟合的随机种子。",
    )


def config_from_args(args: Any) -> ExperimentConfig:
    """
    从命令行参数创建ExperimentConfig对象
    
    Args:
        args: argparse解析后的参数对象
        
    Returns:
        ExperimentConfig配置对象
    """
    return ExperimentConfig(
        period_token=args.period_token,
        feature_version=args.feature_version,
        feature_set=args.feature_set,
        target_col=args.target_col,
        cohort=args.cohort,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
        train_sample_n=args.train_sample_n,
        search_mode=args.search_mode,
        cv_folds=args.cv_folds,
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
    )


# ==================== 数据集处理函数 ====================

def resolve_dataset_bundle(
    dataset_dir: Path = GLOBAL_DATASET_DIR,
    period_token: str | None = None,
    feature_version: str = "v1",
) -> DatasetBundle:
    """
    解析并返回数据集包
    
    根据特征版本和时期标记,定位并返回对应的数据集包。
    V2版本从FEATURE_V2_DATASET_DIR读取,V1版本从全局数据集目录读取。
    
    Args:
        dataset_dir: 数据集目录路径(仅V1使用)
        period_token: 时期标记(如"2004_2025"),None则自动选择最新的
        feature_version: 特征版本("v1"或"v2")
        
    Returns:
        DatasetBundle数据集包对象
        
    Raises:
        FileNotFoundError: 找不到对应的数据集文件
    """
    if feature_version == "v2":
        # V2版本:从FEATURE_V2_DATASET_DIR查找特征文件
        feature_files = sorted(FEATURE_V2_DATASET_DIR.glob("ml_feature_v2_*.parquet"))
        if not feature_files:
            raise FileNotFoundError(
                f"No ML-v2 feature dataset found in {FEATURE_V2_DATASET_DIR}. "
                "Run ml_project/features_v2/07_build_ml_feature_v2.py first."
            )
        # 提取每个文件的时期标记
        feature_by_token = {
            extract_token(path, "ml_feature_v2_"): path for path in feature_files
        }
        # 选择指定的或最新的时期标记
        selected_token = period_token or max(feature_by_token, key=token_sort_key)
        if selected_token not in feature_by_token:
            raise FileNotFoundError(
                f"ML-v2 period token not found in {FEATURE_V2_DATASET_DIR}: {selected_token}"
            )
        selected_file = feature_by_token[selected_token]
        return DatasetBundle(
            period_token=selected_token,
            signal_file=selected_file,
            feature_file=selected_file,
            feature_version="v2",
        )

    # V1版本:使用传统的信号+特征数据集
    return resolve_signal_feature_bundle(
        dataset_dir=dataset_dir,
        period_token=period_token,
    )


def apply_cohort_filter(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """
    应用队列筛选器
    
    根据指定的队列类型筛选数据:
    - all: 保留所有符合条件的病例
    - zolpidem: 仅保留唑吡坦暴露的病例
    - zdrug: 保留任何Z类药物(唑吡坦、扎来普隆、佐匹克隆、右佐匹克隆)暴露的病例
    
    Args:
        df: 输入DataFrame
        cohort: 队列类型("all"、"zolpidem"或"zdrug")
        
    Returns:
        筛选后的DataFrame副本
        
    Raises:
        ValueError: 不支持的队列类型或筛选结果为空
    """
    if cohort not in COHORT_OPTIONS:
        raise ValueError(f"Unsupported cohort: {cohort}")

    if cohort == "all":
        log_step(f"Cohort filter: all eligible cases kept ({len(df):,} rows)")
        return df.copy()

    if cohort == "zolpidem":
        filtered = df[df["is_zolpidem"]].copy()
        label = "zolpidem-exposed"
    else:
        zdrug_cols = ["is_zolpidem", "is_zaleplon", "is_zopiclone", "is_eszopiclone"]
        filtered = df[df[zdrug_cols].any(axis=1)].copy()
        label = "any Z-drug-exposed"

    if filtered.empty:
        raise ValueError(f"Cohort filter produced no rows: {cohort}")
    log_step(
        f"Cohort filter: {label} cases kept ({len(filtered):,} of {len(df):,} rows)"
    )
    return filtered


def configure_feature_schema(
    feature_version: str,
    available_columns: list[str] | None = None,
    feature_set: str = "enhanced",
) -> None:
    """
    配置特征模式
    
    根据特征版本和特征集合选项,动态设置全局特征变量
    (BOOL_FEATURES、NUMERIC_FEATURES、CATEGORICAL_FEATURES、MODEL_FEATURES)。
    
    V1版本:使用预定义的V1特征列表
    V2版本:基于可用列动态构建特征列表,可选择是否包含表型字段
    
    Args:
        feature_version: 特征版本("v1"或"v2")
        available_columns: 可用列名列表(仅V2需要)
        feature_set: 特征集合("core"或"enhanced")
        
    Raises:
        ValueError: 不支持的特征版本或特征集合
    """
    global BOOL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
    if feature_set not in FEATURE_SET_OPTIONS:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    if feature_version == "v1":
        # V1版本:直接使用预定义的特征列表
        BOOL_FEATURES = V1_BOOL_FEATURES.copy()
        NUMERIC_FEATURES = V1_NUMERIC_FEATURES.copy()
        CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES.copy()
    elif feature_version == "v2":
        # V2版本:基于可用列动态构建特征列表
        available = set(available_columns or [])
        # 动态SOC适应症特征(以indi_soc_开头的列)
        dynamic_soc_features = sorted(
            column for column in available if column.startswith("indi_soc_")
        )
        # 筛选可用的V2布尔特征
        v2_bool_features = [column for column in V2_BASE_BOOL_FEATURES if column in available]
        # core模式排除表型字段
        if feature_set == "core":
            v2_bool_features = [
                column for column in v2_bool_features if not column.startswith("pheno_")
            ]
        BOOL_FEATURES = (
            V1_BOOL_FEATURES
            + v2_bool_features
            + dynamic_soc_features
        )
        NUMERIC_FEATURES = V1_NUMERIC_FEATURES + [
            column for column in V2_BASE_NUMERIC_FEATURES if column in available
        ]
        CATEGORICAL_FEATURES = V1_CATEGORICAL_FEATURES + [
            column for column in V2_BASE_CATEGORICAL_FEATURES if column in available
        ]
    else:
        raise ValueError(f"Unsupported feature version: {feature_version}")

    # 更新模型特征列表
    MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOL_FEATURES


# ==================== 数据加载函数 ====================

def load_modeling_frame(
    bundle: DatasetBundle,
    target_col: str,
    cohort: str,
    feature_set: str = "enhanced",
) -> pd.DataFrame:
    """
    加载建模数据框(V1版本)
    
    从信号数据集和特征数据集中加载数据,进行合并、清洗、特征工程,
    最后应用队列筛选。
    
    Args:
        bundle: 数据集包对象
        target_col: 目标变量列名
        cohort: 队列类型
        feature_set: 特征集合("core"或"enhanced")
        
    Returns:
        准备好的建模DataFrame,包含caseid、目标变量和所有特征列
        
    Raises:
        ValueError: 不支持的目标变量列名
    """
    if target_col not in TARGET_OPTIONS:
        raise ValueError(f"Unsupported target column: {target_col}")

    # V2版本委托给专用函数处理
    if bundle.feature_version == "v2":
        return load_modeling_frame_v2(
            bundle,
            target_col=target_col,
            cohort=cohort,
            feature_set=feature_set,
        )

    # 配置V1特征模式
    configure_feature_schema("v1", feature_set=feature_set)

    # 定义需要从Parquet文件读取的列
    raw_bool_features = [
        "is_zolpidem",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "polypharmacy_5",
    ]
    raw_numeric_features = ["drug_n", "distinct_drug_n"]
    
    # 处理目标变量列名(兼容is_fall和is_fall_narrow)
    source_target_col = target_col
    if target_col == "is_fall":
        import pyarrow.parquet as pq

        signal_columns_available = set(pq.read_schema(bundle.signal_file).names)
        if "is_fall" not in signal_columns_available and "is_fall_narrow" in signal_columns_available:
            source_target_col = "is_fall_narrow"

    # 定义要从信号文件和特征文件读取的列
    signal_columns = list(
        dict.fromkeys(["caseid", source_target_col, "age_group", "sex_clean", "quarter", "year"])
    )
    feature_columns = list(
        dict.fromkeys(["caseid", *raw_numeric_features, *raw_bool_features])
    )

    log_step(
        f"Loading signal dataset: {bundle.signal_file.name} and feature dataset: {bundle.feature_file.name}"
    )
    # 读取Parquet文件
    signal_df = pd.read_parquet(bundle.signal_file, columns=signal_columns)
    feature_df = pd.read_parquet(bundle.feature_file, columns=feature_columns)

    # 清洗caseid:转为字符串并去除空格
    signal_df["caseid"] = signal_df["caseid"].astype(str).str.strip()
    feature_df["caseid"] = feature_df["caseid"].astype(str).str.strip()

    # 去除重复的caseid
    signal_df = signal_df.drop_duplicates(subset=["caseid"]).copy()
    feature_df = feature_df.drop_duplicates(subset=["caseid"]).copy()

    # 基于caseid内连接两个DataFrame
    merged = signal_df.merge(feature_df, on="caseid", how="inner")
    merged = merged[merged["caseid"] != ""].copy()

    # 如果源目标列名不同,进行重命名
    if source_target_col != target_col:
        merged[target_col] = merged[source_target_col]

    # 填充布尔特征的缺失值为False
    for col in raw_bool_features + [target_col]:
        merged[col] = merged[col].fillna(False).astype(bool)

    # 转换数值特征,缺失值填充为0
    for col in ["year", *raw_numeric_features]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

    # 添加衍生特征
    merged = add_derived_features(merged)

    # 处理分类特征:缺失值填充为"unknown"
    for col in CATEGORICAL_FEATURES:
        merged[col] = (
            merged[col]
            .where(merged[col].notna(), "unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    # 选择最终需要的列
    final_df = merged[["caseid", target_col, *MODEL_FEATURES]].copy()
    # 应用队列筛选
    final_df = apply_cohort_filter(final_df, cohort=cohort)
    log_step(f"Modeling frame ready with {len(final_df):,} rows")
    return final_df


def load_modeling_frame_v2(
    bundle: DatasetBundle,
    target_col: str,
    cohort: str,
    feature_set: str = "enhanced",
) -> pd.DataFrame:
    """
    加载建模数据框(V2版本)
    
    从ML-V2特征数据集中加载数据,进行数据清洗、泄漏检查、特征配置,
    最后应用队列筛选。
    
    Args:
        bundle: 数据集包对象
        target_col: 目标变量列名
        cohort: 队列类型
        feature_set: 特征集合("core"或"enhanced")
        
    Returns:
        准备好的建模DataFrame
        
    Raises:
        ValueError: 缺少必要的列或包含泄漏列
    """
    log_step(f"Loading ML-v2 feature dataset: {bundle.feature_file.name}")
    df = pd.read_parquet(bundle.feature_file)
    
    # 处理目标变量列名兼容性
    if target_col == "is_fall" and "is_fall" not in df.columns and "is_fall_narrow" in df.columns:
        df[target_col] = df["is_fall_narrow"]
    # 删除不需要的目标变量列
    df = df.drop(
        columns=[
            col
            for col in ["is_fall_narrow", "is_fall_broad"]
            if col in df.columns and col != target_col
        ]
    )
    if target_col not in df.columns:
        raise ValueError(f"ML-v2 feature dataset missing target column: {target_col}")

    # 检查是否存在泄漏列(不应出现在特征数据集中)
    leakage_cols = {"fall_pt_list", "fall_pt_count", "fall_narrow_pt_count"}
    present_leakage = sorted(leakage_cols & set(df.columns))
    if present_leakage:
        raise ValueError(f"ML-v2 feature dataset contains leakage columns: {present_leakage}")

    # 清洗caseid
    df["caseid"] = df["caseid"].astype(str).str.strip()
    df = df[df["caseid"] != ""].drop_duplicates(subset=["caseid"]).copy()

    # 检查必需的基础列是否存在
    base_required = [
        "is_zolpidem",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "polypharmacy_5",
        "drug_n",
        "distinct_drug_n",
        "age_group",
        "sex_clean",
        "quarter",
        "year",
    ]
    missing = [column for column in base_required if column not in df.columns]
    if missing:
        raise ValueError(f"ML-v2 feature dataset missing required base columns: {missing}")

    # 处理布尔特征
    for col in V1_BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    # 处理目标变量
    for col in [target_col]:
        df[col] = df[col].fillna(False).astype(bool)
    # 处理数值特征
    for col in ["year", "drug_n", "distinct_drug_n"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 添加衍生特征
    df = add_derived_features(df)
    # 配置V2特征模式
    configure_feature_schema("v2", available_columns=list(df.columns), feature_set=feature_set)

    # 确保所有特征列都存在,缺失则填充默认值
    for col in BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = (
            df[col]
            .where(df[col].notna(), "unknown")
            .astype(str)
            .str.strip()
            .replace("", "unknown")
        )

    # 选择最终需要的列
    final_df = df[["caseid", target_col, *MODEL_FEATURES]].copy()
    # 应用队列筛选
    final_df = apply_cohort_filter(final_df, cohort=cohort)
    log_step(
        f"ML-v2 modeling frame ready with {len(final_df):,} rows and {len(MODEL_FEATURES):,} features"
    )
    return final_df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加衍生特征
    
    基于原始特征计算新的特征,包括:
    - Z类药物相关:是否其他Z类药物、Z类药物计数、是否多种Z类药物
    - CNS合并用药:CNS药物计数、是否有CNS合并用药
    - 对数变换:药物数量的对数
    - 药物负担标志:高药物负担(≥10)、极高药物负担(≥20)
    - 分桶特征:药物数量、不同药物数量、CNS合并用药数量的分桶
    
    Args:
        df: 输入DataFrame
        
    Returns:
        添加了衍生特征的DataFrame副本
    """
    frame = df.copy()
    zdrug_cols = ["is_zolpidem", "is_zaleplon", "is_zopiclone", "is_eszopiclone"]
    cns_cols = [
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
    ]

    # 确保布尔列为布尔类型
    for col in zdrug_cols + cns_cols:
        frame[col] = frame[col].fillna(False).astype(bool)

    # Z类药物相关特征
    frame["is_other_zdrug"] = frame[["is_zaleplon", "is_zopiclone", "is_eszopiclone"]].any(axis=1)
    frame["zdrug_count"] = frame[zdrug_cols].sum(axis=1).astype(float)
    frame["multiple_zdrug"] = frame["zdrug_count"] >= 2
    
    # CNS合并用药相关特征
    frame["cns_coprescription_count"] = frame[cns_cols].sum(axis=1).astype(float)
    frame["any_cns_coprescription"] = frame["cns_coprescription_count"] >= 1

    # 对数变换特征
    frame["log_drug_n"] = np.log1p(frame["drug_n"].clip(lower=0))
    frame["log_distinct_drug_n"] = np.log1p(frame["distinct_drug_n"].clip(lower=0))
    
    # 药物负担标志
    frame["high_drug_burden_10"] = frame["distinct_drug_n"] >= 10
    frame["very_high_drug_burden_20"] = frame["distinct_drug_n"] >= 20

    # 分桶特征
    frame["drug_n_bucket"] = pd.cut(
        frame["drug_n"],
        bins=[-np.inf, 1, 2, 4, 9, 19, np.inf],
        labels=["0-1", "2", "3-4", "5-9", "10-19", "20+"],
    ).astype("string")
    frame["distinct_drug_n_bucket"] = pd.cut(
        frame["distinct_drug_n"],
        bins=[-np.inf, 1, 2, 4, 9, 19, np.inf],
        labels=["0-1", "2", "3-4", "5-9", "10-19", "20+"],
    ).astype("string")
    frame["cns_coprescription_bucket"] = pd.cut(
        frame["cns_coprescription_count"],
        bins=[-np.inf, 0, 1, 2, np.inf],
        labels=["0", "1", "2", "3+"],
    ).astype("string")
    return frame


# ==================== 数据划分函数 ====================

def sample_training_frame(
    df: pd.DataFrame,
    target_col: str,
    sample_n: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    分层采样训练集
    
    如果指定了样本数量且小于总行数,则进行分层采样以保持正负例比例。
    否则返回完整数据集。
    
    Args:
        df: 输入DataFrame
        target_col: 目标变量列名
        sample_n: 采样数量(None或≤0表示使用全部数据)
        random_state: 随机种子
        
    Returns:
        采样后的DataFrame副本
        
    Raises:
        ValueError: 训练集必须包含正负两类样本
    """
    if sample_n is None or sample_n <= 0 or len(df) <= sample_n:
        log_step(f"Training uses full dataset with {len(df):,} rows")
        return df.copy()

    labels = df[target_col].astype(int)
    if labels.nunique() < 2:
        raise ValueError("Training frame must contain both positive and negative cases.")

    try:
        # 尝试分层采样
        sampled_idx, _ = train_test_split(
            df.index.to_numpy(),
            train_size=sample_n,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError:
        # 如果分层采样失败,退化为随机采样
        sampled_idx = df.sample(
            n=sample_n, random_state=random_state, replace=False
        ).index.to_numpy()

    sampled = df.loc[sampled_idx].copy()
    sampled = sampled.sort_values(["year", "caseid"]).reset_index(drop=True)

    if sampled[target_col].astype(int).sum() == 0:
        raise ValueError(
            "Training sample contains no positive cases. Increase --train-sample-n."
        )
    log_step(
        f"Training sampled down from {len(df):,} to {len(sampled):,} rows with stratification"
    )
    return sampled


def temporal_split(
    df: pd.DataFrame,
    train_end_year: int,
    valid_year: int,
    test_year: int,
) -> dict[str, pd.DataFrame]:
    """
    按时间划分数据集
    
    根据年份将数据划分为训练集、验证集和测试集:
    - 训练集:年份 <= train_end_year
    - 验证集:年份 == valid_year
    - 测试集:年份 == test_year
    
    Args:
        df: 输入DataFrame(必须包含"year"列)
        train_end_year: 训练集截止年份
        valid_year: 验证集年份
        test_year: 测试集年份
        
    Returns:
        字典,包含"train"、"valid"、"test"三个键对应的DataFrame
        
    Raises:
        ValueError: 任一分区为空
    """
    train_df = df[df["year"] <= train_end_year].copy()
    valid_df = df[df["year"] == valid_year].copy()
    test_df = df[df["year"] == test_year].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty partition. "
            f"train_end_year={train_end_year}, valid_year={valid_year}, test_year={test_year}"
        )
    log_step(
        "Temporal split ready: "
        f"train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}"
    )
    return {"train": train_df, "valid": valid_df, "test": test_df}


# ==================== Pipeline构建函数 ====================

def build_preprocessor() -> ColumnTransformer:
    """
    构建数据预处理器
    
    创建ColumnTransformer,对不同特征类型应用不同的预处理:
    - 分类特征:OneHotEncoder(独热编码)
    - 数值特征:StandardScaler(标准化,不中心化以保留稀疏性)
    - 布尔特征:passthrough(直接传递,不做处理)
    
    Returns:
        ColumnTransformer预处理器对象
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(with_mean=False), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """
    构建机器学习Pipeline
    
    将预处理器和估计器组合成Pipeline。
    
    Args:
        estimator: 机器学习估计器(如LogisticRegression)
        
    Returns:
        Pipeline对象,包含预处理器和模型两个步骤
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    获取Pipeline预处理后的特征名称
    
    Args:
        pipeline: 已拟合的Pipeline对象
        
    Returns:
        特征名称列表
    """
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())


# ==================== 交叉验证辅助函数 ====================

def _determine_cv_folds(y: pd.Series | np.ndarray, requested_folds: int) -> int:
    """
    确定实际的交叉验证折数
    
    根据正负样本数量调整折数,确保每折都有足够的正负样本。
    
    Args:
        y: 目标变量数组
        requested_folds: 请求的折数
        
    Returns:
        实际使用的折数
        
    Raises:
        ValueError: 正负样本不足2个
    """
    y_arr = np.asarray(pd.Series(y).astype(int))
    positives = int(y_arr.sum())
    negatives = int(len(y_arr) - positives)
    folds = min(requested_folds, positives, negatives)
    if folds < 2:
        raise ValueError("Cross-validation requires at least 2 positives and 2 negatives.")
    return folds


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    安全计算ROC-AUC(处理单类情况)
    
    Args:
        y_true: 真实标签数组
        y_score: 预测概率数组
        
    Returns:
        ROC-AUC值,如果只有一类则返回None
    """
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    安全计算平均精度(处理单类情况)
    
    Args:
        y_true: 真实标签数组
        y_score: 预测概率数组
        
    Returns:
        平均精度值,如果只有一类则返回None
    """
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _safe_brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """
    安全计算Brier分数(处理单类情况)
    
    Args:
        y_true: 真实标签数组
        y_score: 预测概率数组
        
    Returns:
        Brier分数值,如果只有一类则返回None
    """
    if np.unique(y_true).size < 2:
        return None
    return float(brier_score_loss(y_true, y_score))


def _top_risk_metrics(y_true_arr: np.ndarray, y_score_arr: np.ndarray) -> dict[str, Any]:
    """
    计算高风险群体的指标
    
    分析预测概率最高的5%和10%群体中的正例率和提升度。
    
    Args:
        y_true_arr: 真实标签数组
        y_score_arr: 预测概率数组
        
    Returns:
        包含高风险群体指标的字典
    """
    rows: dict[str, Any] = {}
    if len(y_true_arr) == 0:
        return rows
    order = np.argsort(-y_score_arr)
    baseline = float(y_true_arr.mean()) if len(y_true_arr) else 0.0
    for pct in [0.05, 0.10]:
        n_top = max(1, int(np.ceil(len(y_true_arr) * pct)))
        top_labels = y_true_arr[order[:n_top]]
        rate = float(top_labels.mean()) if n_top else 0.0
        label = f"top_{int(pct * 100)}pct"
        rows[f"{label}_n"] = int(n_top)
        rows[f"{label}_positive_cases"] = int(top_labels.sum())
        rows[f"{label}_positive_rate"] = rate
        rows[f"{label}_lift"] = rate / baseline if baseline > 0 else None
    return rows


# ==================== 模型评估函数 ====================

def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    评估预测结果
    
    计算多个分类指标:ROC-AUC、平均精度、Brier分数、准确率、精确率、
    召回率、F1、特异度、MCC,以及混淆矩阵和高风险群体指标。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 预测概率数组或Series
        threshold: 分类阈值,默认为0.5
        
    Returns:
        包含所有评估指标的字典
    """
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    y_pred = (y_score_arr >= threshold).astype(int)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "n_rows": int(len(y_true_arr)),
        "positive_cases": int(y_true_arr.sum()),
        "positive_rate": float(y_true_arr.mean()),
        "threshold": float(threshold),
        "roc_auc": _safe_roc_auc(y_true_arr, y_score_arr),
        "average_precision": _safe_average_precision(y_true_arr, y_score_arr),
        "brier_score": _safe_brier_score(y_true_arr, y_score_arr),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    # 添加高风险群体指标
    metrics.update(_top_risk_metrics(y_true_arr, y_score_arr))
    return metrics


def build_roc_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    构建ROC曲线数据表
    
    计算不同阈值下的FPR、TPR、特异度和Youden指数。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 预测概率数组或Series
        
    Returns:
        包含ROC曲线数据的DataFrame,列包括threshold、fpr、tpr、specificity、youden_index
    """
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))
    if np.unique(y_true_arr).size < 2:
        return pd.DataFrame(
            columns=["threshold", "fpr", "tpr", "specificity", "youden_index"]
        )

    fpr, tpr, thresholds = roc_curve(y_true_arr, y_score_arr)
    roc_df = pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr})
    roc_df = roc_df[np.isfinite(roc_df["threshold"])].copy()
    roc_df["specificity"] = 1.0 - roc_df["fpr"]
    roc_df["youden_index"] = roc_df["tpr"] - roc_df["fpr"]
    return roc_df.reset_index(drop=True)


def select_threshold_by_youden(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    使用Youden指数选择最优分类阈值
    
    Youden指数 = 灵敏度 + 特异度 - 1,最大化该值可获得最佳平衡点。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 预测概率数组或Series
        
    Returns:
        包含最优阈值及相关指标的字典
    """
    roc_df = build_roc_table(y_true, y_score)
    if roc_df.empty:
        return {
            "threshold": 0.5,
            "youden_index": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "fpr": 0.0,
            "tpr": 0.0,
        }

    best_idx = int(roc_df["youden_index"].idxmax())
    best_row = roc_df.loc[best_idx]
    return {
        "threshold": float(best_row["threshold"]),
        "youden_index": float(best_row["youden_index"]),
        "sensitivity": float(best_row["tpr"]),
        "specificity": float(best_row["specificity"]),
        "fpr": float(best_row["fpr"]),
        "tpr": float(best_row["tpr"]),
    }


def fit_platt_calibrator(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    random_state: int,
) -> LogisticRegression:
    """
    拟合Platt校准器(逻辑回归校准)
    
    使用逻辑回归对原始预测概率进行校准,使预测概率更接近真实概率。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 原始预测概率数组或Series
        random_state: 随机种子
        
    Returns:
        拟合好的LogisticRegression校准器对象
        
    Raises:
        ValueError: 验证标签必须包含两类
    """
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    if np.unique(y_true_arr).size < 2:
        raise ValueError("Validation labels must contain both classes for Platt scaling.")

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    calibrator.fit(np.asarray(y_score, dtype=float).reshape(-1, 1), y_true_arr)
    return calibrator


def apply_platt_calibrator(
    calibrator: LogisticRegression,
    y_score: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    应用Platt校准器
    
    使用已拟合的校准器对预测概率进行校准。
    
    Args:
        calibrator: 已拟合的LogisticRegression校准器
        y_score: 原始预测概率数组或Series
        
    Returns:
        校准后的预测概率数组
    """
    return calibrator.predict_proba(np.asarray(y_score, dtype=float).reshape(-1, 1))[
        :, 1
    ]


def build_calibration_table(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    构建校准曲线数据表
    
    将预测概率分桶,计算每个桶内的平均预测概率和实际观测率,
    用于评估模型校准程度。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 预测概率数组或Series
        n_bins: 分桶数量,默认为10
        
    Returns:
        包含校准曲线数据的DataFrame,列包括bin、n_rows、mean_predicted_probability、observed_rate
    """
    frame = pd.DataFrame(
        {
            "target": np.asarray(pd.Series(y_true).astype(int)),
            "score": np.asarray(pd.Series(y_score).astype(float)),
        }
    )

    unique_scores = int(frame["score"].nunique())
    if unique_scores <= 1:
        return pd.DataFrame(
            [
                {
                    "bin": 1,
                    "n_rows": int(len(frame)),
                    "mean_predicted_probability": float(frame["score"].mean()),
                    "observed_rate": float(frame["target"].mean()),
                }
            ]
        )

    bin_count = min(n_bins, unique_scores)
    frame["bin_interval"] = pd.qcut(frame["score"], q=bin_count, duplicates="drop")
    calibration_df = (
        frame.groupby("bin_interval", observed=True)
        .agg(
            n_rows=("target", "size"),
            mean_predicted_probability=("score", "mean"),
            observed_rate=("target", "mean"),
        )
        .reset_index(drop=True)
    )
    calibration_df.insert(0, "bin", np.arange(1, len(calibration_df) + 1))
    return calibration_df


def bootstrap_metric_intervals(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    使用Bootstrap方法计算指标的置信区间
    
    通过对正负样本分别重采样,计算评估指标的95%置信区间。
    
    Args:
        y_true: 真实标签数组或Series
        y_score: 预测概率数组或Series
        threshold: 分类阈值
        n_bootstrap: Bootstrap迭代次数,默认为1000
        random_state: 随机种子
        metrics: 要计算的指标列表,默认为EVALUATION_METRICS
        
    Returns:
        包含指标点估计和置信区间的DataFrame,列包括metric、point_estimate、ci_low、ci_high
        
    Raises:
        ValueError: Bootstrap需要同时存在正负样本
    """
    y_true_arr = np.asarray(pd.Series(y_true).astype(int))
    y_score_arr = np.asarray(pd.Series(y_score).astype(float))

    pos_idx = np.flatnonzero(y_true_arr == 1)
    neg_idx = np.flatnonzero(y_true_arr == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Bootstrap requires both positive and negative cases.")

    metric_names = metrics or EVALUATION_METRICS
    point_estimates = evaluate_predictions(y_true_arr, y_score_arr, threshold=threshold)
    rng = np.random.default_rng(random_state)

    samples_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for _ in range(n_bootstrap):
        # 分别对正负样本进行有放回抽样
        sampled_pos_idx = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg_idx = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        sampled_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
        rng.shuffle(sampled_idx)

        sampled_metrics = evaluate_predictions(
            y_true_arr[sampled_idx],
            y_score_arr[sampled_idx],
            threshold=threshold,
        )
        for metric in metric_names:
            value = sampled_metrics.get(metric)
            if value is not None:
                samples_by_metric[metric].append(float(value))

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        point_estimate = point_estimates.get(metric)
        metric_samples = np.asarray(samples_by_metric[metric], dtype=float)
        if point_estimate is None or metric_samples.size == 0:
            rows.append(
                {
                    "metric": metric,
                    "point_estimate": point_estimate,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
            continue
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point_estimate),
                "ci_low": float(np.quantile(metric_samples, 0.025)),
                "ci_high": float(np.quantile(metric_samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def _count_search_candidates(search_spec: SearchSpec, search_mode: str) -> int | None:
    """
    计算搜索候选数量
    
    Args:
        search_spec: 搜索规格对象
        search_mode: 搜索模式
        
    Returns:
        候选数量,如果search_mode为"none"则返回None
    """
    if search_mode == "none":
        return None
    param_space = search_spec.param_space_by_mode[search_mode]
    if search_spec.strategy == "grid":
        return len(ParameterGrid(param_space))
    if search_spec.n_iter_by_mode is None:
        return None
    return int(search_spec.n_iter_by_mode[search_mode])


def _normalize_cv_results(cv_results: dict[str, Any]) -> pd.DataFrame:
    """
    标准化交叉验证结果
    
    将cv_results转换为DataFrame并按平均精度排名排序。
    
    Args:
        cv_results: GridSearchCV或RandomizedSearchCV的cv_results_属性
        
    Returns:
        排序后的DataFrame
    """
    return pd.DataFrame(cv_results).sort_values(
        by="rank_test_average_precision", na_position="last"
    )


def _fit_search(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    target_col: str,
    search_spec: SearchSpec,
    search_mode: str,
    cv_folds: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame | None]:
    """
    执行超参数搜索并拟合模型
    
    根据搜索模式(none/fast/full)和策略(grid/random)进行超参数调优,
    返回最佳模型、搜索摘要和搜索结果。
    
    Args:
        pipeline: 基础Pipeline对象
        train_df: 训练集DataFrame
        target_col: 目标变量列名
        search_spec: 搜索规格对象
        search_mode: 搜索模式
        cv_folds: 交叉验证折数
        random_state: 随机种子
        
    Returns:
        元组:(最佳Pipeline, 搜索摘要字典, 搜索结果DataFrame或None)
    """
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df[target_col].astype(int)

    if search_mode == "none":
        log_step("Search mode is none, fitting base model directly")
        pipeline.fit(X_train, y_train)
        return (
            pipeline,
            {
                "search_mode": "none",
                "search_strategy": "none",
                "refit_metric": REFIT_METRIC,
                "candidate_count": None,
                "best_score": None,
                "best_params": {},
            },
            None,
        )

    # 确定实际使用的交叉验证折数
    effective_folds = _determine_cv_folds(y_train, cv_folds)
    cv = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=random_state
    )
    param_space = search_spec.param_space_by_mode[search_mode]
    candidate_count = _count_search_candidates(search_spec, search_mode)
    log_step(
        "Starting hyperparameter search: "
        f"mode={search_mode}, strategy={search_spec.strategy}, "
        f"cv_folds={effective_folds}, candidates={candidate_count}"
    )

    # 根据策略选择网格搜索或随机搜索
    if search_spec.strategy == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_space,
            scoring=SEARCH_SCORING,
            refit=REFIT_METRIC,
            cv=cv,
            n_jobs=SEARCH_N_JOBS,
            return_train_score=False,
            error_score="raise",
            verbose=2,
        )
    else:
        if search_spec.n_iter_by_mode is None:
            raise ValueError("Random search requires n_iter_by_mode.")
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=search_spec.n_iter_by_mode[search_mode],
            scoring=SEARCH_SCORING,
            refit=REFIT_METRIC,
            cv=cv,
            n_jobs=SEARCH_N_JOBS,
            return_train_score=False,
            error_score="raise",
            random_state=random_state,
            verbose=2,
        )

    # 使用指定的并行后端执行搜索
    if SEARCH_BACKEND:
        with parallel_backend(SEARCH_BACKEND):
            search.fit(X_train, y_train)
    else:
        search.fit(X_train, y_train)
    log_step(
        f"Search finished, best {REFIT_METRIC}={search.best_score_:.6f}"
    )
    search_results_df = _normalize_cv_results(search.cv_results_)
    search_summary = {
        "search_mode": search_mode,
        "search_strategy": search_spec.strategy,
        "refit_metric": REFIT_METRIC,
        "cv_folds_used": effective_folds,
        "candidate_count": candidate_count,
        "best_score": float(search.best_score_),
        "best_params": {key: _json_safe(value) for key, value in search.best_params_.items()},
    }
    return search.best_estimator_, search_summary, search_results_df


def run_cross_validation_pipeline(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    target_col: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """
    运行交叉验证Pipeline
    
    在训练集上执行k折交叉验证,记录每折的评估指标。
    
    Args:
        pipeline: 已拟合的Pipeline对象
        train_df: 训练集DataFrame
        target_col: 目标变量列名
        n_splits: 交叉验证折数
        random_state: 随机种子
        
    Returns:
        包含每折评估指标的DataFrame
    """
    y = train_df[target_col].astype(int)
    effective_folds = _determine_cv_folds(y, n_splits)
    splitter = StratifiedKFold(
        n_splits=effective_folds, shuffle=True, random_state=random_state
    )

    rows: list[dict[str, Any]] = []
    log_step(f"Running post-search cross-validation summary with {effective_folds} folds")
    for fold_idx, (train_idx, valid_idx) in enumerate(
        splitter.split(train_df[MODEL_FEATURES], y), start=1
    ):
        fold_train = train_df.iloc[train_idx].copy()
        fold_valid = train_df.iloc[valid_idx].copy()

        # 克隆Pipeline并在当前折上重新训练
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(
            fold_train[MODEL_FEATURES], fold_train[target_col].astype(int)
        )
        fold_scores = fold_pipeline.predict_proba(fold_valid[MODEL_FEATURES])[:, 1]

        metrics = evaluate_predictions(
            fold_valid[target_col], fold_scores, threshold=0.5
        )
        metrics.update(
            {
                "fold": fold_idx,
                "train_rows": int(len(fold_train)),
                "valid_rows": int(len(fold_valid)),
                "train_positive_rate": float(fold_train[target_col].astype(int).mean()),
                "valid_positive_rate": float(fold_valid[target_col].astype(int).mean()),
            }
        )
        log_step(
            f"Cross-validation fold {fold_idx}/{effective_folds} done: "
            f"ap={metrics['average_precision']}, roc_auc={metrics['roc_auc']}"
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_cv_metrics(cv_df: pd.DataFrame) -> dict[str, Any]:
    """
    汇总交叉验证指标
    
    计算各指标在交叉验证中的均值和标准差。
    
    Args:
        cv_df: 交叉验证结果DataFrame
        
    Returns:
        包含汇总统计的字典
    """
    summary: dict[str, Any] = {
        "n_folds": int(len(cv_df)),
        "train_rows_mean": float(cv_df["train_rows"].mean()),
        "valid_rows_mean": float(cv_df["valid_rows"].mean()),
    }
    for metric in EVALUATION_METRICS:
        metric_series = cv_df[metric].dropna()
        if metric_series.empty:
            summary[metric] = {"mean": None, "std": None}
            continue
        summary[metric] = {
            "mean": float(metric_series.mean()),
            "std": float(metric_series.std(ddof=1)) if len(metric_series) > 1 else 0.0,
        }
    return summary


def make_run_dir(
    model_name: str,
    target_col: str,
    period_token: str,
    cohort: str,
    feature_version: str,
    feature_set: str,
) -> Path:
    """
    创建运行输出目录
    
    根据模型名称、目标变量、时期标记等参数生成唯一的输出目录路径。
    
    Args:
        model_name: 模型名称
        target_col: 目标变量列名
        period_token: 时期标记
        cohort: 队列类型
        feature_version: 特征版本
        feature_set: 特征集合
        
    Returns:
        输出目录的Path对象
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_suffix = "" if feature_version == "v1" else f"_{feature_version}"
    feature_set_suffix = "" if feature_version == "v1" else f"_{feature_set}"
    run_dir = (
        OUTPUT_ML_ROOT
        / model_name
        / f"{target_col}_{cohort}_{period_token}{version_suffix}{feature_set_suffix}_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_safe(value: Any) -> Any:
    """
    将值转换为JSON安全的格式
    
    处理NumPy类型、Path对象等非标准JSON类型。
    
    Args:
        value: 任意类型的值
        
    Returns:
        JSON安全的值
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        cast_value = float(value)
        return None if not np.isfinite(cast_value) else cast_value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """
    保存字典为JSON文件
    
    Args:
        path: 输出文件路径
        payload: 要保存的字典数据
    """
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_split_summary(
    splits: dict[str, pd.DataFrame],
    target_col: str,
    output_path: Path,
) -> None:
    """
    保存数据划分摘要
    
    记录训练集、验证集、测试集的行数、正例数、正例率、年份范围等信息。
    
    Args:
        splits: 包含"train"、"valid"、"test"的DataFrame字典
        target_col: 目标变量列名
        output_path: 输出CSV文件路径
    """
    rows: list[dict[str, Any]] = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "split": split_name,
                "n_rows": int(len(split_df)),
                "positive_cases": int(split_df[target_col].astype(int).sum()),
                "positive_rate": float(split_df[target_col].astype(int).mean()),
                "min_year": int(split_df["year"].min()),
                "max_year": int(split_df["year"].max()),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_prediction_table(
    df: pd.DataFrame,
    target_col: str,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """
    保存预测结果表
    
    包含caseid、年份、真实标签、原始预测概率、校准后预测概率、预测标签。
    
    Args:
        df: 包含caseid和year的DataFrame
        target_col: 目标变量列名
        raw_scores: 原始预测概率数组
        calibrated_scores: 校准后预测概率数组
        threshold: 分类阈值
        output_path: 输出CSV文件路径
    """
    pd.DataFrame(
        {
            "caseid": df["caseid"].astype(str),
            "year": df["year"].astype(int),
            "target": df[target_col].astype(int),
            "predicted_probability_raw": raw_scores,
            "predicted_probability_calibrated": calibrated_scores,
            "predicted_label_optimal": (calibrated_scores >= threshold).astype(int),
        }
    ).to_csv(output_path, index=False, encoding="utf-8-sig")


def _extract_model_params(pipeline: Pipeline) -> dict[str, Any]:
    """
    提取模型参数
    
    Args:
        pipeline: 已拟合的Pipeline对象
        
    Returns:
        模型参数字典
    """
    model = pipeline.named_steps["model"]
    params = model.get_params(deep=False)
    if isinstance(model, LogisticRegression) and params.get("penalty") == "deprecated":
        params["penalty"] = "l2 (scikit-learn default)"
    return {key: _json_safe(value) for key, value in sorted(params.items())}


def _build_search_payload(
    result: ExperimentResult,
    model_name: str,
    display_name: str,
) -> dict[str, Any]:
    """
    构建搜索摘要载荷
    
    用于保存best_params.json文件。
    
    Args:
        result: 实验结果对象
        model_name: 模型名称
        display_name: 显示名称
        
    Returns:
        包含搜索和模型参数的字典
    """
    return {
        "model": model_name,
        "display_name": display_name,
        "search_mode": result.search_summary["search_mode"],
        "search_strategy": result.search_summary["search_strategy"],
        "refit_metric": result.search_summary["refit_metric"],
        "cv_folds_used": result.search_summary.get("cv_folds_used"),
        "candidate_count": result.search_summary.get("candidate_count"),
        "best_score": result.search_summary.get("best_score"),
        "best_params": result.search_summary.get("best_params", {}),
        "selected_model_params": _extract_model_params(result.pipeline),
    }


def _build_metrics_payload(
    result: ExperimentResult,
    model_name: str,
    display_name: str,
) -> dict[str, Any]:
    """
    构建指标摘要载荷
    
    用于保存metrics.json文件,包含完整的实验配置和评估结果。
    
    Args:
        result: 实验结果对象
        model_name: 模型名称
        display_name: 显示名称
        
    Returns:
        包含完整实验信息的字典
    """
    return {
        "model": model_name,
        "display_name": display_name,
        "feature_version": result.config.feature_version,
        "feature_set": result.config.feature_set,
        "target_col": result.config.target_col,
        "cohort": result.config.cohort,
        "period_token": result.bundle.period_token,
        "signal_file": str(result.bundle.signal_file),
        "feature_file": str(result.bundle.feature_file),
        "train_end_year": result.config.train_end_year,
        "valid_year": result.config.valid_year,
        "test_year": result.config.test_year,
        "train_sample_n": result.config.train_sample_n,
        "search_mode": result.config.search_mode,
        "cv_folds_requested": result.config.cv_folds,
        "bootstrap_iterations": result.config.bootstrap_iterations,
        "model_features": MODEL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "bool_features": BOOL_FEATURES,
        "search_summary": result.search_summary,
        "cross_validation_summary": result.cv_summary,
        "threshold_selection": result.threshold_selection,
        "validation_metrics": result.validation_metrics,
        "test_metrics": result.test_metrics,
        "validation_metrics_raw_threshold_0_5": result.validation_metrics_raw,
        "test_metrics_raw_threshold_0_5": result.test_metrics_raw,
        "calibration_method": "platt",
    }


def run_model_experiment(
    *,
    config: ExperimentConfig,
    model_name: str,
    display_name: str,
    estimator_factory: Callable[[pd.DataFrame, ExperimentConfig], BaseEstimator],
    search_spec: SearchSpec,
) -> ExperimentResult:
    """
    运行完整的机器学习实验
    
    这是整个ML流程的核心函数,执行以下步骤:
    1. 解析数据集包
    2. 加载建模数据框
    3. 时间序列划分(训练/验证/测试)
    4. 训练集采样(可选)
    5. 超参数搜索和模型拟合
    6. 交叉验证评估
    7. 生成预测概率
    8. Platt校准
    9. 阈值选择和指标评估
    10. 构建ROC曲线、校准曲线、Bootstrap置信区间
    11. 保存所有输出文件
    
    Args:
        config: 实验配置对象
        model_name: 模型名称(用于目录命名)
        display_name: 显示名称(用于报告)
        estimator_factory: 估计器工厂函数,接收(train_df, config)返回BaseEstimator
        search_spec: 超参数搜索规格
        
    Returns:
        ExperimentResult实验结果对象,包含所有实验数据和结果
    """
    experiment_start = perf_counter()
    log_step(
        "Experiment config: "
        f"model={display_name}, target={config.target_col}, "
        f"feature_version={config.feature_version}, feature_set={config.feature_set}, "
        f"cohort={config.cohort}, "
        f"train<= {config.train_end_year}, valid={config.valid_year}, "
        f"test={config.test_year}, search={config.search_mode}, "
        f"train_sample_n={config.train_sample_n}"
    )
    
    # 步骤1: 解析数据集包
    with timed_step("Resolve dataset bundle"):
        bundle = resolve_dataset_bundle(
            period_token=config.period_token,
            feature_version=config.feature_version,
        )
    
    # 步骤2: 加载建模数据框
    with timed_step("Load modeling frame"):
        modeling_df = load_modeling_frame(
            bundle=bundle,
            target_col=config.target_col,
            cohort=config.cohort,
            feature_set=config.feature_set,
        )
    log_frame_summary("Modeling frame", modeling_df, config.target_col)
    
    # 步骤3: 时间序列划分
    with timed_step("Temporal split"):
        splits = temporal_split(
            modeling_df,
            train_end_year=config.train_end_year,
            valid_year=config.valid_year,
            test_year=config.test_year,
        )

    train_full_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]
    log_frame_summary("Train full", train_full_df, config.target_col)
    log_frame_summary("Validation", valid_df, config.target_col)
    log_frame_summary("Test", test_df, config.target_col)

    # 步骤4: 训练集采样(可选)
    with timed_step("Sample training frame"):
        train_df = sample_training_frame(
            train_full_df,
            target_col=config.target_col,
            sample_n=config.train_sample_n,
            random_state=config.random_state,
        )
    log_frame_summary("Train used", train_df, config.target_col)

    # 构建Pipeline
    pipeline = build_pipeline(estimator_factory(train_df, config))
    
    # 步骤5: 超参数搜索和模型拟合
    with timed_step("Fit model and tune hyperparameters"):
        fitted_pipeline, search_summary, search_results_df = _fit_search(
            pipeline=pipeline,
            train_df=train_df,
            target_col=config.target_col,
            search_spec=search_spec,
            search_mode=config.search_mode,
            cv_folds=config.cv_folds,
            random_state=config.random_state,
        )

    # 步骤6: 交叉验证评估
    with timed_step("Run post-search cross-validation"):
        cv_metrics_df = run_cross_validation_pipeline(
            pipeline=fitted_pipeline,
            train_df=train_df,
            target_col=config.target_col,
            n_splits=config.cv_folds,
            random_state=config.random_state,
        )
    cv_summary = summarize_cv_metrics(cv_metrics_df)

    # 步骤7: 生成验证集和测试集的预测概率
    with timed_step("Generate validation and test probabilities"):
        valid_raw_scores = fitted_pipeline.predict_proba(valid_df[MODEL_FEATURES])[:, 1]
        test_raw_scores = fitted_pipeline.predict_proba(test_df[MODEL_FEATURES])[:, 1]

    # 步骤8: Platt校准
    with timed_step("Calibrate probabilities with Platt scaling"):
        calibrator = fit_platt_calibrator(
            valid_df[config.target_col], valid_raw_scores, config.random_state
        )
        valid_scores = apply_platt_calibrator(calibrator, valid_raw_scores)
        test_scores = apply_platt_calibrator(calibrator, test_raw_scores)

    # 步骤9: 阈值选择和指标评估
    with timed_step("Select threshold and evaluate metrics"):
        threshold_selection = select_threshold_by_youden(
            valid_df[config.target_col], valid_scores
        )
        threshold = threshold_selection["threshold"]

        validation_metrics = evaluate_predictions(
            valid_df[config.target_col], valid_scores, threshold=threshold
        )
        test_metrics = evaluate_predictions(
            test_df[config.target_col], test_scores, threshold=threshold
        )
        validation_metrics_raw = evaluate_predictions(
            valid_df[config.target_col], valid_raw_scores, threshold=0.5
        )
        test_metrics_raw = evaluate_predictions(
            test_df[config.target_col], test_raw_scores, threshold=0.5
        )

    # 创建输出目录
    run_dir = make_run_dir(
        model_name=model_name,
        target_col=config.target_col,
        period_token=bundle.period_token,
        cohort=config.cohort,
        feature_version=bundle.feature_version,
        feature_set=config.feature_set,
    )
    log_step(f"Writing outputs to {run_dir}")

    # 步骤10: 构建ROC曲线、校准曲线、Bootstrap置信区间
    with timed_step("Build ROC, calibration, and bootstrap tables"):
        valid_roc_df = build_roc_table(valid_df[config.target_col], valid_scores)
        test_roc_df = build_roc_table(test_df[config.target_col], test_scores)
        valid_calibration_df = build_calibration_table(
            valid_df[config.target_col], valid_scores
        )
        test_calibration_df = build_calibration_table(
            test_df[config.target_col], test_scores
        )
        bootstrap_df = bootstrap_metric_intervals(
            test_df[config.target_col],
            test_scores,
            threshold=threshold,
            n_bootstrap=config.bootstrap_iterations,
            random_state=config.random_state,
        )

    # 构建实验结果对象
    result = ExperimentResult(
        config=config,
        bundle=bundle,
        run_dir=run_dir,
        train_full_df=train_full_df,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        pipeline=fitted_pipeline,
        search_summary=search_summary,
        search_results_df=search_results_df,
        cv_metrics_df=cv_metrics_df,
        cv_summary=cv_summary,
        threshold_selection=threshold_selection,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_metrics_raw=validation_metrics_raw,
        test_metrics_raw=test_metrics_raw,
        valid_raw_scores=valid_raw_scores,
        valid_scores=valid_scores,
        test_raw_scores=test_raw_scores,
        test_scores=test_scores,
    )

    # 步骤11: 保存所有输出文件
    with timed_step("Save core output files"):
        cv_metrics_df.to_csv(
            run_dir / "cv_metrics.csv", index=False, encoding="utf-8-sig"
        )
        if search_results_df is not None:
            search_results_df.to_csv(
                run_dir / "search_results.csv", index=False, encoding="utf-8-sig"
            )
        save_json(
            run_dir / "best_params.json",
            _build_search_payload(result, model_name=model_name, display_name=display_name),
        )
        save_json(
            run_dir / "metrics.json",
            _build_metrics_payload(
                result, model_name=model_name, display_name=display_name
            ),
        )
        save_split_summary(
            {
                "train_full": train_full_df,
                "train_sampled": train_df,
                "valid": valid_df,
                "test": test_df,
            },
            target_col=config.target_col,
            output_path=run_dir / "split_summary.csv",
        )
        save_prediction_table(
            valid_df,
            config.target_col,
            valid_raw_scores,
            valid_scores,
            threshold,
            run_dir / "validation_predictions.csv",
        )
        save_prediction_table(
            test_df,
            config.target_col,
            test_raw_scores,
            test_scores,
            threshold,
            run_dir / "test_predictions.csv",
        )
        valid_roc_df.to_csv(
            run_dir / "validation_roc_curve.csv", index=False, encoding="utf-8-sig"
        )
        test_roc_df.to_csv(
            run_dir / "test_roc_curve.csv", index=False, encoding="utf-8-sig"
        )
        valid_calibration_df.to_csv(
            run_dir / "validation_calibration_curve.csv",
            index=False,
            encoding="utf-8-sig",
        )
        test_calibration_df.to_csv(
            run_dir / "test_calibration_curve.csv", index=False, encoding="utf-8-sig"
        )
        bootstrap_df.to_csv(
            run_dir / "test_bootstrap_metrics.csv", index=False, encoding="utf-8-sig"
        )

    log_step(f"Core outputs saved; experiment runtime={format_duration(perf_counter() - experiment_start)}")
    return result


def summarize_importance_highlights(
    feature_df: pd.DataFrame,
    *,
    feature_col: str,
    score_col: str,
    top_n: int = 10,
) -> list[str]:
    """
    汇总特征重要性亮点
    
    提取最重要的N个特征及其重要性分数。
    
    Args:
        feature_df: 包含特征重要性的DataFrame
        feature_col: 特征名称列名
        score_col: 重要性分数列名
        top_n: 返回前N个特征
        
    Returns:
        特征重要性亮点列表,格式为"特征名: 分数"
    """
    top_df = feature_df.sort_values(score_col, ascending=False).head(top_n)
    highlights = []
    for _, row in top_df.iterrows():
        highlights.append(f"{row[feature_col]}: {row[score_col]:.4f}")
    return highlights


def summarize_logistic_highlights(coefficients_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """
    汇总逻辑回归系数亮点
    
    提取正负关联最强的N个特征及其系数和优势比。
    
    Args:
        coefficients_df: 包含系数的DataFrame,需有"feature"、"coefficient"、"odds_ratio"列
        top_n: 每侧返回前N个特征
        
    Returns:
        系数亮点列表,包含正负关联的特征
    """
    positive_df = coefficients_df.sort_values("coefficient", ascending=False).head(top_n)
    negative_df = coefficients_df.sort_values("coefficient", ascending=True).head(top_n)
    highlights: list[str] = []
    for _, row in positive_df.iterrows():
        highlights.append(
            f"Positive association: {row['feature']} coefficient={row['coefficient']:.4f}, odds_ratio={row['odds_ratio']:.4f}"
        )
    for _, row in negative_df.iterrows():
        highlights.append(
            f"Negative association: {row['feature']} coefficient={row['coefficient']:.4f}, odds_ratio={row['odds_ratio']:.4f}"
        )
    return highlights


def _compact_metrics(metrics: dict[str, Any]) -> str:
    """
    将指标字典压缩为简洁的字符串表示
    
    Args:
        metrics: 评估指标字典
        
    Returns:
        格式化的指标字符串
    """
    return (
        f"AP={format_metric(metrics.get('average_precision'))}, "
        f"ROC-AUC={format_metric(metrics.get('roc_auc'))}, "
        f"Brier={format_metric(metrics.get('brier_score'))}, "
        f"Recall={format_metric(metrics.get('recall'))}, "
        f"Precision={format_metric(metrics.get('precision'))}, "
        f"MCC={format_metric(metrics.get('mcc'))}"
    )


def save_interpretation_summary(
    *,
    output_path: Path,
    display_name: str,
    model_name: str,
    result: ExperimentResult,
    feature_highlights: list[str],
    notes: list[str] | None = None,
) -> None:
    """
    保存解释摘要(Markdown格式)
    
    生成包含运行快照、数据划分、阈值选择、最终指标、最佳参数、
    主要特征信号等信息的解释性文档。
    
    Args:
        output_path: 输出文件路径
        display_name: 显示名称
        model_name: 模型名称
        result: 实验结果对象
        feature_highlights: 特征亮点列表
        notes: 额外备注列表(可选)
    """
    notes = notes or []
    best_params = result.search_summary.get("best_params", {})
    best_params_lines = json.dumps(best_params, ensure_ascii=False, indent=2)
    lines = [
        f"# {display_name} interpretation summary",
        "",
        "## Run snapshot",
        f"- Model: `{model_name}`",
        f"- Target: `{result.config.target_col}`",
        f"- Feature version: `{result.config.feature_version}`",
        f"- Cohort: `{result.config.cohort}`",
        f"- Period token: `{result.bundle.period_token}`",
        f"- Output directory: `{result.run_dir}`",
        "",
        "## Data split",
        f"- Train used: `{len(result.train_df):,}` rows",
        f"- Validation: `{len(result.valid_df):,}` rows",
        f"- Test: `{len(result.test_df):,}` rows",
        "",
        "## Selected threshold",
        f"- Threshold: `{format_metric(result.threshold_selection['threshold'])}`",
        f"- Validation Youden index: `{format_metric(result.threshold_selection['youden_index'])}`",
        "",
        "## Final metrics",
        f"- Validation: {_compact_metrics(result.validation_metrics)}",
        f"- Test: {_compact_metrics(result.test_metrics)}",
        "",
        "## Best tuning parameters",
        "```json",
        best_params_lines,
        "```",
        "",
        "## Main feature signals",
    ]
    lines.extend(f"- {highlight}" for highlight in feature_highlights)
    lines.extend(
        [
            "",
            "## Plain-language caution",
            "- These are prediction signals from FAERS reporting patterns, not causal effects.",
            "- Use them as a ranking and explanation layer, then interpret with the main signal analysis.",
        ]
    )
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_run_summary(
    *,
    display_name: str,
    result: ExperimentResult,
    feature_highlights: list[str],
    top_n: int = 6,
) -> None:
    """
    打印运行摘要到控制台
    
    在终端输出模型配置、评估指标、阈值、主要特征信号等关键信息。
    
    Args:
        display_name: 显示名称
        result: 实验结果对象
        feature_highlights: 特征亮点列表
        top_n: 显示前N个特征亮点
    """
    print("", flush=True)
    print("=== ML run summary ===", flush=True)
    print(
        f"Model: {display_name} | target={result.config.target_col} | "
        f"features={result.config.feature_version} | cohort={result.config.cohort}",
        flush=True,
    )
    print(f"Output: {result.run_dir}", flush=True)
    print(f"Train used: {len(result.train_df):,} rows", flush=True)
    print(f"Validation: {_compact_metrics(result.validation_metrics)}", flush=True)
    print(f"Test: {_compact_metrics(result.test_metrics)}", flush=True)
    print(
        "Threshold: "
        f"{format_metric(result.threshold_selection['threshold'])} "
        f"(validation Youden={format_metric(result.threshold_selection['youden_index'])})",
        flush=True,
    )
    print("Main feature signals:", flush=True)
    for highlight in feature_highlights[:top_n]:
        print(f"- {highlight}", flush=True)
    print("Files: metrics.json, model_card.md, interpretation_summary.md", flush=True)
    print("======================", flush=True)


def save_model_card(
    *,
    output_path: Path,
    display_name: str,
    model_name: str,
    result: ExperimentResult,
    feature_highlights: list[str],
    notes: list[str] | None = None,
) -> None:
    """
    保存模型卡片(Markdown格式)
    
    生成包含任务描述、数据信息、时间划分、搜索配置、选中参数、
    最终指标、特征亮点、局限性说明等的完整模型文档。
    
    Args:
        output_path: 输出文件路径
        display_name: 显示名称
        model_name: 模型名称
        result: 实验结果对象
        feature_highlights: 特征亮点列表
        notes: 额外备注列表(可选)
    """
    notes = notes or []
    best_params_payload = _build_search_payload(
        result, model_name=model_name, display_name=display_name
    )
    best_params_lines = json.dumps(
        best_params_payload["selected_model_params"], ensure_ascii=False, indent=2
    )

    lines = [
        f"# {display_name} model card",
        "",
        "## Task",
        f"- Predict `{result.config.target_col}` from the FAERS global case-level bundle.",
        "- Use the model as a research ranking layer on top of the existing signal detection workflow.",
        "",
        "## Data",
        f"- Signal file: `{result.bundle.signal_file}`",
        f"- Feature file: `{result.bundle.feature_file}`",
        f"- Period token: `{result.bundle.period_token}`",
        f"- Cohort: `{result.config.cohort}`",
        "",
        "## Time split",
        f"- Train: years <= {result.config.train_end_year}",
        f"- Validation: {result.config.valid_year}",
        f"- Test: {result.config.test_year}",
        "",
        "## Search",
        f"- Search mode: `{result.search_summary['search_mode']}`",
        f"- Search strategy: `{result.search_summary['search_strategy']}`",
        f"- Refit metric: `{result.search_summary['refit_metric']}`",
        f"- Candidate count: `{result.search_summary.get('candidate_count')}`",
        "",
        "## Selected parameters",
        "```json",
        best_params_lines,
        "```",
        "",
        "## Final metrics",
        f"- Validation average precision: `{result.validation_metrics['average_precision']}`",
        f"- Validation ROC-AUC: `{result.validation_metrics['roc_auc']}`",
        f"- Test average precision: `{result.test_metrics['average_precision']}`",
        f"- Test ROC-AUC: `{result.test_metrics['roc_auc']}`",
        f"- Test Brier score: `{result.test_metrics['brier_score']}`",
        "",
        "## Feature highlights",
    ]
    lines.extend(f"- {highlight}" for highlight in feature_highlights)
    lines.extend(
        [
            "",
            "## Limitations",
            "- This model is an auxiliary ranking tool and does not replace ROR/PRR or subgroup analysis.",
            "- The output reflects reporting patterns in FAERS, not causal drug effects.",
            "- The current feature set is low-dimensional structured data, so deep learning is intentionally not used here.",
        ]
    )
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

