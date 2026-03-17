from pathlib import Path
from config import RAW_ROOT
from utils import (
    read_faers_txt,
    build_file_path,
    deduplicate_demo_records,
    apply_demo_demographic_criteria,
)


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

    # ========== 步骤 3: 数据清洗、去重与纳排标准处理 ==========
    df = deduplicate_demo_records(df)
    print("按 DEMO 规则清洗并去重后行数:", len(df))
    print("去重后重复 caseid 数量:", df["caseid"].duplicated().sum())

    # 人口学标准化与纳排：年龄标准化、老年筛选、年龄分组、性别清洗
    # 后续若增加 OUTC 严重结局字段，可在此阶段继续扩展（统一病例口径入口）
    pre_filter_n = len(df)
    df = apply_demo_demographic_criteria(df)
    print("应用老年纳排标准后行数:", len(df))
    print("纳排剔除行数:", pre_filter_n - len(df))
    print("年龄分组分布:")
    print(df["age_group"].value_counts(dropna=False))
    print("性别清洗分布:")
    print(df["sex_clean"].value_counts(dropna=False))

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    output_file = output_root / f"demo_{year}{quarter.lower()}.parquet"
    df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")
    return df
