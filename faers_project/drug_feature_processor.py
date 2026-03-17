import re
import pandas as pd
from pathlib import Path

from drug_processor import process_drug


# 病例级药物特征匹配词表。
# 统一使用大写，后续在标准化后的 drugname/prod_ai 中做包含匹配。
DRUG_FEATURE_TERMS = {
    "is_zolpidem": [
        "ZOLPIDEM",
        "AMBIEN",
        "STILNOX",
        "EDLUAR",
        "INTERMEZZO",
        "ZOLPIMIST",
    ],
    "is_zaleplon": ["ZALEPLON", "SONATA"],
    "is_zopiclone": ["ZOPICLONE", "IMOVANE", "ZIMOVANE"],
    "is_eszopiclone": ["ESZOPICLONE", "LUNESTA"],
    "is_benzo": [
        "ALPRAZOLAM",
        "DIAZEPAM",
        "LORAZEPAM",
        "CLONAZEPAM",
        "TEMAZEPAM",
    ],
    "is_antidepressant": [
        "SERTRALINE",
        "ESCITALOPRAM",
        "FLUOXETINE",
        "CITALOPRAM",
        "PAROXETINE",
        "VENLAFAXINE",
        "DULOXETINE",
        "AMITRIPTYLINE",
        "MIRTAZAPINE",
    ],
    "is_antipsychotic": [
        "QUETIAPINE",
        "OLANZAPINE",
        "RISPERIDONE",
        "ARIPIPRAZOLE",
        "HALOPERIDOL",
    ],
    "is_opioid": [
        "OXYCODONE",
        "HYDROCODONE",
        "MORPHINE",
        "FENTANYL",
        "TRAMADOL",
        "CODEINE",
    ],
    "is_antiepileptic": [
        "GABAPENTIN",
        "PREGABALIN",
        "VALPROATE",
        "CARBAMAZEPINE",
        "LAMOTRIGINE",
    ],
}


def _normalize_drug_text(series):
    """
    统一标准化药物文本，便于后续做包含匹配和去重计数。

    处理规则:
        1. 空值转为空字符串
        2. 去除首尾空格
        3. 转为大写
        4. 将连续空白压缩为单个空格
    """
    return (
        series.where(series.notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def _build_contains_pattern(terms):
    """
    根据匹配词列表构建正则模式。

    这里使用 re.escape 对词项转义，避免商品名中若存在特殊字符时影响匹配。
    """
    escaped_terms = [re.escape(term) for term in terms]
    return "|".join(escaped_terms)


def process_drug_feature(year, quarter, output_root):
    """
    处理 FAERS DRUG 数据，并生成病例级药物特征表。

    函数功能:
        基于清洗后的 DRUG parquet，构建以 caseid 为粒度的病例级药物特征。
        每个病例仅保留一条记录，用于后续病例级信号挖掘和协变量分析。

    病例级字段说明:
        - is_zolpidem: 是否命中唑吡坦相关药物
        - is_zaleplon: 是否命中扎来普隆相关药物
        - is_zopiclone: 是否命中佐匹克隆相关药物
        - is_eszopiclone: 是否命中右佐匹克隆相关药物
        - is_benzo: 是否命中选定的苯二氮卓类药物
        - is_antidepressant: 是否命中选定的抗抑郁药
        - is_antipsychotic: 是否命中选定的抗精神病药
        - is_opioid: 是否命中选定的阿片类药物
        - is_antiepileptic: 是否命中选定的抗癫痫药物
        - drug_n: 病例内去重后的药物标识数
        - polypharmacy: 是否满足多药并用（drug_n >= 5）

    处理规则:
        1. 若缺少前置的 DRUG parquet，则自动调用 process_drug 生成
        2. 同时使用 drugname 和 prod_ai 做标准化后的包含匹配
        3. 不按 role_cod 过滤，保留病例中所有报告药物
        4. drug_n 的去重标识优先使用 prod_ai，若缺失则回退到 drugname
        5. 使用 caseid 聚合，布尔特征只要任一药物记录命中即记为 True

    参数:
        year (int): 年份，例如 2024
        quarter (str): 季度，例如 'Q1', 'Q2', 'Q3', 'Q4'
        output_root (str or Path): 输出文件根目录

    返回:
        pd.DataFrame: 病例级药物特征表
    """
    # ========== 步骤 1: 准备输入输出路径 ==========
    output_root = Path(output_root)
    drug_file = output_root / f"drug_{year}{quarter.lower()}.parquet"

    # 若缺少清洗后的 DRUG parquet，则自动补跑前置步骤。
    if not drug_file.exists():
        print(f"未找到 DRUG 结果文件，正在自动生成：{drug_file}")
        process_drug(year, quarter, output_root)

    if not drug_file.exists():
        raise FileNotFoundError(f"找不到 DRUG 文件：{drug_file}")

    # ========== 步骤 2: 读取数据并验证字段 ==========
    df = pd.read_parquet(drug_file)

    required_cols = ["caseid", "drugname", "prod_ai"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DRUG 结果缺少必要字段：{missing_cols}")

    # ========== 步骤 3: 数据清洗与标准化 ==========
    df["caseid"] = df["caseid"].where(df["caseid"].notna(), "").astype(str).str.strip()
    df["drugname"] = _normalize_drug_text(df["drugname"])
    df["prod_ai"] = _normalize_drug_text(df["prod_ai"])

    # 过滤掉无法关联病例或没有药物标识信息的记录。
    df = df[df["caseid"] != ""]
    df = df[~((df["drugname"] == "") & (df["prod_ai"] == ""))]

    # match_text 同时拼接商品名和活性成分，用于病例级药物类别识别。
    df["match_text"] = (df["drugname"] + " " + df["prod_ai"]).str.strip()

    # resolved_drug_name 用于 drug_n 计数：优先 prod_ai，缺失时回退到 drugname。
    df["resolved_drug_name"] = df["prod_ai"].where(df["prod_ai"] != "", df["drugname"])

    # ========== 步骤 4: 构建药物类别标识 ==========
    for feature_name, terms in DRUG_FEATURE_TERMS.items():
        pattern = _build_contains_pattern(terms)
        df[feature_name] = df["match_text"].str.contains(pattern, regex=True, na=False)

    # ========== 步骤 5: 计算病例级 drug_n 和 polypharmacy ==========
    # 先按病例 + 药物标识去重，再统计病例中不同药物的数量。
    drug_count_df = (
        df[df["resolved_drug_name"] != ""][["caseid", "resolved_drug_name"]]
        .drop_duplicates()
        .groupby("caseid", as_index=False)
        .size()
        .rename(columns={"size": "drug_n"})
    )

    # ========== 步骤 6: 聚合病例级布尔特征 ==========
    feature_cols = list(DRUG_FEATURE_TERMS.keys())
    case_feature_df = (
        df[["caseid", *feature_cols]]
        .groupby("caseid", as_index=False)[feature_cols]
        .max()
    )

    case_feature_df = case_feature_df.merge(drug_count_df, on="caseid", how="left")
    case_feature_df["drug_n"] = case_feature_df["drug_n"].fillna(0).astype(int)
    case_feature_df["polypharmacy"] = case_feature_df["drug_n"] >= 5

    print("病例级药物特征表行数:", len(case_feature_df))
    print("多药并用病例数:", int(case_feature_df["polypharmacy"].sum()))

    # ========== 步骤 7: 保存病例级结果 ==========
    output_root.mkdir(parents=True, exist_ok=True)
    output_file = output_root / f"drug_feature_{year}{quarter.lower()}_case.parquet"
    case_feature_df.to_parquet(output_file, index=False)

    print(f"已保存：{output_file}")
    return case_feature_df
