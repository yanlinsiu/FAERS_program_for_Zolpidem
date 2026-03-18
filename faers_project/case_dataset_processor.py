import pandas as pd
from pathlib import Path

OUTC_BOOL_COLS = [
    "is_death",
    "is_life_threatening",
    "is_hospitalization",
    "is_required_intervention",
    "is_disability",
    "is_congenital_anomaly",
    "is_other_serious",
    "is_serious_any",
]

DRUG_EXPOSURE_BOOL_COLS = [
    "suspect_role_any",
    "suspect_role_any_ps",
    "is_zolpidem_suspect",
    "is_zolpidem_suspect_ps",
    "is_other_zdrug_suspect",
    "is_other_zdrug_suspect_ps",
]


def process_case_dataset(year, quarter, output_root):
    """
    合并去重后的 DEMO 与病例级 REAC 跌倒标识，生成病例级分析主表

    函数功能:
        将 DEMO 表（人口统计学信息）与 REAC 表衍生的跌倒结局变量进行合并，
        生成以 caseid（病例 ID）为粒度的分析数据集。每个病例仅保留一条记录，
        包含其基本信息和是否发生跌倒的结局标识。

    参数:
        year (int): 年份，例如 2024
        quarter (str): 季度，例如 'Q1', 'Q2', 'Q3', 'Q4'
        output_root (str or Path): 输出文件的根目录路径

    处理步骤:
        1. 构建输入文件路径并验证文件存在性
        2. 读取 DEMO、REAC 和病例级药物特征 Parquet 文件
        3. 验证必要字段是否存在
        4. 数据清洗：处理空值、类型转换、去除无效记录
        5. 左连接合并 DEMO、REAC 和病例级药物特征数据
        6. 填充缺失的病例级衍生字段默认值
        7. 保存为 Parquet 格式并输出统计信息

    输入文件:
        - demo_{year}{quarter}.parquet: 去重后的 DEMO 数据
        - reac_{year}{quarter}_case.parquet: REAC 衍生的病例级跌倒标识数据
        - drug_feature_{year}{quarter}_case.parquet: DRUG 衍生的病例级药物特征表

    输出文件:
        - case_dataset_{year}{quarter}.parquet: 病例级分析主表

    输出数据结构:
        - caseid: 病例 ID（主键）
        - DEMO 表的所有字段（如 primaryid, fda_dt, caseversion 等）
        - is_fall: 跌倒结局标识（True/False）
        - is_zolpidem 等药物类别标识
        - drug_n: 病例内去重后的药物数
        - polypharmacy_5: 是否满足多药并用（distinct_drug_n >= 5）

    数据处理规则:
        - 使用左连接（left join）保留 DEMO 表中的所有有效病例
        - 未在 REAC 中命中跌倒术语的病例，is_fall 标记为 False
        - 未在 DRUG 中出现的病例，药物类别标记为 False，drug_n 记为 0
        - 过滤掉 caseid 为空的记录
        - is_fall 和药物类别字段统一转换为布尔类型

    异常:
        FileNotFoundError: 当 DEMO 文件或 REAC 病例级文件不存在时抛出
        ValueError: 当缺少必要字段时抛出

    示例:
        process_case_dataset(2024, 'Q1', 'D:\\processed_data')
    """
    # ========== 步骤 1: 准备输出路径 ==========
    # 将输出根目录转换为 Path 对象，便于路径操作
    output_root = Path(output_root)

    # 构建 DEMO 输入文件路径
    # 文件名格式：demo_{年份}{季度小写}.parquet，例如：demo_2024q1.parquet
    demo_file = output_root / f"demo_{year}{quarter.lower()}.parquet"

    # 构建 REAC 病例级输入文件路径
    # 文件名格式：reac_{年份}{季度小写}_case.parquet，例如：reac_2024q1_case.parquet
    reac_case_file = output_root / f"reac_{year}{quarter.lower()}_case.parquet"

    # 构建 DRUG 病例级药物特征输入文件路径
    # 文件名格式：drug_feature_{年份}{季度小写}_case.parquet，例如：drug_feature_2024q1_case.parquet
    drug_feature_file = (
        output_root / f"drug_feature_{year}{quarter.lower()}_case.parquet"
    )
    drug_exposure_file = (
        output_root / f"drug_exposure_{year}{quarter.lower()}_case.parquet"
    )
    outc_case_file = output_root / f"outc_{year}{quarter.lower()}_case.parquet"

    # 缺少前置产物时自动补跑，允许用户直接执行 case 流程。
    if not demo_file.exists():
        print(f"未找到 DEMO 结果文件，正在自动生成：{demo_file}")
        from demo_processor import process_demo

        process_demo(year, quarter, output_root)

    if not demo_file.exists():
        raise FileNotFoundError(f"找不到 DEMO 文件：{demo_file}")

    if not reac_case_file.exists():
        print(f"未找到 REAC 病例级文件，正在自动生成：{reac_case_file}")
        from reac_processor import process_reac

        process_reac(year, quarter, output_root)

    if not reac_case_file.exists():
        raise FileNotFoundError(f"找不到 REAC 病例级文件：{reac_case_file}")

    if not drug_feature_file.exists():
        print(f"未找到 DRUG 病例级特征文件，正在自动生成：{drug_feature_file}")
        from drug_feature_processor import process_drug_feature

        process_drug_feature(year, quarter, output_root)

    if not drug_feature_file.exists():
        raise FileNotFoundError(f"找不到 DRUG 病例级特征文件：{drug_feature_file}")

    # ========== 步骤 2: 读取数据 ==========
    # 读取去重后的 DEMO 数据（Parquet 格式）
    if not drug_exposure_file.exists():
        print(f"未找到 DRUG 病例级暴露定义文件，正在自动生成：{drug_exposure_file}")
        from drug_exposure_processor import process_drug_exposure

        process_drug_exposure(year, quarter, output_root)

    if not drug_exposure_file.exists():
        raise FileNotFoundError(f"找不到 DRUG 病例级暴露定义文件：{drug_exposure_file}")

    if not outc_case_file.exists():
        print(f"未找到 OUTC 病例级文件，正在自动生成：{outc_case_file}")
        from outc_processor import process_outc

        process_outc(year, quarter, output_root)

    if not outc_case_file.exists():
        raise FileNotFoundError(f"找不到 OUTC 病例级文件：{outc_case_file}")
    demo_df = pd.read_parquet(demo_file)

    # 读取 REAC 衍生的病例级跌倒标识数据（Parquet 格式）
    reac_case_df = pd.read_parquet(reac_case_file)

    # 读取 DRUG 衍生的病例级药物特征数据（Parquet 格式）
    drug_feature_df = pd.read_parquet(drug_feature_file)


    drug_exposure_df = pd.read_parquet(drug_exposure_file)
    outc_case_df = pd.read_parquet(outc_case_file)
    # ========== 步骤 3: 字段验证 ==========
    # 定义 DEMO 表必需的字段
    required_demo_cols = ["caseid"]
    # 检查 DEMO 表是否有缺失的必需列
    missing_demo_cols = [
        col for col in required_demo_cols if col not in demo_df.columns
    ]
    if missing_demo_cols:
        raise ValueError(f"DEMO 缺少必要字段：{missing_demo_cols}")

    # 定义 REAC 病例级数据必需的字段
    required_reac_cols = ["caseid", "is_fall"]
    # 检查 REAC 数据是否有缺失的必需列
    missing_reac_cols = [
        col for col in required_reac_cols if col not in reac_case_df.columns
    ]
    if missing_reac_cols:
        raise ValueError(f"REAC 病例级结果缺少必要字段：{missing_reac_cols}")

    required_drug_feature_cols = [
        "caseid",
        "is_zolpidem",
        "is_zaleplon",
        "is_zopiclone",
        "is_eszopiclone",
        "is_benzo",
        "is_antidepressant",
        "is_antipsychotic",
        "is_opioid",
        "is_antiepileptic",
        "drug_n",
        "distinct_drug_n",
    ]
    missing_drug_feature_cols = [
        col for col in required_drug_feature_cols if col not in drug_feature_df.columns
    ]
    # [ 要什么  for  挨个取  in  列表  if  条件 ]
    
    if missing_drug_feature_cols:
        raise ValueError(
            f"DRUG 病例级特征结果缺少必要字段：{missing_drug_feature_cols}"
        )
    if "polypharmacy_5" not in drug_feature_df.columns and "polypharmacy" not in drug_feature_df.columns:
        raise ValueError("DRUG 病例级特征结果缺少必要字段：['polypharmacy_5']")

    # ========== 步骤 4: 数据清洗 ==========
    # 清洗 DEMO 表的 caseid 字段：
    # 1. 将 NaN 替换为空字符串
    # 2. 转换为字符串类型
    # 3. 去除首尾空格
    if "suspect_role_any_ps" not in drug_exposure_df.columns and "suspect_role_any" in drug_exposure_df.columns:
        drug_exposure_df["suspect_role_any_ps"] = drug_exposure_df["suspect_role_any"]
    if (
        "is_zolpidem_suspect_ps" not in drug_exposure_df.columns
        and "is_zolpidem_suspect" in drug_exposure_df.columns
    ):
        drug_exposure_df["is_zolpidem_suspect_ps"] = drug_exposure_df["is_zolpidem_suspect"]
    if (
        "is_other_zdrug_suspect_ps" not in drug_exposure_df.columns
        and "is_other_zdrug_suspect" in drug_exposure_df.columns
    ):
        drug_exposure_df["is_other_zdrug_suspect_ps"] = drug_exposure_df["is_other_zdrug_suspect"]
    if "target_drug_group_ps" not in drug_exposure_df.columns and "target_drug_group" in drug_exposure_df.columns:
        drug_exposure_df["target_drug_group_ps"] = drug_exposure_df["target_drug_group"]

    required_drug_exposure_cols = [
        "caseid",
        "suspect_role_any",
        "suspect_role_any_ps",
        "is_zolpidem_suspect",
        "is_zolpidem_suspect_ps",
        "is_other_zdrug_suspect",
        "is_other_zdrug_suspect_ps",
        "target_drug_group",
        "target_drug_group_ps",
    ]
    missing_drug_exposure_cols = [
        col for col in required_drug_exposure_cols if col not in drug_exposure_df.columns
    ]
    if missing_drug_exposure_cols:
        raise ValueError(
            f"DRUG 病例级暴露定义结果缺少必要字段：{missing_drug_exposure_cols}"
        )

    required_outc_cols = ["caseid", *OUTC_BOOL_COLS]
    missing_outc_cols = [col for col in required_outc_cols if col not in outc_case_df.columns]
    if missing_outc_cols:
        raise ValueError(f"OUTC 病例级结果缺少必要字段：{missing_outc_cols}")
    demo_df["caseid"] = (
        demo_df["caseid"].where(demo_df["caseid"].notna(), "").astype(str).str.strip()
    )

    # 清洗 REAC 数据的 caseid 字段，处理方式与 DEMO 表相同
    reac_case_df["caseid"] = (
        reac_case_df["caseid"]
        .where(reac_case_df["caseid"].notna(), "")
        .astype(str)
        .str.strip()
    )

    drug_feature_df["caseid"] = (
        drug_feature_df["caseid"]
        .where(drug_feature_df["caseid"].notna(), "")
        .astype(str)
        .str.strip()
    )
    drug_exposure_df["caseid"] = (
        drug_exposure_df["caseid"]
        .where(drug_exposure_df["caseid"].notna(), "")
        .astype(str)
        .str.strip()
    )
    outc_case_df["caseid"] = (
        outc_case_df["caseid"].where(outc_case_df["caseid"].notna(), "").astype(str).str.strip()
    )

    # 处理 is_fall 字段：
    # 1. 将 NaN 值填充为 False（未报告跌倒）
    # 2. 转换为布尔类型
    reac_case_df["is_fall"] = reac_case_df["is_fall"].fillna(False).astype(bool)

    if "polypharmacy_5" not in drug_feature_df.columns and "polypharmacy" in drug_feature_df.columns:
        drug_feature_df["polypharmacy_5"] = drug_feature_df["polypharmacy"]
    if "polypharmacy" not in drug_feature_df.columns and "polypharmacy_5" in drug_feature_df.columns:
        # Keep legacy alias for downstream compatibility.
        drug_feature_df["polypharmacy"] = drug_feature_df["polypharmacy_5"]

    drug_bool_cols = [
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
        "polypharmacy",
    ]
    for col in drug_bool_cols:
        drug_feature_df[col] = drug_feature_df[col].fillna(False).astype(bool)

    for col in DRUG_EXPOSURE_BOOL_COLS:
        drug_exposure_df[col] = drug_exposure_df[col].fillna(False).astype(bool)
    drug_exposure_df["target_drug_group"] = (
        drug_exposure_df["target_drug_group"].where(
            drug_exposure_df["target_drug_group"].notna(), "no_suspect_drug"
        )
        .astype(str)
        .str.strip()
    )
    drug_exposure_df["target_drug_group_ps"] = (
        drug_exposure_df["target_drug_group_ps"].where(
            drug_exposure_df["target_drug_group_ps"].notna(), "no_suspect_drug"
        )
        .astype(str)
        .str.strip()
    )

    for col in OUTC_BOOL_COLS:
        outc_case_df[col] = outc_case_df[col].fillna(False).astype(bool)

    drug_feature_df["drug_n"] = (
        pd.to_numeric(drug_feature_df["drug_n"], errors="coerce").fillna(0).astype(int)
    )
    drug_feature_df["distinct_drug_n"] = (
        pd.to_numeric(drug_feature_df["distinct_drug_n"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # 过滤掉 caseid 为空的 DEMO 记录
    demo_df = demo_df[demo_df["caseid"] != ""]

    # 过滤掉 caseid 为空的 REAC 记录
    reac_case_df = reac_case_df[reac_case_df["caseid"] != ""]

    # 过滤掉 caseid 为空的 DRUG 特征记录
    drug_feature_df = drug_feature_df[drug_feature_df["caseid"] != ""]
    drug_exposure_df = drug_exposure_df[drug_exposure_df["caseid"] != ""]
    outc_case_df = outc_case_df[outc_case_df["caseid"] != ""]

    # 合并前校验病例级结果是否保持一病例一行，避免左连接时静默扩增主表行数。
    if reac_case_df["caseid"].duplicated().any():
        duplicated_caseids = reac_case_df.loc[
            reac_case_df["caseid"].duplicated(), "caseid"
        ].unique()
        raise ValueError(
            f"REAC 病例级结果存在重复 caseid，无法合并：{duplicated_caseids[:10].tolist()}"
        )

    if drug_feature_df["caseid"].duplicated().any():
        duplicated_caseids = drug_feature_df.loc[
            drug_feature_df["caseid"].duplicated(), "caseid"
        ].unique()
        raise ValueError(
            "DRUG 病例级特征结果存在重复 caseid，无法合并："
            f"{duplicated_caseids[:10].tolist()}"
        )

    if drug_exposure_df["caseid"].duplicated().any():
        duplicated_caseids = drug_exposure_df.loc[
            drug_exposure_df["caseid"].duplicated(), "caseid"
        ].unique()
        raise ValueError(
            "DRUG 病例级暴露定义结果存在重复 caseid，无法合并："
            f"{duplicated_caseids[:10].tolist()}"
        )

    if outc_case_df["caseid"].duplicated().any():
        duplicated_caseids = outc_case_df.loc[
            outc_case_df["caseid"].duplicated(), "caseid"
        ].unique()
        raise ValueError(
            "OUTC 病例级结果存在重复 caseid，无法合并："
            f"{duplicated_caseids[:10].tolist()}"
        )

    # ========== 步骤 5: 数据合并 ==========
    # 使用左连接（left join）合并 DEMO 和 REAC 数据
    # - 保留 DEMO 表中的所有有效病例
    # - 通过 caseid 进行关联
    # - 未在 REAC 中命中的病例，is_fall 为 NaN
    case_df = demo_df.merge(reac_case_df, on="caseid", how="left")
    case_df = case_df.merge(drug_feature_df, on="caseid", how="left")
    case_df = case_df.merge(drug_exposure_df, on="caseid", how="left")
    case_df = case_df.merge(outc_case_df, on="caseid", how="left")

    # 将合并后 is_fall 为 NaN 的记录填充为 False
    # 这些病例在 REAC 中没有跌倒相关术语记录
    case_df["is_fall"] = case_df["is_fall"].fillna(False).astype(bool)

    for col in drug_bool_cols:
        case_df[col] = case_df[col].fillna(False).astype(bool)

    for col in DRUG_EXPOSURE_BOOL_COLS:
        case_df[col] = case_df[col].fillna(False).astype(bool)
    case_df["target_drug_group"] = (
        case_df["target_drug_group"]
        .where(case_df["target_drug_group"].notna(), "no_suspect_drug")
        .astype(str)
        .str.strip()
    )
    case_df["target_drug_group_ps"] = (
        case_df["target_drug_group_ps"]
        .where(case_df["target_drug_group_ps"].notna(), "no_suspect_drug")
        .astype(str)
        .str.strip()
    )

    for col in OUTC_BOOL_COLS:
        case_df[col] = case_df[col].fillna(False).astype(bool)

    case_df["drug_n"] = (
        pd.to_numeric(case_df["drug_n"], errors="coerce").fillna(0).astype(int)
    )
    case_df["distinct_drug_n"] = (
        pd.to_numeric(case_df["distinct_drug_n"], errors="coerce").fillna(0).astype(int)
    )

    # ========== 步骤 6: 保存结果 ==========
    # 构建输出文件路径
    # 文件名格式：case_dataset_{年份}{季度小写}.parquet
    output_file = output_root / f"case_dataset_{year}{quarter.lower()}.parquet"

    # 保存为 Parquet 格式（高效的列式存储）
    # index=False 表示不保存行索引
    case_df.to_parquet(output_file, index=False)

    # 打印统计信息
    print("病例级分析数据行数:", len(case_df))
    print("跌倒病例数:", int(case_df["is_fall"].sum()))
    print("多药并用病例数(polypharmacy_5):", int(case_df["polypharmacy_5"].sum()))
    print(f"已保存：{output_file}")

    # 返回合并后的 DataFrame，供后续分析使用
    return case_df
