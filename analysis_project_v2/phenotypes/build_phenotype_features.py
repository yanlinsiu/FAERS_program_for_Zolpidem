from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# 项目根目录路径（当前文件向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 输出根目录
OUTPUT_ROOT = PROJECT_ROOT / "OUTPUT"
# 全局数据集目录
GLOBAL_DATASET_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "datasets"
# 表型特征输出目录
PHENOTYPE_OUTPUT_DIR = PROJECT_ROOT / "OUTPUT_GLOBAL" / "phenotypes"


@dataclass(frozen=True)
class PhenotypeSpec:
    """
    表型规格定义
    
    Attributes:
        column: 对应的特征列名
        layer: 表型层级（prodrome/secondary_prodrome/fall_event/consequence）
        label: 表型的显示标签
        pt_terms: 包含的 MedDRA PT（首选术语）列表
    """
    column: str
    layer: str
    label: str
    pt_terms: tuple[str, ...]


# 定义所有跌倒相关表型的规格
# 分为四个层级：前驱症状、次级前驱症状、跌倒事件、后果
PHENOTYPE_SPECS: tuple[PhenotypeSpec, ...] = (
    # 镇静/嗜睡 - 前驱症状层
    PhenotypeSpec(
        column="pheno_sedation_somnolence",
        layer="prodrome",
        label="Sedation / somnolence",
        pt_terms=("SOMNOLENCE", "SEDATION", "HYPERSOMNIA", "LETHARGY"),
    ),
    # 意识/认知改变 - 前驱症状层
    PhenotypeSpec(
        column="pheno_consciousness_cognition",
        layer="prodrome",
        label="Consciousness / cognition change",
        pt_terms=(
            "ALTERED STATE OF CONSCIOUSNESS",
            "DEPRESSED LEVEL OF CONSCIOUSNESS",
            "LOSS OF CONSCIOUSNESS",
            "CONFUSIONAL STATE",
            "DISORIENTATION",
            "DELIRIUM",
            "COGNITIVE DISORDER",
            "DISTURBANCE IN ATTENTION",
            "MEMORY IMPAIRMENT",
            "MENTAL IMPAIRMENT",
            "MENTAL STATUS CHANGES",
        ),
    ),
    # 头晕/眩晕/晕厥 - 前驱症状层
    PhenotypeSpec(
        column="pheno_dizziness_vertigo_syncope",
        layer="prodrome",
        label="Dizziness / vertigo / syncope",
        pt_terms=(
            "DIZZINESS",
            "VERTIGO",
            "VERTIGO POSITIONAL",
            "VERTIGO CNS ORIGIN",
            "VESTIBULAR DISORDER",
            "SYNCOPE",
            "PRESYNCOPE",
        ),
    ),
    # 步态/平衡/运动控制异常 - 前驱症状层
    PhenotypeSpec(
        column="pheno_gait_balance_motor",
        layer="prodrome",
        label="Gait / balance / motor control abnormality",
        pt_terms=(
            "GAIT DISTURBANCE",
            "GAIT INABILITY",
            "BALANCE DISORDER",
            "ATAXIA",
            "COORDINATION ABNORMAL",
            "MOBILITY DECREASED",
            "MOVEMENT DISORDER",
        ),
    ),
    # 低血压/体位性低血压 - 前驱症状层
    PhenotypeSpec(
        column="pheno_hypotension",
        layer="prodrome",
        label="Hypotension / orthostatic hypotension",
        pt_terms=("HYPOTENSION", "ORTHOSTATIC HYPOTENSION", "BLOOD PRESSURE DECREASED"),
    ),
    # 视觉障碍 - 次级前驱症状层
    PhenotypeSpec(
        column="pheno_visual_disturbance",
        layer="secondary_prodrome",
        label="Visual disturbance",
        pt_terms=("VISUAL IMPAIRMENT", "VISUAL ACUITY REDUCED", "VISION BLURRED"),
    ),
    # 跌倒事件 - 跌倒事件层
    PhenotypeSpec(
        column="pheno_fall_event",
        layer="fall_event",
        label="Fall event",
        pt_terms=("FALL",),
    ),
    # 骨折/损伤后果 - 后果层
    PhenotypeSpec(
        column="pheno_fracture_injury",
        layer="consequence",
        label="Fracture / injury consequence",
        pt_terms=(
            "FRACTURE",
            "HIP FRACTURE",
            "FEMUR FRACTURE",
            "INJURY",
            "HEAD INJURY",
            "CRANIOCEREBRAL INJURY",
            "CONTUSION",
            "WOUND",
            "SKIN LACERATION",
        ),
    ),
    # 住院（PT级别）- 后果层
    PhenotypeSpec(
        column="pheno_hospitalisation_pt",
        layer="consequence",
        label="Hospitalisation PT",
        pt_terms=("HOSPITALISATION",),
    ),
)


def _norm_term(value: object) -> str:
    """
    标准化 MedDRA 术语
    
    Args:
        value: 原始术语值
        
    Returns:
        标准化后的术语：空值返回空字符串，否则转为大写并去除首尾空格
    """
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _find_meddra_file() -> Path:
    """
    查找项目根目录下的 MedDRA Excel 文件
    
    Returns:
        按文件名排序后的第一个 MedDRA*.xlsx 文件路径
        
    Raises:
        FileNotFoundError: 当找不到任何 MedDRA*.xlsx 文件时抛出异常
    """
    candidates = sorted(PROJECT_ROOT.glob("MedDRA*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No MedDRA*.xlsx file found in project root.")
    return candidates[0]


def build_meddra_llt_to_pt_map(meddra_file: Path) -> dict[str, str]:
    """
    构建 MedDRA LLT（最低层术语）到 PT（首选术语）的映射字典
    
    Args:
        meddra_file: MedDRA Excel 文件路径
        
    Returns:
        字典，键为 LLT 或 PT 术语，值为对应的 PT 术语
        
    Raises:
        ValueError: 当 MedDRA 文件缺少必需列时抛出异常
    """
    # 读取 MedDRA 文件的第一个工作表
    meddra = pd.read_excel(meddra_file, sheet_name=0)
    required = {"llt_english", "pt_english"}
    missing = required - set(meddra.columns)
    if missing:
        raise ValueError(f"MedDRA sheet missing columns: {sorted(missing)}")

    llt_to_pt = {}
    # 遍历每一行，建立 LLT 到 PT 的映射关系
    for row in meddra[["llt_english", "pt_english"]].dropna(subset=["pt_english"]).itertuples(index=False):
        llt = _norm_term(row.llt_english)
        pt = _norm_term(row.pt_english)
        if llt and pt:
            # LLT 映射到 PT
            llt_to_pt[llt] = pt
            # PT 自身也映射到 PT（保持一致性）
            llt_to_pt[pt] = pt
    return llt_to_pt


def build_dictionary_frame(llt_to_pt: dict[str, str]) -> pd.DataFrame:
    """
    构建表型词典数据框，记录每个表型的定义和映射信息
    
    Args:
        llt_to_pt: LLT 到 PT 的映射字典
        
    Returns:
        包含表型定义的 DataFrame，列包括：
        - phenotype_column: 特征列名
        - layer: 表型层级
        - label: 显示标签
        - pt_term: PT 术语
        - pt_term_in_meddra_map: 该 PT 是否在 MedDRA 映射中存在
        - mapped_pt: 映射后的 PT 术语
        
    Raises:
        ValueError: 当同一个 PT 术语被分配到多个表型类别时抛出异常
    """
    rows: list[dict[str, object]] = []
    term_to_category: dict[str, list[str]] = {}
    
    # 遍历所有表型规格，构建词典条目
    for spec in PHENOTYPE_SPECS:
        for pt in spec.pt_terms:
            term_to_category.setdefault(pt, []).append(spec.column)
            rows.append(
                {
                    "phenotype_column": spec.column,
                    "layer": spec.layer,
                    "label": spec.label,
                    "pt_term": pt,
                    "pt_term_in_meddra_map": pt in llt_to_pt,
                    "mapped_pt": llt_to_pt.get(pt, pt),
                }
            )
    
    dictionary = pd.DataFrame(rows)
    
    # 检查是否有 PT 术语被重复分配到多个表型类别
    duplicate_terms = {
        term: columns for term, columns in term_to_category.items() if len(set(columns)) > 1
    }
    if duplicate_terms:
        raise ValueError(f"PT terms assigned to multiple phenotype categories: {duplicate_terms}")
    
    return dictionary


def load_case_index(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """
    加载病例索引数据并进行清洗和过滤
    
    Args:
        path: 病例索引 Parquet 文件路径
        start_year: 起始年份
        end_year: 结束年份
        
    Returns:
        清洗后的病例索引 DataFrame，包含 caseid、primaryid、year、quarter 列
        
    Raises:
        ValueError: 当文件缺少必需列时抛出异常
    """
    case_index = pd.read_parquet(path)
    required = {"caseid", "primaryid", "year", "quarter"}
    missing = required - set(case_index.columns)
    if missing:
        raise ValueError(f"case index missing columns: {sorted(missing)}")
    
    # 选择必需列并创建副本
    out = case_index[["caseid", "primaryid", "year", "quarter"]].copy()
    
    # 清洗 caseid：空值替换为空字符串，转换为字符串类型并去除空格
    out["caseid"] = out["caseid"].where(out["caseid"].notna(), "").astype(str).str.strip()
    # 转换 primaryid 为数值类型
    out["primaryid"] = pd.to_numeric(out["primaryid"], errors="coerce")
    # 转换 year 为整数类型（可空）
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    # 清洗 quarter：空值替换为空字符串，转大写并去除空格
    out["quarter"] = out["quarter"].where(out["quarter"].notna(), "").astype(str).str.upper().str.strip()
    
    # 过滤指定年份范围内的数据
    out = out[out["year"].between(start_year, end_year, inclusive="both")].copy()
    # 过滤掉 caseid 为空或 primaryid 为空的记录
    out = out[(out["caseid"] != "") & out["primaryid"].notna()].copy()
    
    return out


def _quarter_sort_key(value: str) -> int:
    """
    将季度字符串转换为排序用的整数
    
    Args:
        value: 季度字符串（如 "Q1", "Q2" 等）
        
    Returns:
        季度数字（1-4），如果格式不正确则返回 0
    """
    value = str(value).upper().strip()
    if value.startswith("Q") and value[1:].isdigit():
        return int(value[1:])
    return 0


def process_quarter(
    case_index_quarter: pd.DataFrame,
    year: int,
    quarter: str,
    llt_to_pt: dict[str, str],
    pt_to_column: dict[str, str],
    cleaned_output_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    处理单个季度的 REAC 数据，生成表型特征
    
    Args:
        case_index_quarter: 该季度的病例索引数据
        year: 年份
        quarter: 季度标识
        llt_to_pt: LLT 到 PT 的映射字典
        pt_to_column: PT 术语到表型列名的映射字典
        cleaned_output_root: 清洗后数据的根目录
        
    Returns:
        元组 (表型特征 DataFrame, 质量控制统计字典)
        
    Raises:
        FileNotFoundError: 当找不到清洗后的 REAC 事件文件时抛出异常
    """
    # 构建 REAC 事件文件路径
    file_path = cleaned_output_root / str(year) / "quarterly" / f"reac_event_{year}{quarter.lower()}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned REAC event file not found: {file_path}. "
            "Run faers_project/year_batch_runner.py first."
        )

    # 读取 REAC 数据，只加载需要的列
    reac = pd.read_parquet(file_path, columns=["caseid", "primaryid", "pt"])
    reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
    reac = reac[reac["primaryid"].notna()].copy()
    
    # 标准化报告的术语
    reac["reported_term"] = reac["pt"].map(_norm_term)
    reac = reac[reac["reported_term"] != ""].copy()

    # 清洗 caseid
    reac["caseid"] = reac["caseid"].where(reac["caseid"].notna(), "").astype(str).str.strip()
    
    # 与病例索引合并，确保只保留有索引的记录
    merged = reac.merge(case_index_quarter[["caseid", "primaryid"]], on=["caseid", "primaryid"], how="inner")
    
    # 如果合并且结果为空，返回空结果和质量控制统计
    if merged.empty:
        empty = case_index_quarter[["caseid"]].copy()
        for spec in PHENOTYPE_SPECS:
            empty[spec.column] = False
        empty["phenotype_pt_list"] = ""
        return empty, {
            "year": year,
            "quarter": quarter,
            "reac_rows": int(len(reac)),
            "matched_reac_rows": 0,
            "matched_cases_with_reac": 0,
            "matched_phenotype_rows": 0,
            "unmapped_term_rows": int((~reac["reported_term"].isin(llt_to_pt)).sum()),
        }

    # 将报告术语映射到 PT 术语
    merged["pt_term"] = merged["reported_term"].map(llt_to_pt).fillna(merged["reported_term"])
    # 将 PT 术语映射到表型列名
    merged["phenotype_column"] = merged["pt_term"].map(pt_to_column)
    # 筛选出匹配到表型的记录
    phenotype_rows = merged[merged["phenotype_column"].notna()].copy()

    # 以病例为基础构建特征表
    base = case_index_quarter[["caseid"]].drop_duplicates().copy()
    # 初始化所有表型列为 False
    for spec in PHENOTYPE_SPECS:
        base[spec.column] = False

    # 如果有匹配到表型的记录，设置相应的标志位
    if not phenotype_rows.empty:
        # 使用 pivot 将长格式转换为宽格式，每个表型列一个布尔标志
        flags = (
            phenotype_rows[["caseid", "phenotype_column"]]
            .drop_duplicates()
            .assign(value=True)
            .pivot(index="caseid", columns="phenotype_column", values="value")
            .fillna(False)
            .reset_index()
        )
        # 合并标志位到基础表
        base = base.merge(flags, on="caseid", how="left", suffixes=("", "_hit"))
        # 更新每个表型列的值
        for spec in PHENOTYPE_SPECS:
            hit_col = f"{spec.column}_hit"
            if hit_col in base.columns:
                base[spec.column] = base[hit_col].fillna(False).astype(bool)
                base = base.drop(columns=[hit_col])
            else:
                base[spec.column] = base[spec.column].fillna(False).astype(bool)

        # 构建每个病例匹配的 PT 术语列表（用 | 分隔）
        pt_list = (
            phenotype_rows[["caseid", "pt_term"]]
            .drop_duplicates()
            .sort_values(["caseid", "pt_term"])
            .groupby("caseid")["pt_term"]
            .apply(lambda values: "|".join(values))
            .reset_index(name="phenotype_pt_list")
        )
        base = base.merge(pt_list, on="caseid", how="left")
    else:
        base["phenotype_pt_list"] = ""

    # 填充空值为空字符串
    base["phenotype_pt_list"] = base["phenotype_pt_list"].fillna("")
    
    # 统计未映射的术语数量
    unmapped_rows = int((~merged["reported_term"].isin(llt_to_pt)).sum())
    
    # 构建质量控制统计信息
    qc = {
        "year": year,
        "quarter": quarter,
        "reac_rows": int(len(reac)),
        "matched_reac_rows": int(len(merged)),
        "matched_cases_with_reac": int(merged["caseid"].nunique()),
        "matched_phenotype_rows": int(len(phenotype_rows)),
        "unmapped_term_rows": unmapped_rows,
    }
    
    return base, qc


def build_phenotype_features(
    start_year: int,
    end_year: int,
    case_index_file: Path,
    output_dir: Path,
    cleaned_output_root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    """
    构建完整的表型特征数据集
    
    Args:
        start_year: 起始年份
        end_year: 结束年份
        case_index_file: 病例索引文件路径
        output_dir: 输出目录
        cleaned_output_root: 清洗后数据的根目录
        
    Returns:
        字典，包含生成的文件路径：
        - features: 表型特征 Parquet 文件
        - dictionary: 表型词典 CSV 文件
        - qc: 质量控制 CSV 文件
        - summary: 汇总表型统计 CSV 文件
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    period_token = f"{start_year}_{end_year}"

    # 构建 MedDRA 映射和表型词典
    llt_to_pt = build_meddra_llt_to_pt_map(_find_meddra_file())
    dictionary = build_dictionary_frame(llt_to_pt)
    pt_to_column = dict(zip(dictionary["pt_term"], dictionary["phenotype_column"], strict=True))

    # 加载病例索引
    case_index = load_case_index(case_index_file, start_year, end_year)
    
    # 存储各季度的处理结果
    phenotype_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    
    # 按年份和季度分组处理
    for (year, quarter), quarter_cases in case_index.groupby(["year", "quarter"], sort=True):
        year_int = int(year)
        quarter_str = str(quarter).upper()
        print(f"Processing REAC phenotype features: {year_int} {quarter_str}")
        
        # 处理当前季度的数据
        part, qc = process_quarter(
            case_index_quarter=quarter_cases,
            year=year_int,
            quarter=quarter_str,
            llt_to_pt=llt_to_pt,
            pt_to_column=pt_to_column,
            cleaned_output_root=cleaned_output_root,
        )
        phenotype_parts.append(part)
        qc_rows.append(qc)

    # 合并所有季度的结果
    phenotype = pd.concat(phenotype_parts, ignore_index=True)
    
    # 与病例索引左连接，确保包含所有病例
    phenotype = case_index[["caseid"]].drop_duplicates().merge(phenotype, on="caseid", how="left")
    
    # 填充缺失的表型列为 False
    for spec in PHENOTYPE_SPECS:
        phenotype[spec.column] = phenotype[spec.column].fillna(False).astype(bool)
    
    # 填充 PT 列表为空字符串
    phenotype["phenotype_pt_list"] = phenotype["phenotype_pt_list"].fillna("")

    # 生成表型汇总表
    summary_rows = []
    total_cases = len(phenotype)
    for spec in PHENOTYPE_SPECS:
        n = int(phenotype[spec.column].sum())
        summary_rows.append(
            {
                "phenotype_column": spec.column,
                "layer": spec.layer,
                "label": spec.label,
                "case_count": n,
                "case_percent": round(n / total_cases * 100, 4) if total_cases else 0.0,
            }
        )

    # 定义输出文件路径
    feature_file = output_dir / f"phenotype_features_{period_token}_case.parquet"
    dictionary_file = output_dir / f"phenotype_dictionary_{period_token}.csv"
    qc_file = output_dir / f"phenotype_build_qc_{period_token}.csv"
    summary_file = output_dir / f"phenotype_summary_{period_token}.csv"

    # 保存所有输出文件
    phenotype.to_parquet(feature_file, index=False)
    dictionary.to_csv(dictionary_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(qc_rows).sort_values(
        ["year", "quarter"], key=lambda col: col.map(_quarter_sort_key) if col.name == "quarter" else col
    ).to_csv(qc_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False, encoding="utf-8-sig")

    return {
        "features": feature_file,
        "dictionary": dictionary_file,
        "qc": qc_file,
        "summary": summary_file,
    }


def main() -> None:
    """主函数：解析命令行参数并执行表型特征构建"""
    parser = argparse.ArgumentParser(description="Build case-level fall phenotype features from FAERS REAC.")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--case-index-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PHENOTYPE_OUTPUT_DIR)
    parser.add_argument("--cleaned-output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    # 生成时间段标识符
    period_token = f"{args.start_year}_{args.end_year}"
    
    # 确定病例索引文件路径
    case_index_file = args.case_index_file or GLOBAL_DATASET_DIR / f"global_case_index_{period_token}.parquet"
    
    # 执行表型特征构建
    outputs = build_phenotype_features(
        start_year=args.start_year,
        end_year=args.end_year,
        case_index_file=case_index_file,
        output_dir=args.output_dir,
        cleaned_output_root=args.cleaned_output_root,
    )
    
    print("phenotype feature build completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
