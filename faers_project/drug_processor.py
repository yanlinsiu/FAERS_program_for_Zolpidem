from utils import read_faers_txt, build_file_path
from config import RAW_ROOT
from pathlib import Path


def process_drug(year, quarter, output_root):
    """
    处理 FAERS DRUG 数据（药物信息）
    
    主要步骤:
    1. 读取原始药物数据
    2. 查看数据基本信息（行数、列名）
    3. 保存为 Parquet 格式
    
    DRUG 文件包含的药物信息:
    - caseid: 病例 ID
    - drugname: 药物名称
    - drugdose: 药物剂量
    - drugther: 药物治疗作用
    - 等其他药物相关字段
    """
    # 构建输入文件路径（使用 f-string 格式化路径）
    # DRUG 文件包含病例中报告的所有药物信息
    file = build_file_path(RAW_ROOT, year, quarter, "DRUG")
    print(f"正在处理文件：{file}")

    if not file.exists():
        raise FileNotFoundError(f"找不到文件：{file}")


    # 读取 FAERS 药物数据文件
    # read_faers_txt 会自动处理:
    # - 分隔符 ($)
    # - 编码 (latin1)
    # - 列名转小写
    df = read_faers_txt(file)

    # 打印数据行数，了解数据规模
    print("DRUG 行数:", len(df))

    # 打印所有列名，查看可用的药物信息字段
    print("DRUG 列名:")
    print(list(df.columns))

    # ========== 保存处理后的数据 ==========
    # 构建输出文件路径，保存为 Parquet 格式
    # Parquet 是高效的列式存储格式，适合大规模数据分析
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 保存数据，index=False 表示不保存行索引
    output_file = output_root / f"drug_{year}{quarter.lower()}.parquet"
    df.to_parquet(output_file, index=False)

    # 打印完成提示
    print("DRUG parquet 保存完成:", output_file)
