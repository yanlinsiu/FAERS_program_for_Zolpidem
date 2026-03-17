import argparse
from config import DEFAULT_OUTPUT_ROOT
from case_dataset_processor import process_case_dataset
from demo_processor import process_demo
from drug_exposure_processor import process_drug_exposure
from drug_processor import process_drug
from drug_feature_processor import process_drug_feature
from reac_processor import process_reac


def main():
    """
    FAERS 数据处理脚本的主入口函数

    功能:
    - 接收命令行参数（年份、季度、表名、输出目录）
    - 根据用户选择调用相应的处理函数
    - 支持单个表处理或批量处理所有表

    使用示例:
    # 处理 DEMO 表
    python main.py --year 2024 --quarter Q1 --table demo

    # 处理 DRUG 表
    python main.py --year 2024 --quarter Q1 --table drug

    # 处理 REAC 表
    python main.py --year 2024 --quarter Q1 --table reac

    # 处理病例级药物特征表
    python main.py --year 2024 --quarter Q1 --table drug_feature

    # 处理病例级研究暴露定义表
    python main.py --year 2024 --quarter Q1 --table drug_exposure

    # 构建病例级分析主表
    python main.py --year 2024 --quarter Q1 --table case

    # 批量处理所有表
    python main.py --year 2024 --quarter Q1 --table all
    """
    # ========== 创建命令行参数解析器 ==========
    # argparse 是 Python 标准库，用于解析命令行参数
    parser = argparse.ArgumentParser(description="FAERS 数据处理脚本")

    # ========== 添加必需参数：年份 ==========
    # required=True 表示该参数必须提供
    # type=int 限制输入必须是整数
    parser.add_argument("--year", required=True, type=int, help="年份，例如 2024")

    # ========== 添加必需参数：季度 ==========
    # choices 参数限制了可选值，只能是 Q1-Q4 或 q1-q4
    # 这样可以在解析时自动验证输入是否合法
    parser.add_argument(
        "--quarter",
        required=True,
        type=str,
        choices=["Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"],
        help="季度，例如 Q1",
    )

    # ========== 添加必需参数：表名 ==========
    # 用户可以选择处理哪个表，或者选择 "all" 批量处理所有表
    parser.add_argument(
        "--table",
        required=True,
        type=str,
        choices=["demo", "drug", "drug_feature", "drug_exposure", "reac", "case", "all"],
        help=(
            "要处理的表：demo(患者信息), drug(药物信息), "
            "drug_feature(病例级药物特征表), drug_exposure(病例级研究暴露定义表), "
            "reac(不良反应), "
            "case(病例级分析表), all(全部)"
        ),
    )

    # ========== 添加可选参数：输出目录 ==========
    # default=DEFAULT_OUTPUT_ROOT 表示如果不指定则使用默认输出目录
    # 从 config.py 中导入的常量
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=str,
        help="输出目录（可选，默认使用配置文件中的路径）",
    )

    # ========== 解析命令行参数 ==========
    # parse_args() 会解析用户输入的参数，并返回一个包含所有参数的对象
    args = parser.parse_args()

    # ========== 提取参数值 ==========
    # 从 args 对象中提取各个参数的值，方便后续使用
    year = args.year
    quarter = args.quarter.upper()  # 统一转为大写，如 'q1' → 'Q1'
    table = args.table.lower()  # 统一转为小写，确保匹配逻辑正确
    output_root = args.output

    # ========== 根据表名调用相应的处理函数 ==========
    if table == "demo":
        # 处理 DEMO 表（患者人口统计信息）
        process_demo(year, quarter, output_root)

    elif table == "drug":
        # 处理 DRUG 表（药物信息）
        process_drug(year, quarter, output_root)

    elif table == "reac":
        # 处理 REAC 表（不良反应事件信息）
        process_reac(year, quarter, output_root)

    elif table == "drug_feature":
        # 处理 DRUG 衍生的病例级药物特征表
        process_drug_feature(year, quarter, output_root)

    elif table == "drug_exposure":
        # 处理 DRUG 衍生的病例级研究暴露定义表
        process_drug_exposure(year, quarter, output_root)

    elif table == "case":
        # 构建病例级分析主表（DEMO + REAC + DRUG_FEATURE）
        process_case_dataset(year, quarter, output_root)

    elif table == "all":
        # 批量处理所有表，并构建病例级分析主表
        print("=" * 50)
        print("开始批量处理所有 FAERS 数据表...")
        print("=" * 50)

        process_demo(year, quarter, output_root)
        process_drug(year, quarter, output_root)
        process_drug_feature(year, quarter, output_root)
        process_drug_exposure(year, quarter, output_root)
        process_reac(year, quarter, output_root)
        process_case_dataset(year, quarter, output_root)

        print("=" * 50)
        print("所有数据处理完成！")
        print("=" * 50)


if __name__ == "__main__":
    """
    Python 程序的入口点判断
    
    作用:
    - 当直接运行此文件时 (__name__ == "__main__")，执行 main() 函数
    - 当作为模块被其他文件导入时，不会自动执行 main()
    
    这是 Python 的标准写法，确保代码可以被复用
    """
    main()
