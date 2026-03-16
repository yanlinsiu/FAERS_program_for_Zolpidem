import argparse
from config import DEFAULT_OUTPUT_ROOT
from demo_processor import process_demo
from drug_processor import process_drug
from reac_processor import process_reac
from utils import iter_quarters


def process_one_period(year, quarter, table, output_root):
    if table == "demo":
        process_demo(year, quarter, output_root)
    elif table == "drug":
        process_drug(year, quarter, output_root)
    elif table == "reac":
        process_reac(year, quarter, output_root)
    elif table == "all":
        process_demo(year, quarter, output_root)
        process_drug(year, quarter, output_root)
        process_reac(year, quarter, output_root)


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
    
    # 批量处理所有表
    python main.py --year 2024 --quarter Q1 --table all
    """
    # ========== 创建命令行参数解析器 ==========
    # argparse 是 Python 标准库，用于解析命令行参数
    parser = argparse.ArgumentParser(description="FAERS 数据处理脚本")

    # ========== 添加必需参数：年份 ==========
    # required=True 表示该参数必须提供
    # type=int 限制输入必须是整数
    parser.add_argument(
        "--year", 
        required=True, 
        type=int, 
        help="年份，例如 2024"
    )

    # ========== 添加必需参数：季度 ==========
    # choices 参数限制了可选值，只能是 Q1-Q4 或 q1-q4
    # 这样可以在解析时自动验证输入是否合法
    parser.add_argument(
        "--quarter",
        required=True,
        type=str,
        choices=["Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"],
        help="季度，例如 Q1"
    )

    parser.add_argument(
        "--end-year",
        type=int,
        help="结束年份；不传时默认等于 --year"
    )

    parser.add_argument(
        "--end-quarter",
        type=str,
        choices=["Q1", "Q2", "Q3", "Q4", "q1", "q2", "q3", "q4"],
        help="结束季度；不传时默认等于 --quarter"
    )

    # ========== 添加必需参数：表名 ==========
    # 用户可以选择处理哪个表，或者选择 "all" 批量处理所有表
    parser.add_argument(
        "--table",
        required=True,
        type=str,
        choices=["demo", "drug", "reac", "all"],
        help="要处理的表：demo(患者信息), drug(药物信息), reac(不良反应), all(全部)"
    )

    # ========== 添加可选参数：输出目录 ==========
    # default=DEFAULT_OUTPUT_ROOT 表示如果不指定则使用默认输出目录
    # 从 config.py 中导入的常量
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_ROOT,
        type=str,
        help="输出目录（可选，默认使用配置文件中的路径）"
    )

    # ========== 解析命令行参数 ==========
    # parse_args() 会解析用户输入的参数，并返回一个包含所有参数的对象
    args = parser.parse_args()

    # ========== 提取参数值 ==========
    # 从 args 对象中提取各个参数的值，方便后续使用
    year = args.year
    quarter = args.quarter.upper()  # 统一转为大写，如 'q1' → 'Q1'
    end_year = args.end_year
    end_quarter = args.end_quarter.upper() if args.end_quarter else None
    table = args.table.lower()      # 统一转为小写，确保匹配逻辑正确
    output_root = args.output

    periods = list(iter_quarters(year, quarter, end_year, end_quarter))

    if len(periods) > 1:
        print("=" * 50)
        print(f"开始批量处理 {len(periods)} 个季度...")
        print(f"范围：{periods[0][0]} {periods[0][1]} -> {periods[-1][0]} {periods[-1][1]}")
        print("=" * 50)

    for current_year, current_quarter in periods:
        print(f"当前处理：{current_year} {current_quarter} | 表：{table}")
        process_one_period(current_year, current_quarter, table, output_root)

    if len(periods) > 1:
        print("=" * 50)
        print("批量处理完成！")
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
