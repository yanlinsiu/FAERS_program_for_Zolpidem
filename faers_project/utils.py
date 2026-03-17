import pandas as pd
from pathlib import Path


def read_faers_txt(file_path):
    """
    读取 FAERS ASCII txt 文件
    
    参数:
        file_path (str or Path): FAERS 文本文件的完整路径
        
    返回:
        pd.DataFrame: 读取后的 DataFrame，列名已统一转为小写并去除空格
        
    注意:
        - 使用 $ 作为分隔符（FAERS 数据库标准格式）
        - 使用 latin1 编码以兼容特殊字符
        - low_memory=False 避免列类型推断不一致的问题
    """
    # 读取 FAERS 文本文件，指定分隔符、编码和内存模式
    df = pd.read_csv(file_path, sep="$", encoding="latin1", low_memory=False)
    # 将列名去除空格并转为小写，统一命名规范
    df.columns = df.columns.str.strip().str.lower()
    return df


def build_file_path(raw_root, year, quarter, table_name):
    """
    根据输入参数自动拼接 FAERS 原始数据文件的完整路径
    
    参数:
        raw_root (str): 根目录路径，例如 D:\\program_FAERS
        year (int or str): 年份，例如 2024
        quarter (str): 季度，例如 "Q1"、"Q2"、"Q3"、"Q4"
        table_name (str): 表名，例如 "DEMO"、"DRUG"、"REAC"等
        
    返回:
        Path: 拼接后的完整文件路径对象
        
    示例:
        输入：raw_root="D:\\program_FAERS", year=2024, quarter="Q1", table_name="DRUG"
        输出：D:\\program_FAERS\\2024\\Q1\\ASCII\\DRUG24Q1.txt
        
    路径格式说明:
        {raw_root}/{year}/{quarter}/ASCII/{table_name}{年尾 2 位}{quarter}.txt
    """
    # 统一转换为大写和字符串格式
    year = str(year)
    quarter = quarter.upper()
    table_name = table_name.upper()

    # 取年份的后两位，例如 2024 -> 24
    year_short = year[-2:]
    # 构建文件名，格式为：表名 + 年份后两位 + 季度.txt
    filename = f"{table_name}{year_short}{quarter}.txt"

    # 使用 Path 对象拼接完整路径
    return Path(raw_root) / year / quarter / "ASCII" / filename


def deduplicate_demo_records(df):
    """
    清洗 DEMO 数据并按 caseid 保留每个病例的最新记录
    
    去重策略:
        1. 首先检查必要字段是否存在
        2. 清洗 caseid 字段（去除空格、处理空值）
        3. 转换数值字段为正确的数据类型
        4. 按时间先后和版本号排序
        5. 保留每个 caseid 的最后一条记录（即最新版本）
    
    参数:
        df (pd.DataFrame): 原始 DEMO 数据 DataFrame
        
    返回:
        pd.DataFrame: 去重后的 DataFrame，每个 caseid 仅保留一条最新记录
        
    异常:
        ValueError: 当缺少必要字段时抛出
        
    必要字段说明:
        - caseid: 病例 ID
        - primaryid: 主要标识符
        - fda_dt: FDA 接收日期
        - caseversion: 病例版本号
    """
    # 定义必要的列名
    required_cols = ["caseid", "primaryid", "fda_dt", "caseversion"]
    # 检查是否有缺失的列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DEMO 缺少必要字段：{missing_cols}")

    # 创建副本以避免修改原始数据
    deduped_df = df.copy()
    
    # 清洗 caseid 列：将 NaN 替换为空字符串，转为字符串类型，去除首尾空格
    deduped_df["caseid"] = (
        deduped_df["caseid"]
        .where(deduped_df["caseid"].notna(), "")
        .astype(str)
        .str.strip()
    )
    # 过滤掉 caseid 为空的记录
    deduped_df = deduped_df[deduped_df["caseid"] != ""]
    
    # 将 primaryid 转换为数值类型，无法转换的设为 NaN
    deduped_df["primaryid"] = pd.to_numeric(deduped_df["primaryid"], errors="coerce")
    # 将 caseversion 转换为数值类型，无法转换的设为 NaN
    deduped_df["caseversion"] = pd.to_numeric(
        deduped_df["caseversion"], errors="coerce"
    )
    # 将 fda_dt 转换为日期格式，格式为 YYYYMMDD，无法转换的设为 NaT
    deduped_df["fda_dt"] = pd.to_datetime(
        deduped_df["fda_dt"], format="%Y%m%d", errors="coerce"
    )

    # 按 caseid、日期、版本号、primaryid 升序排序
    # 这样每个 caseid 的最后一条记录就是最新的版本
    deduped_df = deduped_df.sort_values(
        by=["caseid", "fda_dt", "caseversion", "primaryid"],
        ascending=[True, True, True, True],
    )
    # 删除重复的 caseid，保留最后一条（即最新的）记录
    return deduped_df.drop_duplicates(subset="caseid", keep="last")


def apply_demo_demographic_criteria(df):
    """
    在 DEMO 去重后执行人口学标准化与纳排：
    1. 标准化年龄字段 age_years（基于 age + age_cod）
    2. 仅保留 65 <= age_years <= 120
    3. 生成年龄分组 age_group：65-74, 75-84, >=85
    4. 清洗性别 sex_clean：M/F/UNK（优先 sex，缺失则回退 gndr_cod）

    支持年龄单位:
        YR, MON, WK, DY, HR, DEC
    其他单位视为缺失，后续会在年龄筛选中剔除。
    """
    required_age_cols = ["age", "age_cod"]
    missing_age_cols = [col for col in required_age_cols if col not in df.columns]
    if missing_age_cols:
        raise ValueError(f"DEMO 缺少年龄标准化所需字段：{missing_age_cols}")

    out_df = df.copy()

    age_value = pd.to_numeric(out_df["age"], errors="coerce")
    age_unit = (
        out_df["age_cod"]
        .where(out_df["age_cod"].notna(), "")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    age_years = pd.Series(float("nan"), index=out_df.index)
    age_years.loc[age_unit == "YR"] = age_value.loc[age_unit == "YR"]
    age_years.loc[age_unit == "MON"] = age_value.loc[age_unit == "MON"] / 12
    age_years.loc[age_unit == "WK"] = age_value.loc[age_unit == "WK"] / 52
    age_years.loc[age_unit == "DY"] = age_value.loc[age_unit == "DY"] / 365
    age_years.loc[age_unit == "HR"] = age_value.loc[age_unit == "HR"] / (24 * 365)
    age_years.loc[age_unit == "DEC"] = age_value.loc[age_unit == "DEC"] * 10
    out_df["age_years"] = age_years

    out_df = out_df[out_df["age_years"].between(65, 120, inclusive="both")].copy()

    out_df["age_group"] = pd.cut(
        out_df["age_years"],
        bins=[65, 75, 85, float("inf")],
        labels=["65-74", "75-84", ">=85"],
        right=False,
    ).astype(str)

    # 逐行优先 sex，当前行缺失/无效时回退 gndr_cod，最后置为 UNK
    sex_raw = pd.Series("", index=out_df.index, dtype="object")
    if "sex" in out_df.columns:
        sex_raw = (
            out_df["sex"]
            .where(out_df["sex"].notna(), "")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "gndr_cod" in out_df.columns:
        gndr_raw = (
            out_df["gndr_cod"]
            .where(out_df["gndr_cod"].notna(), "")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        sex_raw = sex_raw.where(sex_raw.isin(["M", "F"]), gndr_raw)

    out_df["sex_clean"] = sex_raw.map({"M": "M", "F": "F"}).fillna("UNK")

    return out_df


def load_retained_demo_primaryids(raw_root, year, quarter, output_root=None):
    """
    读取指定年份和季度的 DEMO 数据，返回去重后保留的 primaryid 集合
    
    参数:
        raw_root (str): 根目录路径
        year (int or str): 年份
        quarter (str): 季度
        
    返回:
        set: 去重后 DEMO 记录对应的 primaryid 集合
        
    异常:
        FileNotFoundError: 当找不到 DEMO 文件时抛出
        
    用途:
        用于获取有效的病例 ID 列表，以便在其他表（如 DRUG、REAC）中筛选出这些病例的记录
    """
    if output_root is not None:
        demo_parquet = Path(output_root) / f"demo_{year}{str(quarter).lower()}.parquet"
        if demo_parquet.exists():
            demo_df = pd.read_parquet(demo_parquet)
            if "primaryid" not in demo_df.columns:
                raise ValueError(f"DEMO 结果缺少必要字段：['primaryid']，文件：{demo_parquet}")
            primaryid = pd.to_numeric(demo_df["primaryid"], errors="coerce")
            return set(primaryid.dropna())

    # 回退：若未找到 demo parquet，则从原始 DEMO 读取并应用同口径规则
    demo_file = build_file_path(raw_root, year, quarter, "DEMO")
    if not demo_file.exists():
        raise FileNotFoundError(f"找不到文件：{demo_file}")

    demo_df = read_faers_txt(demo_file)
    demo_df = deduplicate_demo_records(demo_df)
    demo_df = apply_demo_demographic_criteria(demo_df)
    primaryid = pd.to_numeric(demo_df["primaryid"], errors="coerce")
    return set(primaryid.dropna())


def iter_quarters(start_year, start_quarter, end_year=None, end_quarter=None):
    """
    生成从起始季度到结束季度的所有季度迭代器（包含起止季度）
    
    参数:
        start_year (int or str): 起始年份
        start_quarter (str): 起始季度，例如 "Q1"
        end_year (int or str, optional): 结束年份，默认为 None（与 start_year 相同）
        end_quarter (str, optional): 结束季度，默认为 None（与 start_quarter 相同）
        
    返回:
        generator: 产生 (year, quarter) 元组的生成器
        
    异常:
        ValueError: 当季度格式不正确或起始季度晚于结束季度时抛出
        
    示例:
        list(iter_quarters(2023, "Q4", 2024, "Q2"))
        返回：[(2023, 'Q4'), (2024, 'Q1'), (2024, 'Q2')]
        
        list(iter_quarters(2024, "Q1"))
        返回：[(2024, 'Q1')]
    """
    # 定义季度顺序列表
    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    # 创建季度到索引的映射字典，便于比较和计算
    quarter_index = {quarter: index for index, quarter in enumerate(quarter_order)}

    # 转换年份为整数，季度为大写
    start_year = int(start_year)
    start_quarter = start_quarter.upper()
    # 如果未指定结束年份，默认为起始年份
    end_year = start_year if end_year is None else int(end_year)
    # 如果未指定结束季度，默认为起始季度
    end_quarter = start_quarter if end_quarter is None else end_quarter.upper()

    # 验证季度格式是否合法
    if start_quarter not in quarter_index or end_quarter not in quarter_index:
        raise ValueError("quarter must be one of Q1, Q2, Q3, Q4")

    # 创建起始和结束的 (年份，季度索引) 元组用于比较
    start_key = (start_year, quarter_index[start_quarter])
    end_key = (end_year, quarter_index[end_quarter])
    # 确保起始时间不晚于结束时间
    if start_key > end_key:
        raise ValueError("start quarter must be earlier than or equal to end quarter")

    # 初始化当前遍历的年份和季度位置
    year = start_year
    quarter_pos = quarter_index[start_quarter]
    # 循环生成季度，直到超过结束季度
    while (year, quarter_pos) <= end_key:
        # 产出当前季度元组
        yield year, quarter_order[quarter_pos]
        # 移动到下一个季度
        quarter_pos += 1
        # 如果超过 Q4，则年份加 1，季度重置为 Q1
        if quarter_pos == len(quarter_order):
            year += 1
            quarter_pos = 0
