import re
import pandas as pd

def preprocess_drug_name(drug_name):
    """
    对 FAERS 原始药名做预处理，便于后续 RxNav 映射
    """
    # 1. 处理空值
    if pd.isna(drug_name):
        return None

    # 2. 转成字符串 + 小写
    name = str(drug_name).lower().strip()

    # 3. 一些明显无意义的值，直接丢掉
    invalid_values = {
        "", "unknown", "unk", "none", "na", "n/a", 
        "null", "drug", "medication", "other", "?"
    }
    if name in invalid_values:
        return None

    # 4. 将常见分隔符替换为空格
    name = re.sub(r"[/,;:_\-]+", " ", name)

    # 5. 去掉括号内容
    # 例如 "zolpidem (oral)" -> "zolpidem"
    name = re.sub(r"\(.*?\)", " ", name)

    # 6. 去掉剂量信息
    # 例如 10 mg, 12.5mg, 100 mcg, 5ml
    name = re.sub(r"\b\d+(\.\d+)?\s*(mg|g|mcg|μg|ug|ml|iu|%)\b", " ", name)

    # 7. 去掉常见剂型/给药途径词
    dosage_words = {
        "tablet", "tablets", "tab", "tabs",
        "capsule", "capsules", "cap", "caps",
        "injection", "inj", "injectable",
        "oral", "solution", "suspension", "syrup",
        "cream", "ointment", "gel", "patch",
        "extended", "release", "xr", "sr", "er", "cr",
        "film", "coated", "powder", "kit"
    }
    words = name.split()
    words = [w for w in words if w not in dosage_words]
    name = " ".join(words)

    # 8. 只保留字母、数字、空格
    name = re.sub(r"[^a-z0-9\s]", " ", name)

    # 9. 合并多个空格
    name = re.sub(r"\s+", " ", name).strip()

    # 10. 如果清洗后为空，返回 None
    if not name:
        return None

    return name