from datasets import load_dataset, Dataset

# ============================================================
# 1. Streaming 读取原始数据集
#    不需要把完整 60k / 2GB 数据全部下载下来
# ============================================================

dataset = load_dataset(
    "Albert-CAC/dpo",
    split="train",
    streaming=True,
)

# ============================================================
# 2. 只取前 50 条
# ============================================================

samples = list(dataset.take(50))

dataset_50 = Dataset.from_list(samples)

print(dataset_50)
print(dataset_50.column_names)
print(dataset_50[0])

# ============================================================
# 3. 保存成 Hugging Face 常用格式
# ============================================================

# JSONL
dataset_50.to_json(
    "dpo_50.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)

# Parquet
dataset_50.to_parquet(
    "dpo_50.parquet"
)

print("Saved 50 samples.")