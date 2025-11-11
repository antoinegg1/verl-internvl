import pandas as pd
from tqdm import tqdm

# 需要保留的列
MIN_COLS = [
    "id", "image", "height", "width", "conversations",
    "241b_model_iou", "8b_v7_model_iou", "2b_v5_iou",
    "qwen3-vl-2b-cot_model_iou", "qwen3-vl-4b-cot_model_iou",
    "qwen3-vl-2b-direct_box_model_iou", "2b_v7_2_iou",
    "qwen3-vl-4b-direct_box_model_iou"
]

# 明确哪些列应当是数值列（会被强制转成 float 并填 0.0）
NUM_COLS = [
    "height", "width",
    "241b_model_iou", "8b_v7_model_iou", "2b_v5_iou",
    "qwen3-vl-2b-cot_model_iou", "qwen3-vl-4b-cot_model_iou",
    "qwen3-vl-2b-direct_box_model_iou", "2b_v7_2_iou",
    "qwen3-vl-4b-direct_box_model_iou"
]

# 读 JSONL -> 只保留 MIN_COLS -> 统一清洗 -> 写回 JSONL
def filter_jsonl(input_file, output_file):
    df = pd.read_json(input_file, lines=True)

    # 若缺少列则先补列（NaN），并统一列顺序
    for c in MIN_COLS:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df))
    df = df[MIN_COLS]

    # 数值列：把常见占位符替换为缺失，再 to_numeric 强转，最后 fillna(0.0)
    repl = {"": pd.NA, "None": pd.NA, "none": pd.NA, "null": pd.NA, "NaN": pd.NA, "nan": pd.NA}
    for c in tqdm(NUM_COLS, desc="Coercing numeric columns"):
        s = df[c]
        s = s.replace(repl)
        df[c] = pd.to_numeric(s, errors="coerce").fillna(0.0)

    # 非数值列：把 None/NaN 统一填为 0.0（按你的要求）
    non_num_cols = [c for c in MIN_COLS if c not in NUM_COLS]
    for c in tqdm(non_num_cols, desc="Filling non-numeric columns"):
        df[c] = df[c].where(df[c].notna(), 0.0)

    # 写回 JSONL
    df.to_json(output_file, orient="records", lines=True)
    print(f"Filtered data saved to {output_file}")

# 示例：假设你有一个名为 `input.jsonl` 的输入文件
filter_jsonl('/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1104_final.jsonl', '/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1104_final.jsonl')
