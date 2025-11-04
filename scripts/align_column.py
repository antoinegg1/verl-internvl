import json
from datasets import load_dataset, concatenate_datasets
from datasets import Features, Value, Sequence
from tqdm import tqdm

MIN_COLS = ["id","image","height","width","conversations","241b_model_iou","8b_v7_model_iou"]

def align_to_mincols(ds):
    cols = set(ds.column_names)
    keep = [c for c in MIN_COLS if c in cols]
    ds = ds.select_columns(keep)
    for c in MIN_COLS:
        if c not in ds.column_names:
            ds = ds.add_column(c, [None] * len(ds))
    ds = ds.select_columns(MIN_COLS)
    return ds

paths = [
    "/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-exclude-onlyflip_1103.jsonl",
    "/storage/openpsi/data/grounding_sft_v1/stage2_thinking_with_text_sft_train_10_24_3.jsonl"
]

aligned = []
for p in tqdm(paths, desc="Loading & aligning files"):
    d = load_dataset("json", data_files=p)["train"]
    d = align_to_mincols(d)
    aligned.append(d)

ds = concatenate_datasets(aligned)


# 进度条保存为 JSONL
def save_jsonl(ds, out_path):
    ds = ds.with_format("python")
    total = ds.num_rows
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, total=total, desc="Writing JSONL"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

save_jsonl(ds, "/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1103.jsonl")
