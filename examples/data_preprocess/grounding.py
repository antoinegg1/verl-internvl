"""
Preprocess grounding train dataset to parquet format with IoU-based sampling.

Train JSON example:
{"image": "...",
 "conversations": [
    {"from": "human", "value": "<image>..."},
    {"from": "gpt",   "value": "...<box>[x1,y1,x2,y2]</box>..."}
 ],
 "iou_main": 0.82,
 "241b_model_iou": 0.75,
 ...}
"""
from PIL import Image
import argparse
import os
import re
import random
import datasets

# 图片根目录从环境变量读取，未设置时使用默认值
BASE_IMG_PATH = os.getenv("BASE_IMG_PATH")


def extract_answer(answer_raw: str) -> list[int]:
    pattern = r"<box>\s*\[(.*?)\]\s*</box>"
    match = re.search(pattern, answer_raw)
    if not match:
        raise ValueError(f"Answer format is incorrect: {answer_raw}")
    return list(map(int, match.group(1).strip().split(",")))


def make_map_fn(split: str, fmt: str):
    """
    fmt: "default" or "qwen"
    """

    def process_fn(example, idx):
        conversation = example.pop("conversations")

        orig_h = example.get("height")
        orig_w = example.get("width")

        # 归一化 image 路径
        img_rel = example["image"].replace(
            "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/gzhan/", ""
        ).replace(
            "/lustre/fsw/portfolios/nvr/users/yunhaof/datasets/", ""
        )
        img_path = os.path.join(BASE_IMG_PATH, img_rel)

        if os.path.exists(img_path):
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
        else:
            print(f"[Warning] Image not found: {img_path}")

        # 构造问题
        if fmt == "qwen" and "sent" in example:
            prompt = "<image>\nLocate {sent}, output its bbox coordinates using JSON format"
            question_raw = prompt.format(sent=example["sent"])
        else:
            question_raw = conversation[0]["value"]

        # 构造答案
        answer_raw = conversation[1]["value"]
        solution = extract_answer(answer_raw)

        data = {
            "data_source": "grounding",
            "prompt": [
                {
                    "role": "user",
                    "content": question_raw,
                }
            ],
            "images": [img_path],
            "ability": "grounding",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "id": example.get("id"),
                "height": orig_h,
                "width": orig_w,
                "question": question_raw,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess grounding train JSON file to parquet with IoU-based sampling."
    )
    parser.add_argument(
        "--input_json",
        required=True,
        help="Input train JSON file path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output parquet file directory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Number of processes for datasets.map.",
    )
    parser.add_argument(
        "--format",
        choices=["default", "qwen"],
        help="Output prompt format.",
    )
    parser.add_argument(
        "--iou_key",
        required=True,
        help="Field name of main IoU score in JSON rows.",
    )

    args = parser.parse_args()

    input_json = os.path.expanduser(args.input_json)
    output_dir = os.path.expanduser(args.output_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_workers = int(args.num_workers)
    fmt = args.format
    iou_key = args.iou_key

    ds_all = datasets.load_dataset("json", data_files=input_json)["train"]

    # 负样本：主 iou < 0.5 且 241b_model_iou > 0.5
    def keep_neg(ex):
        if iou_key not in ex or "241b_model_iou" not in ex:
            return False
        iou_main = float(ex[iou_key])
        iou241 = float(ex["241b_model_iou"])
        return (iou_main < 0.5) and (iou241 > 0.5)

    ds_neg = ds_all.filter(keep_neg, num_proc=num_workers)
    n_neg = ds_neg.num_rows
    print(f"[grounding preprocess] neg pool: {n_neg}")

    if n_neg == 0:
        raise RuntimeError("No negative samples found with given IoU conditions.")

    # 正样本：IoU >= 0.5（仅主 iou）
    def keep_pos(ex):
        if iou_key not in ex:
            return False
        return float(ex[iou_key]) >= 0.5

    ds_pos = ds_all.filter(keep_pos, num_proc=num_workers)
    n_pos = ds_pos.num_rows
    print(f"[grounding preprocess] pos pool: {n_pos}")

    if n_pos == 0:
        raise RuntimeError("No positive samples found (iou >= 0.5).")

    # 正样本目标数量：与负样本 1:1
    n_pos_target = n_neg

    # IoU 分桶配置：区间与比例
    # 0.5~1 按 0.15, 0.2, 0.2, 0.2, 0.25 分配
    buckets = [
        (0.5, 0.6, 0.15),
        (0.6, 0.7, 0.20),
        (0.7, 0.8, 0.20),
        (0.8, 0.9, 0.20),
        (0.9, 1.01, 0.25),
    ]

    bucket_targets = []
    acc = 0
    for idx_bucket, (_, _, ratio) in enumerate(buckets):
        if idx_bucket < len(buckets) - 1:
            t = int(round(n_pos_target * ratio))
            bucket_targets.append(t)
            acc += t
        else:
            t = max(0, n_pos_target - acc)
            bucket_targets.append(t)

    rng = random.Random(42)
    pos_parts = []

    for (lo, hi, _), target in zip(buckets, bucket_targets):
        if target <= 0:
            print(f"[pos bucket {lo:.2f}-{hi:.2f}) target=0, skip")
            continue

        def in_bucket(ex, lo=lo, hi=hi):
            if iou_key not in ex:
                return False
            v = float(ex[iou_key])
            return (v >= lo) and (v < hi)

        ds_bucket = ds_pos.filter(in_bucket, num_proc=num_workers)
        n_bucket = ds_bucket.num_rows

        if n_bucket == 0:
            print(f"[pos bucket {lo:.2f}-{hi:.2f}) available=0, take=0")
            continue

        take = min(target, n_bucket)
        indices = list(range(n_bucket))
        if take < n_bucket:
            indices = rng.sample(indices, take)
        indices = sorted(indices)

        pos_parts.append(ds_bucket.select(indices))
        print(
            f"[pos bucket {lo:.2f}-{hi:.2f}) available={n_bucket}, "
            f"target={target}, take={take}"
        )

    if len(pos_parts) == 0:
        raise RuntimeError("No positive samples selected; check IoU distribution and buckets.")

    ds_pos_sampled = datasets.concatenate_datasets(pos_parts)
    print(
        f"[grounding preprocess] pos sampled: {ds_pos_sampled.num_rows} "
        f"(target={n_pos_target})"
    )

    # 合并正负样本并打乱
    ds_train_raw = datasets.concatenate_datasets([ds_pos_sampled, ds_neg]).shuffle(seed=42)
    print(f"[grounding preprocess] merged train size (raw): {ds_train_raw.num_rows}")

    # 映射到最终格式
    ds_train = ds_train_raw.map(
        function=make_map_fn("train", fmt),
        with_indices=True,
        num_proc=num_workers,
    )
    print(f"[grounding preprocess] train samples (final): {ds_train.num_rows}")

    ds_train.to_parquet(output_dir/f"train_grounding.parquet")
    print(f"[grounding preprocess] wrote {output_dir/f'train_grounding.parquet'}")