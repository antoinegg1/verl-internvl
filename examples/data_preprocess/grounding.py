"""
Preprocess grounding datasets to parquet format.

Train JSON example:
{"image": "...", "conversations": [{"from":"human","value":"<image>..."}, {"from":"gpt","value":"...<box>[x1,y1,x2,y2]</box>..."}], ...}

Test JSON example:
{"image": "...", "sent": "...", "bbox": [x1,y1,x2,y2], "height": H, "width": W}
"""
from PIL import Image
import argparse
import os
import re
import json
import ast
import numpy as np
import datasets

from verl.utils.hdfs_io import copy, makedirs  # noqa: F401

BASE_IMG_PATH = "/storage/openpsi/data/"
QWEN_MAX_AREA = 641*641

# --------- helpers ---------
def extract_sent(row):
    text = " ".join(x.get("value", "") for x in row["conversations"])
    m = re.findall(r'<ref>\s*["“]?(.+?)["”]?\s*</ref>', text, flags=re.S)
    if not m:
        return ""
    return m[-1].strip()


def extract_answer(answer_raw: str) -> list[int]:
    pattern = r"<box>\s*\[(.*?)\]\s*</box>"
    match = re.search(pattern, answer_raw)
    if not match:
        raise ValueError(f"Answer format is incorrect: {answer_raw}")
    return list(map(int, match.group(1).strip().split(",")))


def bbox_px_to_norm_1000(bbox, w: int, h: int):
    x1, y1, x2, y2 = bbox
    w_f = float(w)
    h_f = float(h)
    nx1 = float(x1) * 1000.0 / w_f
    ny1 = float(y1) * 1000.0 / h_f
    nx2 = float(x2) * 1000.0 / w_f
    ny2 = float(y2) * 1000.0 / h_f
    return [nx1, ny1, nx2, ny2]


def round_to_16(x):
    if x is None:
        return None
    x = int(x)
    return (x + 15) // 16 * 16


def is_multiple_of_16(x):
    if x is None:
        return False
    x = int(x)
    return x % 16 == 0



# --------- map fns ---------
def make_map_fn(split: str, fmt: str, iou_key: str):
    """
    fmt: "default" or "qwen"
    """

    def process_fn(example, idx):
        conversation = example.pop("conversations")

        orig_h = example.get("height")
        orig_w = example.get("width")
        img_rel = example["image"].replace("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/gzhan/","").replace("/lustre/fsw/portfolios/nvr/users/yunhaof/datasets/","")
        img_path = os.path.join(BASE_IMG_PATH, img_rel)

        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")
            return None
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
        # 构造问题/答案
        if fmt == "qwen" and "sent" in example:
            prompt = "<image>\nLocate {sent}, output its bbox coordinates using JSON format"
            question_raw = prompt.format(sent=example["sent"])
        else:
            question_raw = conversation[0]["value"]

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
            "reward_model": {"style": "rule", "ground_truth": solution},
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


def make_map_fn_test(fmt: str):
    def process_fn(example, idx):
        # test 这边原始数据里本来就有 sent/bbox/height/width
        base_img_path = "/storage/openpsi/data" if fmt == "qwen" else BASE_IMG_PATH
        img_rel = example["image"].replace("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/users/gzhan/","").replace("/lustre/fsw/portfolios/nvr/users/yunhaof/datasets/","")
        img_path = os.path.join(base_img_path, img_rel)

        sent = example["sent"]
        bbox = example["bbox"]
        with Image.open(img_path) as im:
            W, H = im.size

        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}")

        if fmt == "qwen":
            content = f"<image>\nLocate {sent}, output its bbox coordinates using JSON format"
        else:
            content = f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref>"

        data = {
            "data_source": "grounding",
            "prompt": [{"role": "user", "content": content}],
            "images": [img_path],
            "ability": "grounding",
            "reward_model": {"style": "rule", "ground_truth": bbox_px_to_norm_1000(bbox, W, H)},
            "extra_info": {
                "split": "test",
                "index": idx,
                "height": H,
                "width": W,
                "question": sent,
            },
        }

        return data

    return process_fn


# --------- main ---------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", required=True, help="Directory to write output parquet files")
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=["train.json"],
        help="One or more train JSON files (relative to --input_dir if not absolute)",
    )
    parser.add_argument(
        "--test_files",
        nargs="+",
        default=["test.json"],
        help="One or more test JSON files (relative to --input_dir if not absolute)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="datasets num_proc",
    )
    parser.add_argument(
        "--format",
        choices=["default", "qwen"],
        help="output prompt format",
    )
    parser.add_argument(
        "--process-test",
        action="store_true",
        help="whether to process test json and write parquet",
    )
    parser.add_argument(
        "--bucket-specs",
        type=str,
        help='JSON list like [[0.5,0.6,3991],[0.6,0.7,7981],...]',
    )
    parser.add_argument(
        "--iou-key",
        type=str,
        help="Key for the main model IOU used for filtering and bucketing",
    )

    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_workers = int(args.num_workers)
    fmt = args.format
    print(fmt)
    iou_key = args.iou_key

    def expand_files(files):
        return [os.path.join(input_dir, f) for f in files]

    train_files = expand_files(args.train_files)
    test_files = expand_files(args.test_files)

    # 解析 bucket specs
    bucket_specs_raw = ast.literal_eval(args.bucket_specs)

    bucket_specs = []
    for item in bucket_specs_raw:
        lo, hi, k = float(item[0]), float(item[1]), int(item[2])
        bucket_specs.append((lo, hi, k))

    # 训练数据
    if len(train_files) > 0:
        train_path = train_files[0]
        ds_all = datasets.load_dataset("json", data_files=train_path)["train"]
        ds_all= ds_all.map(
            function=make_map_fn("train", fmt, iou_key),
            with_indices=True,
            num_proc=num_workers,
        )
        print(f"[grounding preprocess] train samples (final): {len(ds_all)}")
        out_train = os.path.join(
            output_dir,
            "test_amodal_qwen.parquet",
        )
        ds_all.to_parquet(out_train)
        print(f"[grounding preprocess] wrote {out_train}")
        exit(0)
        if fmt == "qwen":
            def keep_qwen_size(ex):
                h = ex.get("height")
                w = ex.get("width")
                if h is not None and w is not None:
                    return int(h) * int(w) <= QWEN_MAX_AREA
                img_rel = ex["image"]
                img_path = os.path.join(BASE_IMG_PATH, img_rel)
                if not os.path.exists(img_path):
                    return False
                with Image.open(img_path) as im:
                    W, H = im.size
                return int(W) * int(H) <= QWEN_MAX_AREA

            ds_all = ds_all.filter(keep_qwen_size, num_proc=num_workers)
            print(f"[grounding preprocess] qwen size-filtered pool: {ds_all.num_rows}")

        # iou 过滤和分桶
        def keep_neg(ex):
            if iou_key not in ex or "241b_model_iou" not in ex:
                return False
            iou_main = float(ex[iou_key])
            iou241 = float(ex["241b_model_iou"])
            return (iou_main < 0.5) and (iou241 > 0.4)

        ds_neg = ds_all.filter(keep_neg, num_proc=num_workers)
        print(f"[grounding preprocess] neg pool: {ds_neg.num_rows}")

        def keep_pos(ex):
            if iou_key not in ex:
                return False
            return float(ex[iou_key]) >= 0.5

        ds_pos = ds_all.filter(keep_pos, num_proc=num_workers)
        print(f"[grounding preprocess] pos pool: {ds_pos.num_rows}")

        pos_parts = []
        rng = np.random.default_rng(42)
        for lo, hi, k in bucket_specs:
            def in_bucket(ex, lo=lo, hi=hi):
                v = float(ex[iou_key])
                return (v >= lo) and (v < hi)

            ds_bucket = ds_pos.filter(in_bucket, num_proc=num_workers)
            n_bucket = ds_bucket.num_rows
            take = min(k, n_bucket)
            if take == 0:
                print(f"[pos bucket {lo:.2f}-{hi:.2f}) available=0, take=0")
                continue
            idx = rng.choice(n_bucket, size=take, replace=False)
            pos_parts.append(ds_bucket.select(sorted(idx)))
            print(f"[pos bucket {lo:.2f}-{hi:.2f}) available={n_bucket}, take={take}")

        if len(pos_parts) == 0:
            raise RuntimeError("No positive samples selected; check input stats.")

        ds_pos_sampled = datasets.concatenate_datasets(pos_parts)
        ds_train = datasets.concatenate_datasets([ds_pos_sampled, ds_neg]).shuffle(seed=42)
        print(f"[grounding preprocess] merged train size: {ds_train.num_rows}")

        ds_train = ds_train.map(
            function=make_map_fn("train", fmt, iou_key),
            with_indices=True,
            num_proc=num_workers,
        )
        print(f"[grounding preprocess] train samples (final): {len(ds_train)}")
        out_train = os.path.join(
            output_dir,
            "train_grounding_qwen4bcot_80K_filter_641.parquet",
        )
        ds_train.to_parquet(out_train)
        print(f"[grounding preprocess] wrote {out_train}")

    # 测试数据（可选）
    if args.process_test and len(test_files) > 0:
        mixed_slices = []
        for file_path in test_files:
            ds_t = datasets.load_dataset("json", data_files=file_path)["train"]
            ds_t = ds_t.map(function=make_map_fn_test(fmt), with_indices=True, num_proc=num_workers)
            n = len(ds_t)
            k = max(1, int(n * 0.1)) if n > 0 else 0
            if k > 0:
                mixed_slices.append(ds_t.shuffle(seed=42).select(range(k)))
        if len(mixed_slices) > 0:
            mixed = datasets.concatenate_datasets(mixed_slices).shuffle(seed=42)
            out_test = os.path.join(output_dir, "test_grounding_{}.parquet".format(fmt))
            mixed.to_parquet(out_test)
            print(f"[grounding preprocess] test samples: {len(mixed)}, wrote {out_test}")
        else:
            print("[grounding preprocess] no test samples produced")
