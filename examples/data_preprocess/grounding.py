"""
Preprocess grounding datasets to parquet format.

Train JSON example:
{"image": "...", "conversations": [{"from":"human","value":"<image>..."}, {"from":"gpt","value":"...<box>[x1,y1,x2,y2]</box>..."}], ...}

Test JSON example:
{"image": "...", "sent": "...", "bbox": [x1,y1,x2,y2], "height": H, "width": W}
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs  # noqa: F401


# {"id": -1, "image": "coco/train2014/COCO_train2014_000000573297.jpg", "height": 640, "width": 384, "conversations": [{"from": "human", "value": "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>man wearing a red costume</ref>"}, {"from": "gpt", "value": "<think>To correctly identify the bounding box for \"man wearing a red costume,\" focus on the distinct red clothing and any unique features such as accessories or armor that align with the description. Ensure the selected area encompasses the entire figure described, differentiating from other similar objects by emphasizing the red color and contextually relevant details.</think><answer><ref>man wearing a red costume</ref><box>[81, 104, 998, 999]</box></answer>"}], "1b_model_iou": 0.9620241352816716, "8b_model_iou": 0.9890789935127147, "14b_model_iou": 0.9845281630184352}
def extract_answer(answer_raw: str) -> list[int]:
    pattern = r"<box>\s*\[(.*?)\]\s*</box>"
    match = re.search(pattern, answer_raw)
    if match:
        return list(map(int, match.group(1).strip().split(",")))
    else:
        raise ValueError(f"Answer format is incorrect: {answer_raw}")


if __name__ == "__main__":
    # Simple hard-coded IoU thresholds for filtering (kept for future use)
    IOU_1B_MIN = 0.0
    IOU_8B_MIN = 0.1
    IOU_14B_MIN = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None, help="Directory containing input JSON files")
    parser.add_argument("--output_dir", default=None, help="Directory to write output parquet files")
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=["train.json"],
        help="One or more train JSON files or glob patterns (relative to --input_dir if not absolute)",
    )
    parser.add_argument(
        "--test_files",
        nargs="+",
        default=["test.json"],
        help="One or more test JSON files or glob patterns (relative to --input_dir if not absolute)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes for map/filter (datasets num_proc)",
    )

    # parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # Resolve IO directories
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    num_workers = int(args.num_workers)
    print(f"[grounding preprocess] using num_workers={num_workers}")

    # Resolve files by simple path join (no globbing)
    def expand_files(files):
        return [os.path.join(input_dir, f) for f in files]

    train_files = expand_files(args.train_files)
    test_files = expand_files(args.test_files)

    # Define a simple per-row filter based on model IoUs
    def _keep_by_iou(ex):
        try:
            iou1 = float(ex.get("1b_model_iou", float("nan")))
            iou8 = float(ex.get("8b_model_iou", float("nan")))
            iou14 = float(ex.get("14b_model_iou", float("nan")))
        except Exception:
            return False
        return (iou1 < 0.5) and (iou14 < iou1 + 1.0) and (iou14 > 0.5)

    BASE_IMG_PATH = "/storage/openpsi/data/coco/train2014/"

    # Convert pixel-space bbox to a 0..1000 grid (float) using width/height
    def _bbox_px_to_norm_1000(bbox, w: int, h: int):
        x1, y1, x2, y2 = bbox
        w_f = float(w)
        h_f = float(h)
        nx1 = float(x1) * 1000.0 / w_f
        ny1 = float(y1) * 1000.0 / h_f
        nx2 = float(x2) * 1000.0 / w_f
        ny2 = float(y2) * 1000.0 / h_f
        return [nx1, ny1, nx2, ny2]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            conversation = example.pop("conversations")
            question_raw = conversation[0]["value"]
            answer_raw = conversation[1]["value"]
            img_name = os.path.basename(example["image"])  # only filename
            img_path = os.path.join(BASE_IMG_PATH, img_name)

            if not os.path.exists(img_path):
                print(f"[Warning] Image not found: {img_path}")
                return None

            images = [img_path]

            # For grounding, use the raw question directly
            question = question_raw
            solution = extract_answer(answer_raw)
            data = {
                "data_source": "grounding",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "images": images,
                "ability": "grounding",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "id": example.get("id"),
                    "height": example.get("height"),
                    "width": example.get("width"),
                    "question": question,
                },
            }
            return data

        return process_fn

    # Map function for test-style records: {image, sent, bbox, height, width}
    def make_map_fn_test():
        def process_fn(example, idx):
            img_name = os.path.basename(example["image"])
            img_path = os.path.join(BASE_IMG_PATH, img_name)
            sent = example["sent"]
            bbox = example["bbox"]
            H = example["height"]
            W = example["width"]
            if not os.path.exists(img_path):
                print(f"[Warning] Image not found: {img_path}")

            content = (
                f"<image>\nPlease provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref>"
            )
            data = {
                "data_source": "grounding",
                "prompt": [{"role": "user", "content": content}],
                "images": [img_path],
                "ability": "grounding",
                # store normalized GT to [0,1000] (float) to match training format
                "reward_model": {"style": "rule", "ground_truth": _bbox_px_to_norm_1000(bbox, W, H)},
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

    # Process train: only one file is expected; write to train.parquet
    if len(train_files) > 0:
        train_path = train_files[0]
        ds = datasets.load_dataset("json", data_files=train_path)["train"]
        ds = ds.filter(_keep_by_iou, num_proc=num_workers)
        ds = ds.map(function=make_map_fn("train"), with_indices=True, num_proc=num_workers)
        print(f"[grounding preprocess] train samples: {len(ds)}")
        ds.to_parquet(os.path.join(output_dir, "train_1B_v3.parquet"))

    # Process test files: sample 10% from each and mix into a single dataset
    # mixed_slices = []
    # for file_path in test_files:
    #     ds = datasets.load_dataset("json", data_files=file_path)["train"]
    #     ds = ds.map(function=make_map_fn_test(), with_indices=True, num_proc=num_workers)
    #     n = len(ds)
    #     k = max(1, int(n * 0.1)) if n > 0 else 0
    #     if k > 0:
    #         ds_slice = ds.shuffle(seed=42).select(range(k))
    #         mixed_slices.append(ds_slice)

    # if len(mixed_slices) > 0:
    #     mixed = datasets.concatenate_datasets(mixed_slices).shuffle(seed=42)
    #     print(f"[grounding preprocess] test samples (mixed 10% per file): {len(mixed)}")
    #     mixed.to_parquet(os.path.join(output_dir, "test_mixed.parquet"))
    # else:
    #     print("[grounding preprocess] no test samples produced (no input or all empty)")

