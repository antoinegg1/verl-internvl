import os, base64, argparse, asyncio
from typing import List, Dict, Any
from datasets import load_dataset, Dataset, concatenate_datasets
from openai import AsyncOpenAI
from tqdm import tqdm
import re
from PIL import Image
import numpy as np
import json
import hashlib

PROMPT_TEMPLATE = (
    # "Please provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref> " 
    "Please provide the amodal bounding box coordinate of the region this sentence describes: <ref>{sent}</ref>" # Amodal
    # "Locate {sent}, output its bbox coordinates using JSON format. " #for Qwen
)



def hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def summarize_ious(ious: List[float]) -> Dict[str, float]:
    valid = [x for x in ious if x is not None]
    if not valid:
        return {"mean_iou": 0.0, "pass_rate_05": 0.0}
    mean_iou = sum(valid) / len(valid)
    pass_rate_05 = sum(1 for x in valid if x > 0.5) / len(valid)
    return {"mean_iou": mean_iou, "pass_rate_05": pass_rate_05}

def scale_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * w / 1000.0)
    x2 = round(x2 * w / 1000.0)
    y1 = round(y1 * h / 1000.0)
    y2 = round(y2 * h / 1000.0)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x1 == x2 and w > 1: x2 = min(x1 + 1, w - 1)
    if y1 == y2 and h > 1: y2 = min(y1 + 1, h - 1)
    return [float(x1), float(y1), float(x2), float(y2)]

def compute_iou(boxA, boxB):
    """计算两个bbox的IoU"""
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    if areaA + areaB - inter_area == 0:
        return 0.0
    return inter_area / (areaA + areaB - inter_area)

async def call_one(client: AsyncOpenAI, model: str, rec: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    rec["image"] = rec["image"].replace("lustre/fsw/portfolios/nvr/users/yunhaof/datasets", "").lstrip("/")
    base_image_dir = "/storage/openpsi/data"
    path = os.path.join(base_image_dir, rec["image"])
    with Image.open(path) as img:
        w, h = img.size
    assert w == rec["width"] and h == rec["height"], (
        f"width/height mismatch: image.size=({w},{h}), rec=({rec['width']},{rec['height']}), image={rec['image']}, sent={rec['sent']}"
    )
    gt_bbox = rec["bbox"]

    prompt = PROMPT_TEMPLATE.format(sent=rec["sent"])
    content = [
        {"type": "image_url", "image_url": {"url": to_data_url(path), "detail": "high"}},
        {"type": "text", "text": prompt},
    ]

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            ch = resp.choices[0]

            print("generation", ch.message.content)
            print("gt_bbox", rec["bbox"], "width", w, "height", h)

            m = re.search(
                r'"bbox_2d"\s*:\s*\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]',
                ch.message.content, flags=re.S
            )
            m2 = re.search(
                r'\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]',
                ch.message.content, flags=re.S
            )
            nums = re.findall(r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?', ch.message.content, flags=re.S)

            remap = False
            if m:
                bbox = [float(m.group(i)) for i in range(1, 5)]
                scaled_bbox = scale_bbox(bbox, w, h) if remap else bbox
            elif m2:
                bbox = [float(m2.group(i)) for i in range(1, 5)]
                scaled_bbox = scale_bbox(bbox, w, h) if remap else bbox
            elif len(nums) >= 4:
                bbox = list(map(float, nums[-4:]))
                scaled_bbox = scale_bbox(bbox, w, h) if remap else bbox
            else:
                print(f"bbox_2d not found: {ch.message.content}")
                scaled_bbox = None

            iou = compute_iou(scaled_bbox, gt_bbox) if scaled_bbox is not None else 0.0
            print("iou", iou)

            return {
                "generation": ch.message.content,
                "scaled_bbox": scaled_bbox,
                "iou": iou,
            }
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print("err", err)
            if attempt == 2:
                return {"generation": None, "scaled_bbox": 0, "iou": 0}
            await asyncio.sleep(1.5 * (attempt + 1))

async def bounded_call(idx: int, rec: Dict[str, Any], sem: asyncio.Semaphore,
                       client: AsyncOpenAI, model: str, max_tokens: int) -> (int, Dict[str, Any]):
    async with sem:
        out = await call_one(client, model, rec, max_tokens)
        return idx, out

# 公用：跑一段 [start, end) 的数据并返回三列与均分
async def _run_range(ds, start, end, client, model, max_tokens, sem, desc="async infer"):
    ds_chunk = ds.select(range(start, end))
    tasks = []
    for j in range(len(ds_chunk)):
        rec = {
            "image": ds_chunk[j]["image"],
            "sent": ds_chunk[j]["sent"],
            "bbox": ds_chunk[j]["bbox"],
            "height": ds_chunk[j]["height"],
            "width": ds_chunk[j]["width"],
        }
        tasks.append(asyncio.create_task(bounded_call(start + j, rec, sem, client, model, max_tokens)))

    generations = [None] * len(ds_chunk)
    scaled_bbox = [None] * len(ds_chunk)
    iou = [None] * len(ds_chunk)

    pbar = tqdm(total=len(tasks), desc=f"{desc} [{start}-{end-1}]")
    for coro in asyncio.as_completed(tasks):
        idx, out = await coro
        j = idx - start
        generations[j] = out.get("generation")
        scaled_bbox[j] = out.get("scaled_bbox")
        iou[j] = out.get("iou")
        pbar.update(1)
    pbar.close()

    return ds_chunk, generations, scaled_bbox, iou

async def main_async(args):
    ds = load_dataset("json", split='train', data_files=args.data_json)
    if "sent" not in ds.column_names:
        def extract_sent(row):
            text = " ".join(x.get("value", "") for x in row["conversations"])
            m = re.findall(r'<ref>\s*["“]?(.+?)["”]?\s*</ref>', text, flags=re.S)
            return m[-1].strip() 
        ds = ds.add_column("sent", [extract_sent(row) for row in ds])
        cols = ds.column_names
    if "bbox" not in cols and "modal_bbox" in cols:
        ds = ds.add_column("bbox", ds["modal_bbox"])
        cols = ds.column_names
    for need in ["image", "sent", "bbox", "width", "height"]:
        if need not in ds.column_names:
            raise ValueError(f"缺少字段 '{need}'；现有列：{ds.column_names}")

    if not args.output_dir:
        raise ValueError("--output_dir 必填")
    os.makedirs(args.output_dir, exist_ok=True)

    client = AsyncOpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key="none")
    sem = asyncio.Semaphore(args.concurrency)

    # 分支一：不分块（原先逻辑）
    if args.flush_every is None or args.flush_every <= 0:
        tasks = []
        for i in range(len(ds)):
            rec = {
                "image": ds[i]["image"],
                "sent": ds[i]["sent"],
                "bbox": ds[i]["bbox"],
                "height": ds[i]["height"],
                "width": ds[i]["width"],
            }
            tasks.append(asyncio.create_task(bounded_call(i, rec, sem, client, args.model_path, args.max_tokens)))

        generations = [None] * len(ds)
        scaled_bbox = [None] * len(ds)
        iou = [None] * len(ds)

        pbar = tqdm(total=len(tasks), desc="async infer (no chunk)")
        for coro in asyncio.as_completed(tasks):
            idx, out = await coro
            generations[idx] = out.get("generation")
            scaled_bbox[idx] = out.get("scaled_bbox")
            iou[idx] = out.get("iou")
            pbar.update(1)
        pbar.close()

        stats = summarize_ious(iou)
        mean_iou = stats["mean_iou"]
        pass_rate = stats["pass_rate_05"]
        print("mean_iou", mean_iou)
        print("pass_rate@0.5", pass_rate)

        detail = {}
        for idx in range(len(ds)):
            v = 0.0 if iou[idx] is None else float(iou[idx])
            detail[idx] = {"v": v, "hash": hash(ds[idx]["sent"])}

        # 写 iou_detail.json
        detail_path = os.path.join(args.output_dir, "iou_detail.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)


        iou_file = os.path.join(args.output_dir, "iou.txt")
        with open(iou_file, "w", encoding="utf-8") as f:
            f.write(f"# IOU scores (N={len(iou)})\n")
            f.write(f"mean_iou: {mean_iou:.6f}\n")
            f.write(f"pass_rate@0.5: {pass_rate:.6f}\n")

        ds_out = (ds.add_column("generated_result", generations)
                    .add_column("scaled_bbox", scaled_bbox)
                    .add_column("iou", iou))
        ds_out.save_to_disk(args.output_dir)
        print(f"✅ 完成（不分块）。输出保存到: {args.output_dir}")
        return

    # 分支二：分块
    shards_dir = os.path.join(args.output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    all_scores = []
    shard_paths = []

    N = len(ds)
    step = args.flush_every

    for start in range(0, N, step):
        end = min(start + step, N)
        shard_path = os.path.join(shards_dir, f"shard_{start}_{end}")

        # 断点续跑：已有则跳过
        if os.path.exists(shard_path):
            print(f"🟡 已存在 {shard_path}，跳过计算此分片。")
            shard_paths.append(shard_path)
            # 读取分片分数（可选）
            try:
                with open(os.path.join(shard_path, "iou.txt"), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("avg_score:"):
                            all_scores.append(float(line.strip().split(":")[1]))
            except Exception:
                pass
            continue

        ds_chunk, generations, scaled_bbox, iou= await _run_range(
            ds, start, end, client, args.model_path, args.max_tokens, sem, desc="async infer (chunk)"
        )
        stats = summarize_ious(iou)
        mean_iou = stats["mean_iou"]
        pass_rate = stats["pass_rate_05"]
        shard_detail = {}
        for j, val in enumerate(iou):
            gid = start + j
            v = 0.0 if val is None else float(val)
            shard_detail[gid] = {"v": v, "hash": sent_hash(ds_chunk[j]["sent"])}

        with open(os.path.join(shard_path, "iou_detail.json"), "w", encoding="utf-8") as f:
            json.dump(shard_detail, f, ensure_ascii=False, indent=2)

        ds_chunk_out = (ds_chunk
                        .add_column("generated_result", generations)
                        .add_column("scaled_bbox", scaled_bbox)
                        .add_column("iou", iou))
        ds_chunk_out.save_to_disk(shard_path)
        with open(os.path.join(shard_path, "iou.txt"), "w", encoding="utf-8") as f:
            f.write(f"# IOU scores for shard [{start}-{end-1}] (N={len(iou)})\n")
            f.write(f"mean_iou: {mean_iou:.6f}\n")
            f.write(f"pass_rate@0.5: {pass_rate:.6f}\n")
        print(f"✅ 分片完成并保存：{shard_path}")
        shard_paths.append(shard_path)

    # 合并所有分片
    shard_datasets = [Dataset.load_from_disk(p) for p in shard_paths]
    ds_merged = concatenate_datasets(shard_datasets)

    ious = ds_merged["iou"]
    iou_above_05 = [x for x in ious if (x is not None and x > 0.5)]
    overall_score = (len(iou_above_05) / len(ious)) if len(ious) > 0 else 0.0

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    ds_merged.save_to_disk(final_dir)
    with open(os.path.join(final_dir, "iou.txt"), "w", encoding="utf-8") as f:
        f.write(f"avg_score: {overall_score}\n")

    print(f"🎉 全部分片已合并到：{final_dir}")
    print(f"overall avg_score: {overall_score}")

def parse_args():
    args = argparse.ArgumentParser("Async concurrent VLM inference via SGLang")
    args.add_argument("--data_json", required=True, help="COCO2014 train 风格 JSON 文件路径")
    args.add_argument("--endpoint", default="http://127.0.0.1:30000")
    args.add_argument("--model_path", required=True, help="SGLang 服务端加载的模型名/路径")
    args.add_argument("--output_dir", required=True)
    args.add_argument("--max_tokens", type=int, default=2048)
    args.add_argument("--concurrency", type=int, default=512, help="并发请求数量上限")
    args.add_argument("--flush_every", type=int, default=0, help="分块大小。0 或负数表示不分块（原始逻辑）")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
