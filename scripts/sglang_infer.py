
import os, base64, argparse, asyncio
from typing import List, Dict, Any
from datasets import load_dataset, DatasetDict
from openai import AsyncOpenAI
from tqdm import tqdm
import re
from PIL import Image
import os
import base64
import numpy as np
PROMPT_TEMPLATE = (
    # "Please provide the bounding box coordinate of the region this sentence describes: <ref>{sent}</ref> " #for InternVL
    "Locate {sent}, output its bbox coordinates using JSON format. " #for Qwen
)


def to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

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
    rec["image"]=rec["image"].replace("lustre/fsw/portfolios/nvr/users/yunhaof/datasets","data").lstrip("/")
    base_image_dir="/storage/openpsi/"
    path=os.path.join(base_image_dir, rec["image"])
    with Image.open(path) as img:
        w, h = img.size 
    assert w == rec["width"] and h == rec["height"], (
        f"width/height mismatch: image.size=({w},{h}), rec=({rec['width']},{rec['height']}), image={rec['image']}, sent={rec['sent']}"
    )

    gt_bbox = rec["bbox"]


    prompt = PROMPT_TEMPLATE.format(sent=rec["sent"])
    # print(prompt)
    content = [
        {"type": "image_url", "image_url": {"url": to_data_url(path), "detail": "high"}},
        {"type": "text", "text": prompt},
    ]
#{"role": "system", "content": "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"},
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            ch = resp.choices[0]
            
            print("generation",ch.message.content)
            print("gt_bbox",rec["bbox"],"width",w,"height",h )
            m = re.search(r'"bbox_2d"\s*:\s*\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]',
                ch.message.content, flags=re.S)
            m2 = re.search(r'\[\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*\]',
                                ch.message.content, flags=re.S)
            nums = re.findall(r'[+-]?(?:\d*\.\d+|\d+)(?:[eE][+-]?\d+)?', ch.message.content, flags=re.S)          
            remap=True
            if m:
                bbox = [float(m.group(i)) for i in range(1, 5)]
                if remap:
                    scaled_bbox =scale_bbox(bbox, w, h)
                else:
                    scaled_bbox = bbox

                iou = compute_iou(scaled_bbox, gt_bbox)
                print("iou", iou)
            elif m2:
                bbox = [float(m2.group(i)) for i in range(1, 5)]
                if remap:
                    scaled_bbox =scale_bbox(bbox, w, h)
                else:
                    scaled_bbox = bbox
                iou = compute_iou(scaled_bbox, gt_bbox)
                print("iou", iou)
            elif len(nums) >= 4:
                bbox = list(map(float, nums[-4:]))  # 最后四个独立数字
                if remap:
                    scaled_bbox =scale_bbox(bbox, w, h)
                else:
                    scaled_bbox = bbox
                iou = compute_iou(scaled_bbox, gt_bbox)
                print("iou", iou)
            else:
                print(f"bbox_2d not found: {ch.message.content}")
                scaled_bbox = None
                iou = 0

            

            return {
                "generation": ch.message.content,
                "scaled_bbox": scaled_bbox,
                "iou": iou,
            }
        except Exception as e:
            
            err = f"{type(e).__name__}: {e}"
            print("err",err)
            if attempt == 2:
                return {"generation": None,"scaled_bbox": 0, "iou": 0}
            await asyncio.sleep(1.5 * (attempt + 1))

async def bounded_call(idx: int, rec: Dict[str, Any], sem: asyncio.Semaphore,
                       client: AsyncOpenAI, model: str, max_tokens: int) -> (int, Dict[str, Any]):
    async with sem:
        out = await call_one(client, model, rec, max_tokens)
        return idx, out

async def main_async(args):
    ds = load_dataset("json",split='train', data_files=args.data_json)
    for need in ["image", "sent", "bbox"]:
        if need not in ds.column_names:
            raise ValueError(f"缺少字段 '{need}'；现有列：{ds.column_names}")

    client = AsyncOpenAI(base_url=args.endpoint.rstrip("/") + "/v1", api_key="none")
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []
    for i in range(len(ds)):
        rec = {"image": ds[i]["image"], "sent": ds[i]["sent"], "bbox": ds[i]["bbox"], "height": ds[i]["height"],"width": ds[i]["width"]}
        tasks.append(asyncio.create_task(bounded_call(i, rec, sem, client, args.model_path, args.max_tokens)))


    generations = [None] * len(ds)
    scaled_bbox = [None] * len(ds)
    iou = [None] * len(ds)

    pbar = tqdm(total=len(tasks), desc="async infer")
    for coro in asyncio.as_completed(tasks):
        idx, out = await coro
        generations[idx] = out.get("generation")
        scaled_bbox[idx] = out.get("scaled_bbox")
        iou[idx] = out.get("iou")
        pbar.update(1)
    pbar.close()
    iou_above_05 = [score for score in iou if score > 0.5]

    # 计算得分比例（大于0.5的个数除以总数）
    score = len(iou_above_05) / len(iou) if len(iou) > 0 else 0
    print("avg_score", score)

    # 保存到文件
    iou_file = os.path.join(args.output_dir, "iou.txt")
    with open(iou_file, "a", encoding="utf-8") as f:
        f.write(f"avg_score: {score}\n")

    ds_out = (ds.add_column("generated_result", generations)
        .add_column("scaled_bbox", scaled_bbox)
        .add_column("iou", iou))
    
    ds_out.save_to_disk(args.output_dir)
    print(f"✅ 完成。输出保存到: {args.output_dir}")

def parse_args():
    args = argparse.ArgumentParser("Async concurrent VLM inference via SGLang")
    args.add_argument("--data_json", required=True, help="COCO2014 train 风格 JSON 文件路径")
    args.add_argument("--endpoint", default="http://127.0.0.1:30000")
    args.add_argument("--model_path", required=True, help="SGLang 服务端加载的模型名/路径")
    args.add_argument("--output_dir")
    args.add_argument("--max_tokens", type=int, default=256)
    args.add_argument("--concurrency", type=int, default=256, help="并发请求数量上限")
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main_async(args))
