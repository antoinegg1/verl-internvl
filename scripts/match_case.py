#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 B(Arrow) 的 'sent' 与 'iou' 合并到 A(JSONL) 中（仅使用“图片同名 + 文本相似度”，不看 bbox/IoU）。
"""
'''
python scripts/match_case.py \
--a /storage/openpsi/data/grounding_sft_v1/stage2_thinking_with_text_sft_train_10_24_3.jsonl \
--b /storage/openpsi/data/grounding_sft_v1_result/internvl3_5_8b_v7_2_8B/refcoco_trainv4_result/data-00000-of-00001.arrow \
--out /storage/openpsi/data/grounding_sft_v1/stage2_thinking_with_text_sft_train_10_26_1.jsonl \
--sim_th 0.7
'''
import argparse
import json
import os
import re

import sys
from tqdm import tqdm
import glob
from difflib import SequenceMatcher
from typing import List, Dict, Optional

import pyarrow as pa
import pyarrow.ipc as pa_ipc

REF_RE = re.compile(r"<ref>(.*?)</ref>", re.IGNORECASE | re.DOTALL)

def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`,.;:!?()\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()

def basename(p: str) -> str:
    return os.path.basename(p)

def extract_ref_from_conversations(conv: List[dict]) -> Optional[str]:
    for c in conv:
        if c.get("from") == "human":
            m = REF_RE.search(c.get("value", ""))
            if m:
                return m.group(1).strip()
    joined = " ".join(c.get("value", "") for c in conv)
    m = REF_RE.search(joined)
    return m.group(1).strip() if m else None

def load_a_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[A] JSON parse error at line {ln}: {e}", file=sys.stderr)
                continue
            rows.append(obj)
    return rows

def index_b_from_arrow(dir_or_file: str) -> Dict[str, List[dict]]:
    """
    读取 HuggingFace 风格 Arrow 数据集（目录或 .arrow 文件），构建：
        { image_basename: [ {"image": "...", "sent": "...", "iou": float} , ... ] }
    """
    arrow_path = dir_or_file
    if os.path.isdir(dir_or_file):
        candidates = sorted(glob.glob(os.path.join(dir_or_file, "data-*.arrow")))
        if not candidates:
            raise FileNotFoundError(f"No data-*.arrow under: {dir_or_file}")
        arrow_path = candidates[0]
    elif not (dir_or_file.endswith(".arrow") and os.path.exists(dir_or_file)):
        raise FileNotFoundError(f"Arrow path not found: {dir_or_file}")

    # HF 常见为 IPC stream
    with pa_ipc.open_stream(arrow_path) as reader:
        table = reader.read_all()

    cols = set(table.column_names)

    # image / sent / iou 列名（如实际不同，请改这里）
    image_col = None
    for k in ["image", "image_path", "img", "path"]:
        if k in cols:
            image_col = k
            break
    if image_col is None:
        raise KeyError(f"Missing image column. Got {sorted(cols)}")

    sent_col = "sent" if "sent" in cols else ("sentence" if "sentence" in cols else None)
    if sent_col is None:
        raise KeyError(f"Missing 'sent'/'sentence' column. Got {sorted(cols)}")

    iou_col = "iou"
    if iou_col not in cols:
        raise KeyError(f"Missing 'iou' column. Got {sorted(cols)}")

    img_arr = table[image_col].to_pylist()
    sent_arr = table[sent_col].to_pylist()
    iou_arr = table[iou_col].to_pylist()

    idx: Dict[str, List[dict]] = {}
    for img, sent, iou in zip(img_arr, sent_arr, iou_arr):
        if not img or not sent:
            continue
        # HF Image 类型可能是 dict
        if isinstance(img, dict):
            img_path = img.get("path") or ""
        else:
            img_path = str(img)
        img_bn = basename(img_path)
        if not img_bn:
            continue
        idx.setdefault(img_bn, []).append({
            "image": img_path,
            "sent": str(sent),
            "iou": float(iou),
        })
    return idx

def main():
    ap = argparse.ArgumentParser(description="Merge B(Arrow).sent/iou into A(JSONL) by image+text similarity.")
    ap.add_argument("--a", required=True, help="Path to A JSONL")
    ap.add_argument("--b", required=True, help="Path to B Arrow dataset directory (or .arrow file)")
    ap.add_argument("--out", required=True, help="Output JSONL (A structure + added fields from B)")
    ap.add_argument("--sim_th", type=float, default=0.7, help="Text similarity threshold [0,1]")
    args = ap.parse_args()

    A_rows = load_a_jsonl(args.a)
    B_idx = index_b_from_arrow(args.b)

    matched = 0
    total = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for obj in tqdm(A_rows, total=len(A_rows), desc="Merging", ncols=100):
            total += 1
            img = obj["image"]
            img_bn = basename(img)
            conv = obj["conversations"]
            ref_a = extract_ref_from_conversations(conv) if isinstance(conv, list) else None

            best = None
            best_sim = -1.0
            if img_bn and ref_a and img_bn in B_idx:
                for cand in B_idx[img_bn]:
                    sent_b = cand["sent"]
                    if not sent_b:
                        continue
                    s = text_sim(ref_a, sent_b)
                    if s > best_sim:
                        best_sim = s
                        best = cand

            new_obj = dict(obj)
            new_obj["8b_v7_2_model_iou"] = best["iou"]
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            matched += 1

    print(f"Processed {total} A rows. Matched {matched} rows (sim_th={args.sim_th}).")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
