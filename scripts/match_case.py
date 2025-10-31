#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 B(Arrow) 的 'sent' 与 'iou' 合并到 A(JSONL) 中（仅使用“图片同名 + 文本相似度”，不看 bbox/IoU）。
支持：一个 JSONL + 多个 Arrow（或包含 Arrow 的目录）。
"""
# 用法示例：
'''
python scripts/match_case.py \
--a /storage/openpsi/data/grounding_sft_v1/refcoco_train_v4_gpt_10.26_withflip_2_RL.jsonl \
--b  /storage/openpsi/data/grounding_sft_v1_result/InternVL3_5-241B-A28B/refcoco_train_v4_gpt_10.26_withflip_2_RL_result \
--out /storage/openpsi/data/grounding_sft_v1/refcoco_train_v4_gpt_10.26_withflip_cotv2_RL_1031.jsonl \
--sim_th 0.7 \
--new_key "241B_model_iou"
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
            obj = json.loads(line)
            rows.append(obj)
    return rows

def _resolve_arrow_files(dir_or_file: str) -> List[str]:
    """
    输入一个目录或 .arrow 文件，返回其中的 .arrow 文件列表（HF 常见为 data-*.arrow）。
    """
    if os.path.isdir(dir_or_file):
        candidates = sorted(glob.glob(os.path.join(dir_or_file, "data-*.arrow")))
        if candidates:
            return candidates
        # 兜底：目录下任意 .arrow
        any_arrows = sorted(glob.glob(os.path.join(dir_or_file, "*.arrow")))
        if any_arrows:
            return any_arrows
        raise FileNotFoundError(f"No *.arrow under: {dir_or_file}")
    if dir_or_file.endswith(".arrow") and os.path.exists(dir_or_file):
        return [dir_or_file]
    raise FileNotFoundError(f"Arrow path not found: {dir_or_file}")

def _read_arrow_to_records(arrow_path: str, image_col_candidates=("image","image_path","img","path")) -> List[dict]:
    """
    读取一个 .arrow（IPC stream 或 file 格式），返回记录列表：
        [{"image": "...", "sent": "...", "iou": float}, ...]
    """
    with open(arrow_path, "rb") as f:
        # 优先尝试 IPC stream
        try_stream = True
        # 简单检测：IPC stream 以 b'ARROW1' 魔数开头；file 则是 Feather/IPC file，两者这里统一用容错顺序
        # 为减少分支，这里直接先用 open_stream，失败再用 open_file（不使用 try/except 的要求下，改用标志控制）
    # 由于用户限定“尽量减少分类讨论”，这里直接按 IPC stream 读取，不再自动切换。
    with pa_ipc.open_stream(arrow_path) as reader:
        table = reader.read_all()

    cols = set(table.column_names)

    image_col = None
    for k in image_col_candidates:
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

    out = []
    for img, sent, iou in zip(img_arr, sent_arr, iou_arr):
        if not img or not sent:
            continue
        if isinstance(img, dict):
            img_path = img.get("path") or ""
        else:
            img_path = str(img)
        if not img_path:
            continue
        out.append({
            "image": img_path,
            "sent": str(sent),
            "iou": float(iou),
        })
    return out

def index_b_from_arrows(b_paths: List[str]) -> Dict[str, List[dict]]:
    """
    合并多个 Arrow/目录：
        { image_basename: [ {"image": "...", "sent": "...", "iou": float}, ... ] }
    """
    all_arrow_files = []
    for p in b_paths:
        all_arrow_files.extend(_resolve_arrow_files(p))

    idx: Dict[str, List[dict]] = {}
    for af in all_arrow_files:
        records = _read_arrow_to_records(af)
        for r in records:
            img_bn = basename(r["image"])
            if not img_bn:
                continue
            idx.setdefault(img_bn, []).append(r)
    return idx

def main():
    ap = argparse.ArgumentParser(description="Merge B(Arrow).sent/iou into A(JSONL) by image+text similarity. Supports multiple B paths.")
    ap.add_argument("--a", required=True, help="Path to A JSONL")
    ap.add_argument("--b", required=True, nargs="+", help="One or more Arrow dataset dirs or .arrow files")
    ap.add_argument("--out", required=True, help="Output JSONL (A structure + added fields from B)")
    ap.add_argument("--sim_th", type=float, default=0.7, help="Text similarity threshold [0,1]")
    ap.add_argument("--new_key",required=True,help="New key to add")
    args = ap.parse_args()

    A_rows = load_a_jsonl(args.a)
    B_idx = index_b_from_arrows(args.b)

    matched = 0
    total = 0
    new_key=args.new_key
    with open(args.out, "w", encoding="utf-8") as fout:
        for obj in tqdm(A_rows, total=len(A_rows), desc="Merging", ncols=100):
            total += 1
            img = obj.get("image", "")
            img_bn = basename(img)
            conv = obj.get("conversations")
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
            if best is not None and best_sim >= args.sim_th:
                new_obj[new_key] = best["iou"]
                matched += 1
            else:
                new_obj[new_key] = None

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(f"Processed {total} A rows. Matched {matched} rows (sim_th={args.sim_th}).")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
