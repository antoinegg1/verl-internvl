import json
import argparse
import sys
import os
import difflib
import re

MATCH_KEYS = ["image", "height", "width", "conversations"]

def norm_image(value, relax=True):
    if not isinstance(value, str):
        return value
    p = value
    p = p.replace("coco_flip/", "coco/")
    for anchor in ("train2014/", "val2014/", "test2014/", "test2015/"):
        i = p.find(anchor)
        if i != -1:
            p = p[i:]
            break
    p = p.lower()
    return p

def _normalize_space(s):
    return " ".join(s.lower().strip().split())

def _extract_box_string(s):
    if not isinstance(s, str):
        return None
    m = re.search(r"<box>\s*\[([^\]]+)\]\s*</box>", s, flags=re.IGNORECASE)
    if not m:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
    if not nums:
        return None
    return ",".join(nums)

def extract_human_and_box(conversations):
    human_norm = None
    box_norm = None
    if isinstance(conversations, list):
        for item in conversations:
            if human_norm is None and isinstance(item, dict) and item.get("from") == "human":
                v = item.get("value")
                if isinstance(v, str):
                    human_norm = _normalize_space(v)
        for item in conversations:
            if isinstance(item, dict):
                v = item.get("value")
                b = _extract_box_string(v)
                if b is not None:
                    box_norm = b
                    break
    return human_norm, box_norm

def build_match_key(obj, relax_image=True):
    image_norm = norm_image(obj.get("image", None), relax=relax_image)
    h = obj.get("height", None)
    w = obj.get("width", None)
    human_norm, box_norm = extract_human_and_box(obj.get("conversations", None))
    return (image_norm, h, w, human_norm, box_norm)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def count_lines(path):
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def render_progress(done, total, width=30, prefix="Progress"):
    if total <= 0:
        total = 1
    pct = int(done * 100 / total)
    filled = int(width * pct / 100)
    bar = "#" * filled + "." * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct:3d}% ({done}/{total})")
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")

def _similarity_human_box(target_key, cand_key):
    th, tb = target_key[3], target_key[4]
    ch, cb = cand_key[3], cand_key[4]
    human_ratio = difflib.SequenceMatcher(None, th or "", ch or "").ratio()
    if (tb or "") == (cb or ""):
        box_ratio = 1.0 if (tb is not None or cb is not None) else 0.0
    else:
        box_ratio = difflib.SequenceMatcher(None, tb or "", cb or "").ratio()
    return 0.7 * human_ratio + 0.3 * box_ratio

def migrate(src_path, tgt_path, out_path, fields, overwrite=False, show_progress=True):
    # 1) 建 src 索引
    index = {}
    bucket_ihw = {}
    bucket_i = {}

    for row in load_jsonl(src_path):
        k = build_match_key(row, relax_image=True)
        payload = {}
        for f in fields:
            if f in row:
                payload[f] = row[f]
        if payload:
            index[k] = payload
            ihw = (k[0], k[1], k[2])
            bucket_ihw.setdefault(ihw, []).append((k, payload))
            bucket_i.setdefault((k[0],), []).append((k, payload))

    # 2) 逐行处理 tgt
    total_tgt = count_lines(tgt_path)
    matched_rows = 0
    migrated_rows = 0
    migrated_each = {f: 0 for f in fields}

    processed = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for row in load_jsonl(tgt_path):
            processed += 1
            k = build_match_key(row, relax_image=True)
            if k in index:
                payload = index[k]
                matched_rows += 1
                changed = False
                if overwrite:
                    for f in fields:
                        if f in payload:
                            row[f] = payload[f]
                            migrated_each[f] += 1
                            changed = True
                else:
                    for f in fields:
                        if f not in row and f in payload:
                            row[f] = payload[f]
                            migrated_each[f] += 1
                            changed = True
                if changed:
                    migrated_rows += 1
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                ihw = (k[0], k[1], k[2])
                cand_list = bucket_ihw.get(ihw, [])
                if not cand_list:
                    cand_list = bucket_i.get((k[0],), [])

                best = None
                best_score = -1.0
                for full_k, payload in cand_list:
                    base = 1.0 if (full_k[0], full_k[1], full_k[2]) == ihw else 0.0
                    score = base + _similarity_human_box(k, full_k)
                    if score > best_score:
                        best_score = score
                        best = (full_k, payload)

                report = {
                    "non_hit_target_key": {
                        "image": k[0],
                        "height": k[1],
                        "width": k[2],
                        "human_norm": k[3],
                        "box_norm": k[4],
                    },
                    "nearest_from_src": None,
                    "similarity": round(best_score, 6),
                    "hint": "match uses human text and <box> only; image path relaxed; buckets: (image,height,width)->image"
                }
                if best is not None:
                    nk = best[0]
                    report["nearest_from_src"] = {
                        "image": nk[0],
                        "height": nk[1],
                        "width": nk[2],
                        "human_norm": nk[3],
                        "box_norm": nk[4],
                        "has_fields": {f: (f in best[1]) for f in fields},
                    }
                print(json.dumps(report, ensure_ascii=False, indent=2))
                sys.exit(1)

            if show_progress:
                render_progress(processed, total_tgt, prefix="Merging")

    stats = {
        "tgt_total": total_tgt,
        "matched_rows": matched_rows,
        "migrated_rows": migrated_rows,
        "migrated_each": migrated_each,
        "overwrite": overwrite,
        "fields": fields,
        "src_path": os.path.abspath(src_path),
        "tgt_path": os.path.abspath(tgt_path),
        "out_path": os.path.abspath(out_path),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

# 示例调用
# 想迁移任意一列，比如叫 "iou_detail"
# migrate(src, tgt, merged, fields=["iou_detail"], overwrite=True)
# 或原来的两列：
# migrate(src, tgt, merged, fields=["2b_v5_iou", "2b_v7_2_iou"], overwrite=True)




# 使用示例：
src="/storage/openpsi/data/grounding_sft_v1/8b_internvl_v7_3_50itr_iou_on_refcoco_train_570K/new_vanilla_grounding_refcoco_train_data_8b_v7_3_step_50.jsonl"
tgt="/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1105_2.jsonl"
merged="/storage/openpsi/data/grounding_sft_v1/refcoco-train-refbox-addflip570K_1107.jsonl"
migrate(src, tgt, merged,fields="8b_v7_3_step_50", overwrite=True)
