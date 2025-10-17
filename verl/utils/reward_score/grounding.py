import json
import math
import re
from typing import Any, Iterable, List, Sequence, Tuple, Union


Number = Union[int, float]
Box = Tuple[Number, Number, Number, Number]  # (x1, y1, x2, y2)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return math.nan


def _normalize_box(b: Sequence[Any]) -> Box:
    if len(b) != 4:
        raise ValueError(f"Box must have 4 elements, got {len(b)}")
    x1, y1, x2, y2 = (_to_float(b[0]), _to_float(b[1]), _to_float(b[2]), _to_float(b[3]))
    # ensure x1<=x2, y1<=y2 when possible
    if not (math.isnan(x1) or math.isnan(x2)) and x1 > x2:
        x1, x2 = x2, x1
    if not (math.isnan(y1) or math.isnan(y2)) and y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _parse_boxes_from_str(s: str) -> List[Box]:
    s = s.strip()
    # Try JSON first
    try:
        obj = json.loads(s)
        return _parse_boxes(obj)
    except Exception:
        pass

    # Fallback: extract numbers and group by 4
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    vals = [float(n) for n in nums]
    if len(vals) < 4:
        return []
    boxes = []
    for i in range(0, len(vals) - 3, 4):
        boxes.append(_normalize_box(vals[i : i + 4]))
    return boxes


def _parse_boxes(data: Any) -> List[Box]:
    # Supported formats:
    # - list[[x1,y1,x2,y2], ...]
    # - dict{"boxes": [[x1,y1,x2,y2], ...]} or list of dicts with a box key
    # - string (JSON or free text with numbers)
    if data is None:
        return []
    if isinstance(data, str):
        return _parse_boxes_from_str(data)
    if isinstance(data, (tuple, list)):
        # single box or list of boxes or list of dicts
        if len(data) == 4 and all(isinstance(v, (int, float, str)) for v in data):
            return [_normalize_box(data)]
        out: List[Box] = []
        for item in data:
            if isinstance(item, (tuple, list)):
                out.append(_normalize_box(item))
            elif isinstance(item, dict):
                if "box" in item:
                    out.append(_normalize_box(item["box"]))
                elif all(k in item for k in ("x1", "y1", "x2", "y2")):
                    out.append(_normalize_box([item["x1"], item["y1"], item["x2"], item["y2"]]))
        return out
    if isinstance(data, dict):
        if "boxes" in data and isinstance(data["boxes"], (list, tuple)):
            return _parse_boxes(data["boxes"])
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            return [_normalize_box([data["x1"], data["y1"], data["x2"], data["y2"]])]
    # unsupported
    return []


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    if any(math.isnan(v) for v in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _greedy_match(preds: List[Box], gts: List[Box]) -> Tuple[int, float, List[float]]:
    """Greedy one-to-one matching by IoU. Returns (matched, best_overall_iou, per_gt_best_iou)."""
    used_pred = set()
    matched = 0
    best_overall = 0.0
    per_gt_best: List[float] = []

    for gt in gts:
        best_iou = 0.0
        best_j = -1
        for j, pr in enumerate(preds):
            if j in used_pred:
                continue
            iou = _iou(pr, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        per_gt_best.append(best_iou)
        best_overall = max(best_overall, best_iou)
        if best_j >= 0:
            used_pred.add(best_j)
            matched += 1
    return matched, best_overall, per_gt_best


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


def compute_score(
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs,
):

    # Allow override via kwargs or extra_info dict
    if isinstance(extra_info, dict) and "iou_thresh" in extra_info:
        iou_thresh = float(extra_info.get("iou_thresh", 0.5))
    else:
        iou_thresh = float(kwargs.get("iou_thresh", 0.5))

    # Try box-based evaluation first
    pred_boxes = _parse_boxes(solution_str)
    gt_boxes = _parse_boxes(ground_truth)

    if pred_boxes or gt_boxes:
        matched, best_overall, per_gt_best = _greedy_match(pred_boxes, gt_boxes)
        # Count matches above threshold
        tp = sum(1 for v in per_gt_best if v >= iou_thresh)
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)
        precision = (tp / n_pred) if n_pred > 0 else 0.0
        recall = (tp / n_gt) if n_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Only expose required keys
        return {"score": float(f1), "iou": float(best_overall)}

    # Text fallback: EM ignoring spaces and case
    try:
        pred = str(solution_str)
        gt = str(ground_truth)
        em = _normalize_text(pred) == _normalize_text(gt)
        # Fallback: provide score and a dummy IoU (0.0 as IoU is not applicable)
        return {"score": 1.0 if em else 0.0, "iou": 0.0}
    except Exception:
        return {"score": 0.0, "iou": 0.0}
