import math
import re
from typing import Any, List, Sequence, Tuple, Union


Number = Union[int, float]
Box = Tuple[Number, Number, Number, Number]  # (x1, y1, x2, y2)
BOX_RE = re.compile(r"<box>\s*\[(.*?)\]\s*</box>", flags=re.IGNORECASE | re.DOTALL)
INT_RE = re.compile(r"-?\d+")

def extract_box_from_text(s: str) -> List[int]:
    if not isinstance(s, str):
        return []
    # Prefer training format: <box>[x1, y1, x2, y2]</box>
    m = BOX_RE.search(s)
    if m:
        try:
            vals = [int(v.strip()) for v in m.group(1).split(",")]
            if len(vals) >= 4:
                return vals[:4]
        except Exception:
            pass
    # Fallback: first four integers
    nums = INT_RE.findall(s)
    if len(nums) >= 4:
        try:
            return [int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])]
        except Exception:
            return []
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

def to_box_tuple(b: Sequence[Union[int, float]]) -> Box:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return (math.nan, math.nan, math.nan, math.nan)
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

def compute_score(
    solution_str: Any,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs,
):
    """
    Compute IoU between predicted and ground-truth bounding boxes (single example).

    Inputs:
      - ground_truth: list[int] of 4 coords
      - solution_str: str containing the coords (e.g., "<box>[x1,y1,x2,y2]</box>")

    Returns:
      - float IoU for the single example.
    """
    pb = extract_box_from_text(solution_str)
    if len(pb) != 4 or not isinstance(ground_truth, (list, tuple)) or len(ground_truth) != 4:
        return 0.0
    # Use provided scale_bbox (bbox, width, height) if height/width available
    if isinstance(extra_info, dict) and ("height" in extra_info and "width" in extra_info):
        h = extra_info["height"]
        w = extra_info["width"]
        pred_box = scale_bbox(pb, w, h)
        gt_box = scale_bbox(ground_truth, w, h)
    else:
        pred_box = pb
        gt_box = ground_truth
    return float(_iou(to_box_tuple(pred_box), to_box_tuple(gt_box)))
