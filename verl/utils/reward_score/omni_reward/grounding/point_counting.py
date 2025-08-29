import re
import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore


"""
Point-based counting reward function (single category).

Key differences from bbox version:
- Parse points like: "类别1(x1,y1)1(x2,y2)2(x3,y3)3\n类别1总数量：3"
  or:        "category1(x1,y1)1(x2,y2)2(x3,y3)3\ntotal count for category:13"
- Only one category considered for now.
- Use soft distance-based Hungarian matching instead of IoU.
"""


# Format reward components (sum to 1.0)
LABEL_REWARD = 0.25  #存在 label
POINTS_REWARD = 0.25 #能解析出 points 列表
TOTAL_REWARD = 0.25 #存在 declared_total
CONSIS_REWARD = 0.25 #declared_total 与 points 数量一致


def _extract_points(pred: str) -> List[List[int]]:
    pairs = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\d*", pred)
    return [[int(x), int(y)] for x, y in pairs]


def _extract_total_zh(line: str) -> Tuple[Optional[str], Optional[int]]:
    line = line.strip()
    # Find trailing number
    m_num = re.search(r"(\d+)\s*$", line)
    if not m_num:
        return None, None
    total = int(m_num.group(1))

    # Strictly match the last occurrence of '总数量' from right to left
    idx = line.rfind('总数量')
    if idx != -1:
        label = line[:idx].strip().strip('：').strip(':').strip() or None
        return label, total
    return None, total


def _extract_total_en(line: str) -> Tuple[Optional[str], Optional[int]]:
    s = line.strip()
    # Require the compact keyword after removing whitespace
    compact = re.sub(r"\s+", "", s.lower())
    if "totalcountfor" not in compact:
        return None, None

    try:
        # Extract the part after 'total\s*count\s*for'
        after_for = re.split(r"total\s*count\s*for\s*", s, flags=re.IGNORECASE, maxsplit=1)[1].strip()
    except Exception:
        return None, None

    m_num = re.search(r"(\d+)\s*$", after_for)
    if not m_num:
        return None, None
    total = int(m_num.group(1))
    label = after_for[: m_num.start()].strip().rstrip(':：') or None
    return label, total


def _extract_label_from_prefix(pred: str) -> Optional[str]:
    m = re.search(r"([^\(\)\n]+)\(", pred)
    if not m:
        return None
    prefix = m.group(1)
    # Take the last run of letters/digits/underscore/CJK.
    m2 = re.search(r"([\u4e00-\u9fa5A-Za-z0-9_]+)\s*$", prefix)
    return m2.group(1) if m2 else None


def parse_prediction_points(prediction: str) -> Dict[str, Any]:
    """Normalize and parse prediction string to extract label, points, and declared total.
    Returns dict:
    {
      'label': Optional[str],
      'points': List[List[int]],
      'declared_total': Optional[int]
    }
    """
    text = prediction.strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last_two_line = ''.join(lines[:-2]) if lines else ''

    # Extract points from the whole text
    points = _extract_points(text)

    # Try to extract total (and label) from last line
    label, total = _extract_total_en(last_two_line)
    if total is None:
        label, total = _extract_total_zh(last_two_line)

    # If label still None, try to infer from prefix
    if label is None:
        label = _extract_label_from_prefix(text)

    return {
        'label': label,
        'points': points,
        'declared_total': total,
    }


def format_reward_points(prediction: str) -> float:
    """Compute format reward based on availability of label, points, total, and consistency.
    Reward in [0,1].
    """
    parsed = parse_prediction_points(prediction)
    reward = 0.0
    if parsed['label']:
        reward += LABEL_REWARD
    if isinstance(parsed['points'], list) and len(parsed['points']) >= 0:
        reward += POINTS_REWARD
    if parsed['declared_total'] is not None:
        reward += TOTAL_REWARD
    if parsed['declared_total'] is not None and parsed['declared_total'] == len(parsed['points']):
        reward += CONSIS_REWARD
    return reward


def _check_points_shape(points: Any) -> bool:
    if not isinstance(points, list):
        return False
    for p in points:
        if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p)):
            return False
    return True


def _euclidean(a: List[int], b: List[int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if linear_sum_assignment is not None:
        return linear_sum_assignment(cost)
    # Greedy fallback: iteratively pick smallest cost unique row/col
    n = cost.shape[0]
    used_r = set()
    used_c = set()
    pairs: List[Tuple[int, int]] = []
    flat = [(cost[r, c], r, c) for r in range(n) for c in range(n)]
    flat.sort(key=lambda x: x[0])
    for _, r, c in flat:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
        if len(pairs) == n:
            break
    rows = np.array([p[0] for p in pairs], dtype=int)
    cols = np.array([p[1] for p in pairs], dtype=int)
    return rows, cols


def soft_hungarian_f1_points(prediction: str, groundtruth: Dict[str, Any], tolerance: int = 20) -> Dict[str, Any]:
    """Soft distance-based matching F1 using Hungarian algorithm.

    cost = clip(dist / tolerance, 0, 1), similarity s = 1 - cost in [0,1].
    precision = sum(s) / pred_num, recall = sum(s) / gt_num, f1 = harmonic mean.
    """
    parsed = parse_prediction_points(prediction)
    pred_points = parsed['points'] if _check_points_shape(parsed['points']) else []

    # Single category groundtruth
    gt_points_list = groundtruth.get('points', [[]])
    gt_points = gt_points_list[0] if gt_points_list else []

    gt_num = len(gt_points)
    pred_num = len(pred_points)

    # Count correctness (single category)
    count_correct = 1 if gt_num == pred_num else 0

    if gt_num == 0 and pred_num == 0:
        return {
            'f1': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'matched_num': 1.0,
            'gt_num': 0,
            'pred_num': 0,
            'count_correct': count_correct,
        }

    if gt_num == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'matched_num': 0.0,
            'gt_num': 0,
            'pred_num': pred_num,
            'count_correct': count_correct,
        }

    if pred_num == 0:
        return {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'matched_num': 0.0,
            'gt_num': gt_num,
            'pred_num': 0,
            'count_correct': count_correct,
        }

    n = max(gt_num, pred_num)
    cost = np.ones((n, n), dtype=float)
    for i in range(pred_num):
        for j in range(gt_num):
            d = _euclidean(pred_points[i], gt_points[j])
            c = min(d / float(tolerance), 1.0)
            cost[i, j] = c

    rows, cols = _hungarian(cost)

    sum_similarity = 0.0
    for r, c in zip(rows, cols):
        if r < pred_num and c < gt_num:
            s = 1.0 - cost[r, c]  # similarity in [0,1]
            sum_similarity += s

    precision = (sum_similarity / pred_num) if pred_num > 0 else 0.0
    recall = (sum_similarity / gt_num) if gt_num > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'matched_num': float(sum_similarity),
        'gt_num': gt_num,
        'pred_num': pred_num,
        'count_correct': count_correct,
    }


def compute_score(prediction: str, groundtruth: Dict[str, Any], tolerance: int = 20, **kwargs) -> Dict[str, Any]:
    """Overall score combining format reward and soft F1 (single category)."""
    fmt = format_reward_points(prediction)

    parsed = parse_prediction_points(prediction)
    parse_ok = _check_points_shape(parsed['points'])

    if parse_ok:
        metrics = soft_hungarian_f1_points(prediction, groundtruth, tolerance=tolerance)
    else:
        metrics = {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'matched_num': 0.0,
            'gt_num': 0,
            'pred_num': 0,
            'count_correct': 0,
        }

    final = 0.2 * fmt
    if fmt > 0.8 and parse_ok:
        final += (0.1 + 0.7 * metrics['f1'])

    return {
        'score': final,
        'format': fmt,
        'label': parsed['label'],
        'declared_total': parsed['declared_total'],
        **metrics,
    }
