from .grits import BeautifulSoup, calculate_grits
import numpy as np

def gini(x):
    if len(x) == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        x = x - x.min()  # 平移到非负
    mean = x.mean()
    if mean == 0:
        return 0.0       # 全零序列
    diff_sum = np.abs(x[:, None] - x).sum()
    return diff_sum / (2 * len(x) ** 2 * mean)


def table_shape(html: str):
    """返回 (总行数, [各行实际列数])；列数已按 colspan 展开"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return 0, []

    rows = table.find_all("tr")
    row_count = len(rows)

    col_counts = []
    for tr in rows:
        cols = tr.find_all(["td", "th"])          # 既处理数据单元，也处理表头
        expanded_len = sum(
            int(c.get("colspan", 1))             # 没写 colspan 时按 1 计算
            for c in cols
        )
        col_counts.append(expanded_len)

    return row_count, col_counts


def calculate_shape_reward(pred, gt):
    pred_row_count, pred_col_counts = table_shape(pred)
    gt_row_count, _ = table_shape(gt)

    gini_score = gini(pred_col_counts)
    return {
        "shape_row": min(pred_row_count, gt_row_count) / (max(pred_row_count, gt_row_count) + 1e-6),
        "shape_col": (1 - gini_score),
    }


def calculate_grits_reward(pred, gt, is_val=False, **kwargs):
    """
    Calculate GRITS reward score based on topological and content similarity.
    
    Args:
        pred (str): Predicted HTML table string
        gt (str): Ground truth HTML table string
        
    Returns:
        float: Harmonic mean of topological and content similarity scores, 
               or 0 if either score is 0
    """

    pred = pred.rsplit("<|im_start|>", 1)[-1].rsplit("</think>", 1)[-1]

    if not is_val:
        if pred.count('<table') != 1 or gt.count('</table') != 1:
            return {"score": 0, "grits_top": 0, "grits_con": 0, "shape_row": 0, "shape_col": 0}
        pred = '<table' + pred.split('<table', 1)[1].split('</table', 1)[0] + '</table>'

    grits_top, grits_con = calculate_grits(pred, gt)
    if min(grits_top, grits_con) == 0:
        score = 0
    else:
        score = 2 * grits_top * grits_con / (grits_top + grits_con)

    if not is_val:
        try:
            shape_score_dict = calculate_shape_reward(pred, gt)
        except Exception as e:
            shape_score_dict = {"shape_row": 0, "shape_col": 0}
        score = score * 0.8 +shape_score_dict["shape_row"] * 0.1 + shape_score_dict["shape_col"] * 0.1
    else:
        shape_score_dict = {
            "shape_row": 1,
            "shape_col": 1,
        }

    return {
        "score": score,
        "grits_top": grits_top,
        "grits_con": grits_con,
        **shape_score_dict,
    }