import itertools
import numpy as np
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from verl.utils.reward_score.utils import ErrCode

INCORRECT_SCORE = 0.
CORRECT_SCORE = 1.

def html_table_to_grid(html: str):
    """
    将 <table>...</table> 字符串解析成二维列表（list[list[str]]）。
    空白单元格会填 ''；跨行/跨列的单元格会在覆盖到的所有位置填同一文本。

    Parameters
    ----------
    html : str
        含有单个 <table> 的 HTML 字符串

    Returns
    -------
    list[list[str]]
        表格网格，等价于 cells_to_grid 的输出格式
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return np.array([[]]), np.array([[]])

    grid_text = []                           # 最终二维列表
    grid_topo = []
    rows = table.find_all("tr")         # 行遍历
    if len(rows) == 0:
        return np.array([[]]), np.array([[]])

    for r_idx, tr in enumerate(rows):
        # 确保当前行存在
        while len(grid_text) <= r_idx:
            grid_text.append([])
            grid_topo.append([])

        col_ptr = 0                     # 当前行的列指针
        cells = tr.find_all(["th", "td"])

        for cell in cells:
            # 跳过由于 rowspan/colspan 已经占用的位置
            while col_ptr < len(grid_text[r_idx]) and grid_text[r_idx][col_ptr] is not None:
                col_ptr += 1

            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            text = cell.get_text(strip=True) or ""     # 取纯文本，空白填 ''

            # 把文本写入 span 覆盖到的所有格子
            for dr in range(rowspan):
                while len(grid_text) <= r_idx + dr:
                    grid_text.append([])
                    grid_topo.append([])
                for dc in range(colspan):
                    target_row_text = grid_text[r_idx + dr]
                    target_row_topo = grid_topo[r_idx + dr]
                    while len(target_row_text) <= col_ptr + dc:
                        target_row_text.append(None)
                        target_row_topo.append(None)
                    if target_row_text[col_ptr + dc] is None or (target_row_text[col_ptr + dc] == "" and text != "" and dc > 0):
                        target_row_text[col_ptr + dc] = text
                        target_row_topo[col_ptr + dc] = [-dc, -dr, colspan-dc, rowspan-dr]
            col_ptr += colspan

    max_length = max(len(row) for row in grid_text)

    # 把仍为 None 的空位补成 ''
    for row in grid_text:
        if len(row) < max_length:
            row.extend([None] * (max_length - len(row)))
        for i, val in enumerate(row):
            if val is None:
                row[i] = ""

    for row in grid_topo:
        if len(row) < max_length:
            row.extend([None] * (max_length - len(row)))
        for i, val in enumerate(row):
            if val is None:
                row[i] = [0, 0, 1, 1]

    return np.array(grid_text), np.array(grid_topo)

def compute_fscore(num_true_positives, num_true, num_positives):
    """
    Compute the f-score or f-measure for a collection of predictions.

    Conventions:
    - precision is 1 when there are no predicted instances
    - recall is 1 when there are no true instances
    - fscore is 0 when recall or precision is 0
    """
    if num_positives > 0:
        precision = num_true_positives / num_positives
    else:
        precision = 1
    if num_true > 0:
        recall = num_true_positives / num_true
    else:
        recall = 1
        
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0

    return fscore, precision, recall 

def initialize_DP(sequence1_length, sequence2_length):
    """
    Helper function to initialize dynamic programming data structures.
    """
    # Initialize DP tables
    scores = np.zeros((sequence1_length + 1, sequence2_length + 1))
    pointers = np.zeros((sequence1_length + 1, sequence2_length + 1))
    
    # Initialize pointers in DP table
    for seq1_idx in range(1, sequence1_length + 1):
        pointers[seq1_idx, 0] = -1
        
    # Initialize pointers in DP table
    for seq2_idx in range(1, sequence2_length + 1):
        pointers[0, seq2_idx] = 1

    return scores, pointers

def traceback(pointers):
    """
    Dynamic programming traceback to determine the aligned indices
    between the two sequences.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    """
    seq1_idx = pointers.shape[0] - 1
    seq2_idx = pointers.shape[1] - 1
    aligned_sequence1_indices = []
    aligned_sequence2_indices = []
    while not (seq1_idx == 0 and seq2_idx == 0):
        if pointers[seq1_idx, seq2_idx] == -1:
            seq1_idx -= 1
        elif pointers[seq1_idx, seq2_idx] == 1:
            seq2_idx -= 1
        else:
            seq1_idx -= 1
            seq2_idx -= 1
            aligned_sequence1_indices.append(seq1_idx)
            aligned_sequence2_indices.append(seq2_idx)
            
    aligned_sequence1_indices = aligned_sequence1_indices[::-1]
    aligned_sequence2_indices = aligned_sequence2_indices[::-1]

    return aligned_sequence1_indices, aligned_sequence2_indices

def align_1d(sequence1, sequence2, reward_lookup, return_alignment=False):
    '''
    Dynamic programming alignment between two sequences,
    with memoized rewards.

    Sequences are represented as indices into the rewards lookup table.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    sequence1_length = len(sequence1)
    sequence2_length = len(sequence2)

    scores, pointers = initialize_DP(sequence1_length,
                                     sequence2_length)
        
    for seq1_idx in range(1, sequence1_length+1):
        for seq2_idx in range(1, sequence2_length+1):
            reward = reward_lookup[sequence1[seq1_idx-1] + sequence2[seq2_idx-1]]
            diag_score = scores[seq1_idx-1, seq2_idx-1] + reward
            skip_seq2_score = scores[seq1_idx, seq2_idx-1]
            skip_seq1_score = scores[seq1_idx-1, seq2_idx]
               
            max_score = max(diag_score, skip_seq1_score, skip_seq2_score)
            scores[seq1_idx, seq2_idx] = max_score
            if diag_score == max_score:
                pointers[seq1_idx, seq2_idx] = 0
            elif skip_seq1_score == max_score:
                pointers[seq1_idx, seq2_idx] = -1
            else: # skip_seq2_score == max_score
                pointers[seq1_idx, seq2_idx] = 1
    
    score = scores[-1, -1]
    
    if not return_alignment:
        return score
    
    # Traceback
    sequence1_indices, sequence2_indices = traceback(pointers)
    
    return sequence1_indices, sequence2_indices, score

def align_2d_outer(true_shape, pred_shape, reward_lookup):
    '''
    Dynamic programming matrix alignment posed as 2D
    sequence-of-sequences alignment:
    Align two outer sequences whose entries are also sequences,
    where the match reward between the inner sequence entries
    is their 1D sequence alignment score.

    Traceback convention: -1 = up, 1 = left, 0 = diag up-left
    '''
    
    scores, pointers = initialize_DP(true_shape[0], pred_shape[0])
        
    for row_idx in range(1, true_shape[0] + 1):
        for col_idx in range(1, pred_shape[0] + 1):
            reward = align_1d([(row_idx-1, tcol) for tcol in range(true_shape[1])],
                              [(col_idx-1, prow) for prow in range(pred_shape[1])],
                              reward_lookup)
            diag_score = scores[row_idx - 1, col_idx - 1] + reward
            same_row_score = scores[row_idx, col_idx - 1]
            same_col_score = scores[row_idx - 1, col_idx]
               
            max_score = max(diag_score, same_col_score, same_row_score)
            scores[row_idx, col_idx] = max_score
            if diag_score == max_score:
                pointers[row_idx, col_idx] = 0
            elif same_col_score == max_score:
                pointers[row_idx, col_idx] = -1
            else:
                pointers[row_idx, col_idx] = 1
    
    score = scores[-1, -1]
            
    aligned_true_indices, aligned_pred_indices = traceback(pointers)
    
    return aligned_true_indices, aligned_pred_indices, score


import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

# 把计算 reward 的逻辑提取到模块顶层函数，确保可 picklable
def _compute_reward_iou(params):
    trow, tcol, prow, pcol = params
    r = iou(true_cell_grid[trow, tcol],
                        pred_cell_grid[prow, pcol])
    return trow, tcol, prow, pcol, r

def _compute_reward_lcs(params):
    trow, tcol, prow, pcol = params
    r = lcs_similarity(true_cell_grid[trow, tcol],
                        pred_cell_grid[prow, pcol])
    return trow, tcol, prow, pcol, r

def build_reward_dicts(true_cell_grid, pred_cell_grid, reward_function, max_workers=None):
    # 参数生成器
    coords = itertools.product(
        range(true_cell_grid.shape[0]),
        range(true_cell_grid.shape[1]),
        range(pred_cell_grid.shape[0]),
        range(pred_cell_grid.shape[1])
    )

    pre_computed_rewards = {}
    transpose_rewards = {}

    if reward_function == iou:
        compute_reward = _compute_reward_iou
    elif reward_function == lcs_similarity:
        compute_reward = _compute_reward_lcs

    # 并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 也可以，但 as_completed 更灵活，能配合进度条
        futures = { executor.submit(compute_reward, p): p for p in coords }
        for future in as_completed(futures):
            trow, tcol, prow, pcol, r = future.result()
            pre_computed_rewards[(trow, tcol, prow, pcol)] = r
            transpose_rewards[(tcol, trow, pcol, prow)] = r

    return pre_computed_rewards, transpose_rewards


def factored_2dmss(true_cell_grid, pred_cell_grid, reward_function):
    """
    Factored 2D-MSS: Factored two-dimensional most-similar substructures

    This is a polynomial-time heuristic to computing the 2D-MSS of two matrices,
    which is NP hard.

    A substructure of a matrix is a subset of its rows and its columns.

    The most similar substructures of two matrices, A and B, are the substructures
    A' and B', where the sum of the similarity over all corresponding entries
    A'(i, j) and B'(i, j) is greatest.
    """
    pre_computed_rewards = {}
    transpose_rewards = {}
    for trow, tcol, prow, pcol in itertools.product(range(true_cell_grid.shape[0]),
                                                    range(true_cell_grid.shape[1]),
                                                    range(pred_cell_grid.shape[0]),
                                                    range(pred_cell_grid.shape[1])):

        reward = reward_function(true_cell_grid[trow, tcol], pred_cell_grid[prow, pcol])

        pre_computed_rewards[(trow, tcol, prow, pcol)] = reward
        transpose_rewards[(tcol, trow, pcol, prow)] = reward

    num_pos = pred_cell_grid.shape[0] * pred_cell_grid.shape[1]
    num_true = true_cell_grid.shape[0] * true_cell_grid.shape[1]

    true_row_nums, pred_row_nums, row_pos_match_score = align_2d_outer(true_cell_grid.shape[:2],
                                                                pred_cell_grid.shape[:2],
                                                                pre_computed_rewards)

    true_column_nums, pred_column_nums, col_pos_match_score = align_2d_outer(true_cell_grid.shape[:2][::-1],
                                                                         pred_cell_grid.shape[:2][::-1],
                                                                         transpose_rewards)

    pos_match_score_upper_bound =  min(row_pos_match_score, col_pos_match_score)
    upper_bound_score, _, _ = compute_fscore(pos_match_score_upper_bound, num_pos, num_true)

    positive_match_score = 0
    for true_row_num, pred_row_num in zip(true_row_nums, pred_row_nums):
        for true_column_num, pred_column_num in zip(true_column_nums, pred_column_nums):
            positive_match_score += pre_computed_rewards[(true_row_num, true_column_num, pred_row_num, pred_column_num)]

    fscore, precision, recall = compute_fscore(positive_match_score,
                                               num_true,
                                               num_pos)
    
    return fscore, precision, recall, upper_bound_score

IGNORE_SNIPPETS = [' ', '\n', '<br>', '#']

def lcs_similarity(pred_string, gt_string):
    for snippet in IGNORE_SNIPPETS:
        if snippet in pred_string:
            pred_string = pred_string.replace(snippet, '')
        if snippet in gt_string:
            gt_string = gt_string.replace(snippet, '')

    if len(pred_string) == 0 and len(gt_string) == 0:
        return 1
    s = SequenceMatcher(None, pred_string, gt_string)
    lcs = ''.join([pred_string[block.a:(block.a + block.size)] for block in s.get_matching_blocks()])
    return 2*len(lcs)/(len(pred_string)+len(gt_string))

def bbox_area(bbox):
    """计算单个 bbox 的面积（xyxy 格式）。"""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersect_area(b1, b2):
    """
    计算两个 bbox 的交集面积（xyxy）。
    返回值：float
    """
    x_left   = max(b1[0], b2[0])
    y_top    = max(b1[1], b2[1])
    x_right  = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    return inter_w * inter_h

def union_area(b1, b2):
    """
    计算两个 bbox 的并集面积（xyxy）。
    并集 = 面积1 + 面积2 − 交集面积
    """
    return bbox_area(b1) + bbox_area(b2) - intersect_area(b1, b2)

def iou(bbox1, bbox2):
    """
    Compute the intersection-over-union of two bounding boxes.
    """
    intersection = intersect_area(bbox1, bbox2)
    union = union_area(bbox1, bbox2)
    
    if union > 0:
        return intersection / union
    
    return 0


def calculate_grits(pred, gt):
    """
    Calculate GRITS (Grid Table Similarity) metrics for table structure and content.

    Args:
        pred (str): Predicted HTML table string
        gt (str): Ground truth HTML table string

    Returns:
        tuple: A tuple containing:
            - grits_top (float): Topological similarity score (0-1)
            - grits_con (float): Content similarity score (0-1)
            
    The function converts HTML tables to grid representations, then calculates:
    1. Topological similarity using IoU of table structures
    2. Content similarity using LCS (Longest Common Subsequence) of table text
    If conversion fails for predicted table, returns (0, 0).
    """
    try:
        pred_grid_text, pred_grid_topo = html_table_to_grid(pred)
    except Exception as e:
        print(e)
        return 0, 0
    gt_grid_text, gt_grid_topo = html_table_to_grid(gt)
    h, w = gt_grid_text.shape
    pred_grid_text = pred_grid_text[:h * 2, :w * 2]
    pred_grid_topo = pred_grid_topo[:h * 2, :w * 2]

    grits_con = factored_2dmss(gt_grid_text, pred_grid_text, lcs_similarity)[0] 
    grits_top = factored_2dmss(gt_grid_topo, pred_grid_topo, iou)[0] 

    return grits_top, grits_con


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
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not pred.rstrip().endswith(eos_token):
            return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}
            
    pred = pred.rsplit("<|im_start|>", 1)[-1].rsplit("</think>", 1)[-1]

    if not is_val:
        if pred.count('<table') != 1 or gt.count('</table') != 1:
            return {"score": 0, "grits_top": 0, "grits_con": 0}
        pred = '<table' + pred.split('<table', 1)[1].split('</table', 1)[0] + '</table>'

    grits_top, grits_con = calculate_grits(pred, gt)
    if min(grits_top, grits_con) == 0:
        score = 0
    else:
        score = 2 * grits_top * grits_con / (grits_top + grits_con)

    return {
        "score": score,
        "grits_top": grits_top,
        "grits_con": grits_con,
    }
