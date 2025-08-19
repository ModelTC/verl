import re
import ast
from func_timeout import func_set_timeout, FunctionTimedOut
from verl.utils.reward_score.utils import ErrCode

INCORRECT_SCORE = 0.
CORRECT_SCORE = 1.

# bounding box format: <box>[[x1,y1,x2,y2]]</box>
def extract_box(model_response):
    bbox_pattern = r'<box>(.*?)</box>'
    bbox_matches = re.findall(bbox_pattern, model_response, re.DOTALL)
    if bbox_matches:
        # get the last match
        content = bbox_matches[-1].strip()
        try:
            # convert string to list
            answer = ast.literal_eval(content)[0]
            if len(answer) == 4:
                return answer
            else:
                return None
        except:
            return None
    return None

def compute_IoU(box1, box2):
    # compute intersect
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    
    # compute union
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    # compute IoU
    iou = inter / union
    return iou

@func_set_timeout(30)
def _compute_score(model_response, ground_truths, eos_token):

    model_answer = extract_box(model_response)
    if model_answer is None:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}

    ground_truths = ground_truths["absolute_coordinates"]
    if ground_truths is None:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.GROUND_TRUTH_ERR}  
    
    iou_score = compute_IoU(model_answer, ground_truths)
    if iou_score > 0:
        return {"score": iou_score, "error_code": ErrCode.SUCCESS}
    else:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.WRONG_ANSWER}


def compute_score(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}
    try:
        return _compute_score(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.WRONG_ANSWER}
    

    


    