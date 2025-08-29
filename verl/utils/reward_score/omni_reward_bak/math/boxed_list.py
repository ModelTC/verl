import re
from func_timeout import func_set_timeout, FunctionTimedOut
from .utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from verl.utils.reward_score.utils import ErrCode

INCORRECT_SCORE = 0.
CORRECT_SCORE = 1.

@func_set_timeout(30)
def _compute_score(model_response, ground_truths, eos_token):
    if not model_response.rstrip().endswith(eos_token):
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}
    
    model_answers = re.findall(r"\\boxed\{(.*?)\}", model_response, re.DOTALL)
    if len(model_answers) == 0:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}

    # Process the ground truth(s)
    if ground_truths is None:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.GROUND_TRUTH_ERR}
    
    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.GROUND_TRUTH_ERR}

    # Check against all possible correct answers
    num_correct = 0
    for ground_truth in processed_ground_truths:
        is_correct = False
        for model_answer in model_answers:
            is_correct = is_correct or grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)

        num_correct += int(is_correct)
    
    if num_correct:
        return {"score": num_correct / (max(len(model_answers), len(ground_truths)) + 1e-6), "error_code": ErrCode.SUCCESS}

    return {"score": INCORRECT_SCORE, "error_code": ErrCode.WRONG_ANSWER}


def compute_score(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    try:
        return _compute_score(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.WRONG_ANSWER}

