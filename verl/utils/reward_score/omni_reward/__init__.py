from .math.boxed import compute_score as math_boxed 
from .ocr.grits import calculate_grits_reward as ocr_grits
from .ocr.grits_shape import calculate_grits_reward as ocr_grits_shape
from .math.boxed_list import compute_score as math_boxed_list
from .code.python_unittest import compute_score as code_python_unittest
from .grounding.iou_single import compute_score as grounding_iou_single
from .gui.diou import compute_score as gui_diou
import re
compute_score_fns = {
    "math_boxed": math_boxed,
    "math_boxed_list": math_boxed_list,
    "ocr_grits": ocr_grits,
    "ocr_grits_shape": ocr_grits_shape,
    "code_python_unittest": code_python_unittest,
    "grounding_iou_single": grounding_iou_single,
    "gui_diou": gui_diou
}

# add think reward
class ThinkRewardWrapper:
    """
    maker think prefix function for think reward
    detect <think>...</think> in model_response
    output format: {"score": (1 - think_reward_weight) * task_score + think_reward_weight * think_score, "think_score": think_score, "task_score": task_score, {other infos in result}}
    """

    def __init__(self, score_fns, think_reward_weight=0.1):
        self.score_fns = score_fns
        self.think_score_fns = {}
        self.think_reward_weight = think_reward_weight
        for key, fn in score_fns.items():
            self.think_score_fns[f"think_{key}"] = self._make_think_fn(fn)
        

    def _make_think_fn(self, fn):
        def wrapped_fn(model_response, ground_truths, **kwargs):
            pattern = r"<\|im_start\|>(.*?)<\|im_end\|>"
            matches = re.findall(pattern, model_response, re.DOTALL)
            think_score = 1
            for _, match in enumerate(matches, 1):
                if "assistant" in match:
                    think_pattern = r"<think>(.*?)</think>"
                    think_matches = re.findall(think_pattern, match, re.DOTALL)
                    if len(think_matches) == 1:
                        continue
                    else:
                        think_score = 0
                        break
                    
            result = fn(model_response, ground_truths, **kwargs)
            task_score = result["score"]
            score = (1 - self.think_reward_weight) * task_score + self.think_reward_weight * think_score
            result["score"] = score
            result["think_score"] = think_score
            result["task_score"] = task_score
            return result
 
        return wrapped_fn

# 实例化ThinkRewardWrapper，并将think_前缀的函数加入compute_score_fns
think_wrapper = ThinkRewardWrapper(compute_score_fns)
compute_score_fns.update(think_wrapper.think_score_fns)


