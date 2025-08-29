import re
import ast
import numpy as np
import enum
import json
from func_timeout import func_set_timeout, FunctionTimedOut
from verl.utils.reward_score.utils import ErrCode

TEXT_ANLS_THRESHOLD = 0.5

# 格式reward
FORMAT_FAULT_REWARD = 0.0
FORMAT_CORRECT_REWARD = 0.5

# 计算reward
VALUE_CORRECT_REWARD = 1.0

def extract_ocr(model_response):
    bbox_pattern = r'<ocr>(.*?)</ocr>'
    bbox_matches = re.findall(bbox_pattern, model_response, re.DOTALL)
    if bbox_matches:
        # get the last match
        try:
                content = bbox_matches[-1].strip()
                return content
        except:
                return None
    return None


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def levenshtein_similarity(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min(
                    distances[i1],     # 替换
                    distances[i1 + 1], # 插入
                    distances_[-1]     # 删除
                ))
        distances = distances_

    edit_distance = distances[-1]
    max_len = max(len(s1), len(s2))
    similarity = 1 - (edit_distance / max_len)
    return similarity

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
                return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
                num_left_braces_open += 1
        if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

@func_set_timeout(30)
def _calculate_levenshtein(model_response, ground_truths, eos_token):
    model_answer = extract_ocr(model_response)
    if model_answer is None:
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    
    gt_text = ground_truths['text']
    try:
        retval = levenshtein_similarity(model_answer, gt_text) + FORMAT_CORRECT_REWARD
        return {"score": retval, "error_code": ErrCode.SUCCESS}

    except Exception as e:
        print(e)

    return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

@func_set_timeout(30)
def _calculate_levenshtein_ignore_capital(model_response, ground_truths, eos_token):
      model_answer = extract_ocr(model_response)
      if model_answer is None:
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
      
      gt_text = ground_truths['text'].lower()
      model_answer = model_answer.lower()
      try:
            retval = levenshtein_similarity(model_answer, gt_text) + FORMAT_CORRECT_REWARD
            return {"score": retval, "error_code": ErrCode.SUCCESS}

      except Exception as e:
            print(e)

      return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

@func_set_timeout(30)
def _calculate_levenshtein_w_value(model_response, ground_truths, eos_token):
      model_answer = extract_ocr(model_response)
      if model_answer is None:
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
      
      print(f"eval gt: {ground_truths}")
      if isinstance(ground_truths['text'], float) or isinstance(ground_truths['text'], int):
            try:
                  if '.' in model_answer:
                        model_answer = float(model_answer)
                  else:
                        model_answer = int(model_answer)
            
                  if ground_truths['text'] == model_answer:
                        return {"score": retval, "error_code": ErrCode.SUCCESS}
                  else:
                        return {"score": FORMAT_CORRECT_REWARD, "error_code": ErrCode.WRONG_ANSWER}
            except:
                  return {"score": FORMAT_CORRECT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

      else:
            gt_text = json.dumps(ground_truths['text'])
            try:
                  retval = levenshtein_similarity(model_answer, gt_text) + FORMAT_CORRECT_REWARD
                  return {"score": retval, "error_code": ErrCode.SUCCESS}

            except Exception as e:
                  print(e)

      return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

@func_set_timeout(30)
def _calculate_levenshtein_multi_answers(model_response, ground_truths, eos_token):
    model_answer = extract_ocr(model_response)
    if model_answer is None:
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    
    print(f"eval gt: {ground_truths}")
    gt_answers = ground_truths['answers']
    try:
        for answer in gt_answers:
                each_answer_reward = levenshtein_similarity(model_answer, answer) + FORMAT_CORRECT_REWARD
                retval = max(each_answer_reward, retval)
        return {"score": retval, "error_code": ErrCode.SUCCESS}
    
    except Exception as e:
        print(e)

    return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

@func_set_timeout(30)
def _calculate_captcha_task1(model_response, ground_truths, eos_token):
    model_answer = extract_ocr(model_response)
    
    string_in_last_boxed = last_boxed_only_string(model_response)
    if string_in_last_boxed is None:
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}

    model_caculate_value = remove_boxed(string_in_last_boxed)
    if model_caculate_value is None:
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}

    print(f"eval gt: {ground_truths}")
    gt_text = json.dumps(ground_truths['text'])
    gt_answer = ground_truths['answer']

    try:
        similarity = levenshtein_similarity(model_answer, gt_text)
        retval = similarity + FORMAT_CORRECT_REWARD
        if similarity < TEXT_ANLS_THRESHOLD:
                return {"score": retval, "error_code": ErrCode.SUCCESS}
        else:
                if int(model_caculate_value) != gt_answer:
                    retval += VALUE_CORRECT_REWARD
                return {"score": retval, "error_code": ErrCode.SUCCESS}
        
    except Exception as e:
        print(e)

    return {"score": FORMAT_CORRECT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

def calculate_levenshtein(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    try:
        return _calculate_levenshtein(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

# 忽略大小写
def calculate_levenshtein_ignore_capital(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    try:
        return _calculate_levenshtein_ignore_capital(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

## gt中可能包含数值
def calculate_levenshtein_w_value(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    try:
        return _calculate_levenshtein(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}

def calculate_levenshtein_multi_answers(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    try:
        return _calculate_levenshtein_multi_answers(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}
      
def calculate_captcha_task1(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    if kwargs.get('check_eos_flag', False):
        if not model_response.rstrip().endswith(eos_token):
            return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.FMT_ERR}
    try:
        return _calculate_captcha_task1(model_response, ground_truths, eos_token)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": FORMAT_FAULT_REWARD, "error_code": ErrCode.WRONG_ANSWER}