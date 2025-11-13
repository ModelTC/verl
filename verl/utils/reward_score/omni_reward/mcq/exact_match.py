import re
import string
import random
import json

INCORRECT_SCORE = 0.
CORRECT_SCORE = 1.

def normalize_answer(s):
    def remove_articles(text):
        if text.strip().lower() in ["a"]:
            return text
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(model_response):
    """Extract the equation from the solution string."""
    # First try to extract from <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'  
    match = re.finditer(answer_pattern, model_response, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return model_response.split('<|im_end|>')[0].strip()
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score(model_response, ground_truth, **kwargs):
    """The scoring function for exact match (EM).
    """
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    model_response = model_response.rsplit("<|im_start|>", 1)[-1].rsplit("</think>", 1)[-1]
    answer = extract_solution(model_response=model_response)  # 正则表达式提取答案 <answer>(.*?)</answer>

    # do_print = random.randint(1, 64) == 1
    # if do_print:
    #     print(f"--------------------------------")
    #     print(f"Golden answers: {ground_truth}")
    #     print(f"Extracted answer: {answer}")
    #     print(f"Solution string: {model_response}")
    
    if answer is None:
        score = 0
    else:
        if em_check(answer, ground_truth):
            score = 1
        else:
            score = 0
    return {"score": score}


if __name__ == '__main__':
    model_response = '<answer> A.</answer><|im_end|>'
    # model_response = 'ab'
    ground_truth = 'A'
    print(compute_score(model_response, ground_truth))