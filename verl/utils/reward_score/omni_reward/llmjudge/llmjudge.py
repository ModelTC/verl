import re
import ast
from func_timeout import func_set_timeout, FunctionTimedOut
from verl.utils.reward_score.utils import ErrCode
from openai import OpenAI
import requests
import random
openai_api_key = "EMPTY"

INCORRECT_SCORE = 0.
CORRECT_SCORE = 1.

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt


def get_judge_model(openai_api_base_list):
    client_list = []
    for api_base in openai_api_base_list:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=api_base,
        )
        client_list.append(client)
    model_name_list = []
    for client in client_list:
        response = requests.get(f"{api_base}/models")
        models = response.json()
        model_name_list.append(models['data'][0]['id'])
    return client_list, model_name_list

def get_judge_output(judge_input, client, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": judge_input}],
    )
    return response.choices[0].message.content

@func_set_timeout(30)
def _compute_score(model_response, ground_truths, eos_token, user_question, client_list, model_name_list):
    pred = model_response.rsplit("<|im_start|>", 1)[-1].rsplit("</think>", 1)[-1]
    judge_input = get_prompt(pred, ground_truths, user_question)
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    response = get_judge_output(judge_input, client, model_name)
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
    if '1' in response:
        acc_reward = 1.0
        return {"score": acc_reward, "error_code": ErrCode.SUCCESS}
    elif '0' in response:
        return {"score": 0, "error_code": ErrCode.WRONG_ANSWER}
    else:
        print(f' [WARNING] resp format error {response=}')
        return {"score": 0, "error_code": ErrCode.FMT_ERR}
    
def compute_score(model_response, ground_truths, **kwargs):
    eos_token = kwargs.get('eos_token', "<|im_end|>")
    judge_model_url = kwargs.get('judge_model_url', None)
    assert judge_model_url is not None, f"judge_model_url not set for llmjudge reward!"
    client_list, model_name_list = get_judge_model(judge_model_url)
    if kwargs.get('check_eos_flag', False):

        if not model_response.rstrip().endswith(eos_token):
            return {"score": INCORRECT_SCORE, "error_code": ErrCode.FMT_ERR}
    try:
        return _compute_score(model_response, ground_truths, eos_token, kwargs["user_question"], client_list, model_name_list)
    except FunctionTimedOut:
        print("A task timed out!")
        print(model_response)
        print(f'ground_truth:{ground_truths}')
        return {"score": INCORRECT_SCORE, "error_code": ErrCode.WRONG_ANSWER}