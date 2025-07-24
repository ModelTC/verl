"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union
from dataclasses import dataclass

from verl.utils.reward_score.mmscore.deepscaler.reward_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from verl.utils.reward_score.mmscore.deepscaler.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from verl.utils.reward_score.mmscore.deepscaler.system_prompts import ORM_PROMPT
from verl.utils.reward_score.mmscore.deepscaler.utils import call_gemini_llm, call_oai_rm_llm
import re   

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

@dataclass
class RewardConfig:
    # Use LLM as ORM to evaluate correctness.
    use_math_orm: bool = False
    
    # General reward constants.
    correct_reward: float = 1.0
    incorrect_reward: float = 0
    format_error_reward: float = 0
    unk_error_reward: float = 0


# add different format check
def think_and_answer_format_fn(solution):
    pattern = r'^\s?<think>.*?</think>\s?<answer>.*?</answer>\s?$'
    match = re.match(pattern, solution, re.DOTALL)
    print("use think and answer")
    return match is not None

def think_format_fn(solution):
    solution = solution.strip()

    if solution.count(THOUGHT_DELIMITER_START) != 1 or solution.count(THOUGHT_DELIMITER_END) != 1:
        return False

    start_index = solution.find(THOUGHT_DELIMITER_START)
    end_index = solution.find(THOUGHT_DELIMITER_END)

    return start_index == 0 and start_index < end_index

def free_format_fn(solution):
    return True

def get_format_reward(format_reward_type):
    if format_reward_type == "think":
        return think_format_fn
    elif format_reward_type == "think_and_answer":
        return think_and_answer_format_fn
    else:
        return free_format_fn

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput, check_format) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response

        # # Extract solution.
        # if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        
        # model_answer = extract_answer(model_solution)

        
        model_answer = extract_answer(model_response)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
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
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:

                # check format
                if get_format_reward(check_format)(model_response):
                    return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                else:
                    return RewardOutput(reward=self.config.correct_reward-0.1, is_correct=True)

                
        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_gemini_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

# add timeout setting
from func_timeout import func_set_timeout
@func_set_timeout(30)
def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], check_format="think", is_val = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = False
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}), check_format=check_format if not is_val else "free")
    return reward_response.reward, {'score': reward_response.reward}

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
