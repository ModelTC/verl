
INPUT_TEMPLATE = {
    "base": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {} Please reason step by step, and put your final answer within \\boxed{{}}. Assistant: <think>\n",
    "distill": "<｜begin▁of▁sentence｜><｜User｜>{} Let's think step by step and output the final answer within \\boxed{{}}.<｜Assistant｜><think>\n"
}