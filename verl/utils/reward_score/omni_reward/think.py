import re


class ThinkReward:
    def __init__(
        self,
        message_pattern: str = (
            r"<\|im_start\|>(?P<role>[^\n]+)\n(?P<content>.*?)<\|im_end\|>"
        ),
        think_pattern: str = r"<think>(.+?)</think>"
    ):
        self.message_pattern = re.compile(message_pattern, re.DOTALL)
        self.think_pattern = re.compile(think_pattern, re.DOTALL)

    def __call__(self, model_response: str, ground_truth: bool, **kwargs):
        think_count_of_all_responses = [
            len(self.think_pattern.findall(i.group("content"))) > 0
            for i in self.message_pattern.finditer(model_response)
            if i.group("role") == "assistant"
        ]
        if len(think_count_of_all_responses) == 0:
            return {"score": 0}

        return {
            "score": int(think_count_of_all_responses[-1] == ground_truth)
        }
