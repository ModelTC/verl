import re


class ThinkReward:
    def __init__(
        self,
        response_prefix: str = "<|im_start|>assistant\n",
        message_pattern: str = (
            r"<\|im_start\|>(?P<role>[^\n]+)\n(?P<content>.*?)<\|im_end\|>"
        ),
        think_pattern: str = r"^<think>\s*(\S.*?)\s*</think>\s*(\S.*?)\s*$",
    ):
        self.response_prefix = response_prefix
        self.message_pattern = re.compile(message_pattern, re.DOTALL)
        self.think_pattern = re.compile(think_pattern, re.DOTALL)

    def __call__(self, model_response: str, ground_truth: bool, **kwargs):
        # the last message should be replied by the assistant
        message_list = list(
            self.message_pattern.finditer(
                self.response_prefix + model_response))
        if (
            len(message_list) == 0 or
            message_list[-1].group("role") != "assistant"
        ):
            return {"score": 0}

        think_match_of_last_response = list(
            self.think_pattern.finditer(message_list[-1].group("content")))
        think_of_last_response = len(think_match_of_last_response) > 0
        return {
            "score": int(think_of_last_response == ground_truth)
        }
