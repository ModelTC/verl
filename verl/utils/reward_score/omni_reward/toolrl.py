import json
import re
from verl.utils.reward_score.omni_reward.common import string_iou
import verl.utils.reward_score.toolrl as deepscaler_toolrl


class ToolRlReward:
    def __init__(
        self,
        extract_pattern: str = r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
        scale: float = 1.0,
    ):
        self.extract_pattern = extract_pattern
        self.scale = scale

    def __call__(self, model_response: str, ground_truth: str, **kwargs):
        extra_info = {
            "extract_pattern": self.extract_pattern,
            **kwargs
        }
        raw_result = deepscaler_toolrl.compute_score(
            model_response, ground_truth, extra_info)

        result = {"score": self.scale * raw_result["score"]}
        return result


class ToolFormatReward:

    tool_call_parsers = {
        "json": lambda s: json.loads(s),
        "jsonl": lambda s: [
            json.loads(i) for i in s.split("\n") if len(i.strip()) > 0
        ],
    }
    to_json_types = {
        "bool": "boolean",
        "dict": "object",
        "float": "number",
        "int": "number",
        "list": "array",
        "str": "string",
    }

    def __init__(
        self,
        message_pattern: str = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
        tool_call_pattern: str = r"<tool_call>(.*?)</tool_call>",
        tool_call_style: str = "jsonl",
        func_name_score_ratio: float = 0.5,
        default_score: float = 0.5,
    ):
        self.message_pattern = re.compile(message_pattern, re.DOTALL)
        self.tool_call_pattern = re.compile(tool_call_pattern, re.DOTALL)
        self.tool_call_style = tool_call_style
        self.func_name_score_ratio = func_name_score_ratio
        self.default_score = default_score

    def __call__(self, model_response: str, ground_truth: list, **kwargs):
        tool_calls = [
            ToolFormatReward.tool_call_parsers[self.tool_call_style](j)
            for i in self.message_pattern.findall(model_response)
            for j in self.tool_call_pattern.findall(i)
        ]
        scores = []
        for tool_call in tool_calls:
            func_name_max_iou = max(
                string_iou(tool_call["name"], i["function"]["name"])
                for i in ground_truth
            )
            if func_name_max_iou == 1.0:
                tool = [
                    i for i in ground_truth
                    if tool_call["name"] == i["function"]["name"]
                ][0]
                tool_params = tool["function"]["parameters"]
                required_params = sorted(tool_params["required"])
                optional_params = [
                    i for i in tool_params["properties"].keys()
                    if i not in required_params
                ]
                non_optional_args = sorted(
                    i for i in tool_call["arguments"].keys()
                    if i not in optional_params
                )

                param_name_type_string = ",".join(
                    "{}:{}".format(i, tool_params["properties"][i]["type"])
                    for i in required_params
                )
                arg_name_type_string = ",".join(
                    "{}:{}".format(
                        i, (
                            ToolFormatReward.to_json_types[type(i).__name__]
                            if type(i).__name__ in ToolFormatReward.to_json_types
                            else "object"
                        ))
                    for i in non_optional_args
                )
                arg_name_type_iou = (
                    string_iou(arg_name_type_string, param_name_type_string)
                    if len(arg_name_type_string) > 0 or
                    len(param_name_type_string) > 0
                    else 1.0 # both args and params are empty
                )
            else:
                arg_name_type_iou = 0.0

            scores.append(
                func_name_max_iou * self.func_name_score_ratio +
                arg_name_type_iou * (1.0 - self.func_name_score_ratio)
            )

        return {
            "score": (
                sum(scores) / len(scores)
                if len(scores) > 0
                else self.default_score
            )
        }
