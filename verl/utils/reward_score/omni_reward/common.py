import re


class CompositeReward:
    def __init__(
        self, func_list: list, ground_truth_key_list: list, weight_list: list,
    ):
        """The weighted composite reward.
        Args:
            func_list: The reward functions.
            ground_truth_key_list: The key to extract the ground truth for the
                reward functions, consistent with the order of the func_list.
            weight_list: The score weight for the reward functions in the
                func_list order, consistent with the order of the func_list.
        """
        assert all(callable(i) for i in func_list)
        assert abs(sum(weight_list) - 1) < 1e-4
        self.func_list = func_list
        self.ground_truth_key_list = ground_truth_key_list
        self.weight_list = weight_list

    def __call__(self, model_response: str, ground_truth: dict, **kwargs):
        return {
            "score": sum(
                func(
                    model_response, ground_truth[gt_key], **kwargs
                )["score"] * weight
                for func, gt_key, weight in zip(
                    self.func_list, self.ground_truth_key_list,
                    self.weight_list,
                )
            )
        }


def check_last_response_answer_iou(
    model_response: str, ground_truths: list, **kwargs
):
    responses = extract_assistant_responses(model_response)
    last_response = responses[-1] if len(responses) > 0 else ""
    turn_panalty = kwargs.get("turn_penalty", 0.85)
    score = (
        max(string_iou(last_response, i) for i in ground_truths) *
        turn_panalty ** (len(responses) - 1)
    )

    return {"score": score, "error_code": 0}


def extract_assistant_responses(model_response: str):
    pattern = r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"
    return [
        i
        for i in re.findall(pattern, model_response, re.DOTALL)
        if i.strip() != ""
    ]


def longest_common_substring_length(a: str, b: str):
    state = [[0 for _ in range(1 + len(b))] for _ in range(2)]
    for i in range(1, 1 + len(a)):
        previous_row = state[(i - 1) % 2]
        row = state[i % 2]
        for j in range(1, 1 + len(b)):
            row[j] = max(
                max(previous_row[j], row[j - 1]),
                previous_row[j - 1] + int(a[i - 1] == b[j - 1]))

    return max(state[len(a) % 2])


def string_iou(a: str, b: str):
    lcs_len = longest_common_substring_length(a, b)
    return lcs_len / (len(a) + len(b) - lcs_len)
