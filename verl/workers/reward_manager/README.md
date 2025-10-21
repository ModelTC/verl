# Reward Managers

## Omni Reward

[VERL is designed to select the reward function using the dataset name as the key.](https://verl.readthedocs.io/en/v0.5.x/preparation/reward_function.html) This means that when adding a new dataset, regardless of whether an existing reward function is used, [the function lookup logic in the code](/verl/utils/reward_score/__init__.py) must be updated.

[Omni Reward Manager](/verl/workers/reward_manager/omni_reward.py) modifies this design by selecting the reward function based on the `reward_model.reward_fn` field in the data sample, or `reward_fn` field in the dataset meta. This allows iterations of RL datasets within the same category to only require configuration changes without modifying the code, and also makes configuring multi-dataset combined training easier.

A predefined dictionary of reward functions is [here](/verl/utils/reward_score/omni_reward/__init__.py). Here is the example of dataset meta:

```json
{
    "dataset_name": {
        "annotation": "/path/to/data_samples.jsonl",
        ...
        "reward_fn": "math_boxed"
    }
}
```

In addition, Omni Reward Manager also supports configuring callable objects, allowing reward functions to be invoked without the need for registration. Here is the example:

```json
{
    "dataset_name": {
        "annotation": "/path/to/data_samples.jsonl",
        ...
        "reward_fn": {
            "_class_name": "verl.utils.reward_score.omni_reward.toolrl.ToolRlReward",
            "scale": 0.25
        }
    }
}
```

All available reward functions are located in `verl.utils.reward_score.omni_reward`.

### [common](/verl/utils/reward_score/omni_reward/common.py).CompositeReward

The combined reward functions with configurable weights.

The parameters:

* `func_list`: The reward functions.
* `ground_truth_key_list`: The key to extract the ground truth for the reward functions, consistent with the order of the func_list.
* `weight_list`: The score weight for the reward functions in the func_list order, consistent with the order of the func_list.

The GT: A dictionary that uses the items in "ground_truth_key_list" as keys and the corresponding ground truth (GT) required by the reward functions as values.

### [think](/verl/utils/reward_score/omni_reward/think.py).ThinkReward

The reward function to check if the think behavior of the model response follows the expected GT.

The parameters:

* `message_pattern`: The template to extract the message of roles, `r"<\|im_start\|>(?P<role>[^\n]+)\n(?P<content>.*?)<\|im_end\|>"` by default.
* `think_pattern`: The template to extract the content of think behavior, `r"<think>(.+?)</think>"` by default.

The GT: A boolean indicating if the think behavior is expected.

### [toolrl](/verl/utils/reward_score/omni_reward/toolrl.py).ToolRlReward

The reward function is implemented based on the paper ["ToolRL: Reward is All Tool Learning Needs"](https://arxiv.org/abs/2504.13958).

The parameters:

* `extract_pattern`: The template to extract the assistant message, `r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"` by default.
* `scale`: The scale of the reward score, 1.0 by default (-3.0 ~ 4.0).

The GT: the model response text containing the expected tool call in [the internal format](https://p283t9u4d9.feishu.cn/wiki/ITj5wcJp7iIsagkCv5dc5Jvxntb).

### [toolrl](/verl/utils/reward_score/omni_reward/toolrl.py).ToolFormatReward

The reward function to check if the tool calling of the model response follows the tool schema.

The parameters:

* `message_pattern`: The template to extract the assistant message, `r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>"` by default.
* `tool_call_pattern`: The template to extract the tool call blocks in the message content, `r"<tool_call>(.*?)</tool_call>"` by default.
* `tool_call_style`: The format to parse the content of tool call blocks, either `"jsonl"`(default) or `"json"`.
* `func_name_score_ratio`: The score weight of the function name matching, 0.5 by default. The weight for parameter matching is calculated as one minus the current parameter.
* `default_score`: The score if no tool calling is detected, 0.5 by default.

The GT: the list of [tool definitions in OpenAI format](https://platform.openai.com/docs/guides/function-calling?strict-mode=enabled#defining-functions).
