# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any
import re

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.utils.dataset.utils import IMG_CONTEXT_TOKEN, PAD_TOKEN, IMG_CONTEXT_TOKEN_QWEN


@register("instruction_following")
class InstructionFollowingRewardManager():
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", model_path=None, **kwargs) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        
        self.is_val = kwargs.get('is_val', False)
        self.service_url = kwargs.get('service_url', None)
        self.tool_penalty = kwargs.get('tool_penalty', 0.0)
        self.log_num_round = kwargs.get('log_num_round', False)
        self.judge_model_url = kwargs.get('judge_model_url', [])
        
        self.model_name = "/".join(model_path.split("/")[-2:]) if model_path is not None else None

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # 如果已有 rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": defaultdict(list)}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        reward_sources = ["if_verifier"]
        data_infos = defaultdict(list)
        already_print_data_sources = {}

        # 第一轮：整理数据
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode - 移除特殊token
            prompt_str = self.tokenizer.decode(valid_prompt_ids).replace(IMG_CONTEXT_TOKEN_QWEN, "").replace(PAD_TOKEN, "").replace(IMG_CONTEXT_TOKEN, "")
            response_str = self.tokenizer.decode(valid_response_ids).replace(IMG_CONTEXT_TOKEN_QWEN, "").replace(PAD_TOKEN, "").replace(IMG_CONTEXT_TOKEN, "")

            # 提取user question
            user_question = None
            match = re.search(r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>', prompt_str, re.DOTALL)
            if match:
                user_question = match.group(1)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns
            
            # 获取supporting_facts和reward_fn_coefs
            supporting_facts = data_item.non_tensor_batch['reward_model'].get('supporting_facts', None)
            reward_fn_coefs = data_item.non_tensor_batch['reward_model'].get('reward_fn_coefs', None)

            # 保存信息用于批量计算
            data_infos["prompt_str"].append(prompt_str)
            data_infos["response_str"].append(response_str)
            data_infos["ground_truth"].append(ground_truth)
            data_infos["extra_info"].append(extra_info)
            data_infos["valid_response_length"].append(valid_response_length)
            data_infos["data_source"].append(data_source)
            data_infos["user_question"].append(user_question)
            data_infos["supporting_facts"].append(supporting_facts)
            data_infos["reward_fn_coefs"].append(reward_fn_coefs)

        # 第二轮：批量计算 reward
        for reward_source in reward_sources:
            try:
                # default_compute_score 只接受基本参数
                scores = self.compute_score(
                    prompt_str=data_infos["prompt_str"],
                    data_source=reward_source,
                    solution_str=data_infos["response_str"],
                    ground_truth=data_infos["ground_truth"],
                    extra_info=data_infos["extra_info"],
                )
            except Exception as e:
                print(f"!!!Error in compute_score for {reward_source}: {e}")
                scores = [0] * len(data_infos["response_str"])

            for i, score in enumerate(scores):
                resp = data_infos["response_str"][i].strip()
                
                # 检查是否overlong
                if resp.endswith("<|im_end|>"):
                    if isinstance(score, dict):
                        reward = score.get("score", score)
                        score['overlong'] = False
                    else:
                        reward = score
                else:
                    # 如果没有正确结束,给负奖励
                    if isinstance(score, dict):
                        score['score'] = -0.1
                        score['overlong'] = True
                        reward = -0.1
                    else:
                        reward = -0.1
                
                # 收集额外信息
                if isinstance(score, dict):
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                
                # 如果需要记录轮数
                if self.log_num_round:
                    num_round = data_infos["response_str"][i].count("<|im_start|>assistant")
                    reward_extra_info['num_round'].append(num_round)
                
                reward_tensor[i, data_infos["valid_response_length"][i] - 1] += reward

                # 打印前几条样例
                data_source = data_infos["data_source"][i]
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", data_infos["prompt_str"][i])
                    print("[response]", data_infos["response_str"][i])
                    print("[ground_truth]", data_infos["ground_truth"][i])
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

        # 返回
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # data_source = data_item.non_tensor_batch['data_source']
            data_source = "hf_rm"

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                prompt_str=prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores