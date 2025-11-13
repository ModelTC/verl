from verl import DataProto
import torch
import json
from collections import defaultdict
from verl.common import make_object_from_config
from verl.utils.dataset.utils import AUDIO_CONTEXT_TOKEN, AUDIO_END_TOKEN, AUDIO_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, PAD_TOKEN, IMG_CONTEXT_TOKEN_QWEN
from verl.utils.reward_score.omni_reward import compute_score_fns
from verl.workers.reward_manager import register
from verl.utils.reward_score.utils import ErrCode

@register("omni")
class OmniRewardManager:

    @staticmethod
    def get_reward_function(reward_function_dict: dict, reward_indicator):
        if isinstance(reward_indicator, str):
            return reward_function_dict[reward_indicator]
        elif isinstance(reward_indicator, dict):
            func_key = json.dumps(reward_indicator, sort_keys=True)
            if func_key not in reward_function_dict:
                reward_function_dict[func_key] = make_object_from_config(
                    reward_indicator)

            return reward_function_dict[func_key]
        else:
            raise NotImplementedError(
                "The reward indicator should be str or dict.")

    """The reward manager.
    """
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score_fns.copy()
        self.reward_fn_key = reward_fn_key

        self.is_val = kwargs.get('is_val', False)
        self.service_url = kwargs.get('service_url', None)
        self.tool_penalty  = kwargs.get('tool_penalty', 0.0)
        self.log_num_round = kwargs.get('log_num_round', False)
        self.judge_model_url = kwargs.get('judge_model_url', None)

    def __call__(self, data: DataProto, return_dict: bool=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        def _print(data_item, score=None):
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids).replace(IMG_CONTEXT_TOKEN_QWEN, "").replace(PAD_TOKEN, "").replace(IMG_CONTEXT_TOKEN, "")
            response_str = self.tokenizer.decode(valid_response_ids).replace(IMG_CONTEXT_TOKEN_QWEN, "").replace(PAD_TOKEN, "").replace(IMG_CONTEXT_TOKEN, "")
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # print(f"[prompt]:{prompt_str}")
            # print(f"[response]:{response_str}")
            # print(f"[ground_truth]:{ground_truth}")
            # if score is not None:
            #     print(f"[score]:{score}")


        def _process_item(args):
            i, data_item, already_print_data_sources, is_val = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            valid_response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            reward_fn = OmniRewardManager.get_reward_function(
                self.compute_score,
                data_item.non_tensor_batch['reward_model']['reward_fn'],
            )
            user_question = self.tokenizer.decode(prompt_ids).split("|im_start|>user\n")[-1].split("|im_end|")[0].replace(IMG_CONTEXT_TOKEN_QWEN, "").replace(PAD_TOKEN, "").replace(IMG_CONTEXT_TOKEN, "").replace(AUDIO_CONTEXT_TOKEN, "").replace(IMG_START_TOKEN, "").replace(IMG_END_TOKEN, "").replace(AUDIO_START_TOKEN, "").replace(AUDIO_END_TOKEN, "")
            try:
                score = reward_fn(
                    valid_response_str, 
                    ground_truth, 
                    eos_token=self.tokenizer.eos_token,
                    service_url=self.service_url,
                    data_non_tensor_batch=data_item.non_tensor_batch,
                    tool_penalty=self.tool_penalty,
                    is_val=is_val,
                    user_question=user_question,
                    judge_model_url=self.judge_model_url
                )
            except Exception as e:
                print(f"!!!Error in reward_fn: {e}")
                score = {"score": 0., "error_code": ErrCode.RUNTIME_ERR}

            assert isinstance(score, dict), f"score is not a dict: {score}"

            if self.log_num_round:
                score['num_round'] = valid_response_str.count("<|im_start|>assistant")
            
            return i, score, valid_response_length

        results = []
        for i in range(len(data)):
            args = (i, data[i], already_print_data_sources, self.is_val)
            result = _process_item(args)
            results.append(result)

        _print(data[0], results[0][1])

        if isinstance(results[0][1], dict):
            all_keys = set()
            for _, score, _ in results:
                all_keys.update(score.keys())

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key in all_keys:
                    reward_extra_info[key].append(score.get(key, None))
            else:
                reward = score
            reward_tensor[i, valid_response_length - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
