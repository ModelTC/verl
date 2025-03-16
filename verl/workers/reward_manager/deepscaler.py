from verl import DataProto
# from verl.utils.reward_score import _default_compute_score
import torch
import json

class DeepScalerRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, is_val=False, log_file=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.is_val = is_val
        self.log_file = log_file
        self.compute_score_fn = compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_info = {}

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()

        def process_timeout_item(args, log_file):
            # i, data_item, already_print_data_sources = args
            i, data_item, already_print_data_sources, is_val = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            print("A task timed out!")
            print(sequences_str)
            print(f'ground_truth:{ground_truth}')

            # Write query-score pairs to JSONL if log_file is provided
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    record = {
                        "sequence": sequences_str,
                        "ground_truth": ground_truth,
                        "timeout": True,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            return i, 0., valid_response_length, {'score': 0.}


        def _print(data_item, reward_info, log_file=None):
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            print(sequences_str)
            print(f'ground_truth:{ground_truth}')
            print(reward_info)

            # Write query-score pairs to JSONL if log_file is provided
            if log_file:
                with open(log_file, "a", encoding="utf-8") as f:
                    record = {
                        "sequence": sequences_str,
                        "ground_truth": ground_truth,
                        "reward": reward_info,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        def process_item(args):
            # i, data_item, already_print_data_sources = args
            i, data_item, already_print_data_sources, is_val = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            # compute_score_fn = _select_rm_score_fn(data_source)
            # score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            score, info = self.compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, is_val=is_val)
            
            return i, score, valid_response_length, info

        # Process items in parallel using ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     args = [(i, data[i], already_print_data_sources, self.is_val) for i in range(len(data))]
        #     results = list(executor.map(process_item, args))
        
        import func_timeout
        results = []
        for i in range(len(data)):
            args = (i, data[i], already_print_data_sources, self.is_val)
            try:
                result = process_item(args)
            except func_timeout.FunctionTimedOut:
                result = process_timeout_item(args, self.log_file)
            results.append(result)

        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     args = [(i, data[i], already_print_data_sources, self.is_val) for i in range(len(data))]
        #     futures = [executor.submit(process_item, arg) for arg in args]

        #     results = []
        #     for i, future in enumerate(futures):
        #         try:
        #             result = future.result(timeout=60)
        #             results.append(result)
        #         except TimeoutError:
        #             print("A task timed out!")
        #             result = process_timeout_item(args[i], self.log_file)
        #             results.append(result)

        _print(data[0], results[0][-1], log_file=self.log_file)
        # for i in range(len(data)):
        #     _print(data[i], results[i][-1], log_file=self.log_file)

        # Fill reward tensor with results
        for i, score, valid_response_length, info in results:
            reward_tensor[i, valid_response_length - 1] = score
            for k, v in info.items():
                if k not in reward_info:
                    reward_info[k] = torch.zeros(len(data))
                reward_info[k][i] = v

        # if self.is_val:
        #     return reward_tensor
        # else:
        #     return reward_tensor, reward_info
        return reward_tensor