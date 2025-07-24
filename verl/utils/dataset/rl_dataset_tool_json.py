import os
from typing import Optional
import json
from transformers import PreTrainedTokenizer, ProcessorMixin
import torch
import random
from PIL import UnidentifiedImageError

import copy
import re
import io
from omegaconf import DictConfig, ListConfig
from typing import List, Optional, Union

import base64

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.dataset.utils import dynamic_preprocess_v3, build_transform, preprocess_internlm_v3_mm_chat, IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
from verl.utils.dataset.rl_dataset_json import process_image

SYSTEM_PROMPT = "You are capable of invoking tools such as visual perception and Python code execution. Upon receiving an uploaded image, you may utilize visual perception for interpretation as needed, and employ programming capabilities for computation, image processing, and related tasks."
INTERPRETER_PROMPT = "\n\tYou now have access to a Jupyter notebook environment supporting Python code execution. Just send code to python to run in this stateful environment. This feature is suitable for:- Data analysis or processing (such as data manipulation and graphic creation)\n- Complex calculations (such as math and physics problems)\n- Programming examples (for understanding programming concepts or language features)\n- Text processing and analysis (including text analysis and natural language processing)\n- Machine learning and data science (model training and data visualization)\n- File operations and data import (handling CSV, JSON, etc. formats)"
INTERNVL_SYSTEM_PROMPT = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "interpreter", "content": INTERPRETER_PROMPT},
]
QWENVL_SYSTEM_PROMPT = [
    {"role": "system", "content": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"python\", \"description\": \"Execute Python code in a stateful Jupyter notebook environment.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"code\": {\"type\": \"string\", \"description\": \"Python code to execute\"}}, \"required\": [\"code\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"},
]


class RLHFToolJSONDataset(RLHFDataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        from verl.utils.dataset.utils import TCSLoader
        self.tcs_loader = TCSLoader(conf_path=f'{os.getenv("HOME")}/aoss.conf')
        
        meta_file = data_files
        assert os.path.exists(meta_file), f'meta_file {meta_file} not found'
        self.meta_info = json.load(open(meta_file, 'r'))

        self.apply_internvl_preprocess = config.get('internvl_config', {}).get('apply_internvl_preprocess', False)
        if self.apply_internvl_preprocess:
            self.system_prompt = INTERNVL_SYSTEM_PROMPT
        else:
            self.system_prompt = QWENVL_SYSTEM_PROMPT

        super().__init__(data_files, tokenizer, config, processor)
        

    def _read_files_and_tokenize(self):
        self.dataframe = []
        for dataset_name, dataset_info in self.meta_info.items():
            data_list = self.tcs_loader(dataset_info['annotation'], file_type='jsonl')
            assert len(data_list) == dataset_info['length'], f"dataset {dataset_name} has {len(data_list)} samples, but meta file says there are {dataset_info['length']} samples"
            if 'index' not in data_list[0]:
                for i, d in enumerate(data_list):
                    d['index'] = i

            # process repeat_time
            full_repeats = data_list * int(dataset_info['repeat_time'])
            extra_samples = random.sample(data_list, int(len(data_list) * (dataset_info['repeat_time'] - int(dataset_info['repeat_time']))))
            data_list = full_repeats + extra_samples

            for d in data_list:
                d = d.copy()
                d['data_source'] = dataset_name
                d['id'] = f"{dataset_name}-{d['index']}"

                # process image path
                if dataset_info['root'] is not None and self.image_key in d:
                    image = d[self.image_key]
                    if not isinstance(image, list):
                        image = [image]
                    image = [os.path.join(dataset_info['root'], i) for i in image]
                    d[self.image_key] = image
                
                self.dataframe.append(d)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            self.dataframe = [doc for doc in self.dataframe if len(
                tokenizer.apply_chat_template(self.system_prompt + doc[self.prompt_key], add_generation_prompt=True)) <= self.max_prompt_length]

            print(f'filter dataset len: {len(self.dataframe)}')

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item].copy()
        if self.apply_internvl_preprocess:
            return self._internvl_preprocess(row_dict)
        else:
            return self._qwenvl_preprocess(row_dict)

    def _internvl_preprocess(self, row_dict):
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']

        chat = row_dict.pop(self.prompt_key)
        images = row_dict.pop(self.image_key)
        assert len(images) == 1
        image = images[0]
        
        content = chat[-1]['content']
        chat[-1]['content'] = f"<|图片编号|>1<|图片名|>{os.path.basename(image)}\n![IMAGE]({image})\n{content}"
        chat = self.system_prompt + list(chat)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        raw_prompt = prompt_with_chat_template

        model_inputs = self.tokenizer(prompt_with_chat_template, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        output_dict['input_ids'] = input_ids[0]
        output_dict['attention_mask'] = attention_mask[0]
        output_dict['position_ids'] = position_ids[0]
        output_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        output_dict["prompt_length"] = output_dict['position_ids'].max().item() + 1

        # encode prompts without chat template
        if self.return_raw_chat:
            output_dict['raw_prompt'] = chat

        # add index for each prompt
        output_dict["index"] = row_dict['id']

        return output_dict

    def _qwenvl_preprocess(self, row_dict):
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']

        chat = row_dict.pop(self.prompt_key)
        images = row_dict.pop(self.image_key)
        assert len(images) == 1
        image = images[0]
        
        content = chat[-1]['content']
        chat[-1]['content'] = f"![IMAGE]({image})\n{content}"
        chat = self.system_prompt + list(chat)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        raw_prompt = prompt_with_chat_template

        model_inputs = self.tokenizer(prompt_with_chat_template, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        output_dict['input_ids'] = input_ids[0]
        output_dict['attention_mask'] = attention_mask[0]
        output_dict['position_ids'] = position_ids[0]
        output_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        output_dict["prompt_length"] = output_dict['position_ids'].max().item() + 1

        # encode prompts without chat template
        if self.return_raw_chat:
            output_dict['raw_prompt'] = chat

        # add index for each prompt
        output_dict["index"] = row_dict['id']

        return output_dict