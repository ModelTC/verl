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
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import os
import json
import random
import re
from typing import List, Union
from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset_json import process_image


class SFTJSONDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, meta_file: str, tokenizer, config):
        from verl.utils.dataset.utils import TCSLoader
        self.tcs_loader = TCSLoader(conf_path=f'{os.getenv("HOME")}/aoss.conf') 

        assert os.path.exists(meta_file), f"meta_file {meta_file} not found"
        self.meta_info = json.load(open(meta_file, "r"))

        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_length = max_length

        if config.get('processor', None):
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(config.processor)
        else:
            self.processor = None

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.data_list = []
        for dataset_name, dataset_info in self.meta_info.items():
            root = dataset_info["root"]
            annotations = dataset_info["annotations"]
            length = dataset_info["length"]
            repeat_time = dataset_info["repeat_time"]

            # load annotations
            assert os.path.exists(annotations), f"annotations {annotations} not found"
            data_list = []
            with open(annotations, "r") as f:
                for line in f.readlines():
                    data_list.append(json.loads(line))
            assert len(data_list) == length , f"length of data_list {len(data_list)} is not equal to {length}"
            if 'index' not in data_list[0]:
                for i, d in enumerate(data_list):
                    d['index'] = i

            # process repeat_time
            data_list = data_list * int(repeat_time // 1) + random.sample(data_list, int(len(data_list) * (repeat_time % 1)))

            for i, d in enumerate(data_list):
                d = d.copy()
                d['id'] = f"{dataset_name}-{d['index']}"
                if 'image' in d:
                    if isinstance(d['image'], str):
                        d['image'] = [os.path.join(root, d['image'])]
                    else:
                        d['image'] = [os.path.join(root, image) for image in d['image']]
                
                self.data_list.append(d)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.data_list[item]
        try:
            return self.process_data(data)
        except Exception as e:
            print(f"data {data['id']} error: {e}")
            return self.__getitem__(random.randint(0, len(self)-1))


    def process_data(self, data):
        tokenizer = self.tokenizer
        messages = data['messages']
        pattern = r'<\|im_start\|>system\n.*?<\|im_end\|>\n'

        input_ids, attention_mask, loss_mask = [], [], []
        
        images = data.pop('image', [])
        has_image = len(images) > 0
        if not has_image:
            images = [Image.new('RGB', (28, 28), (255, 255, 255))]
        multi_modal_data = {'image': [process_image(self.tcs_loader(image), max_pixels=1024*1024) for image in images]}
        image_inputs = self.processor.image_processor(multi_modal_data['image'], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        multi_modal_inputs = {key: val for key, val in image_inputs.items()}
        multi_modal_inputs['image_flags'] = torch.tensor([1] * multi_modal_inputs['pixel_values'].size(0), dtype=torch.long) * int(has_image)
        image_index = 0
        if image_grid_thw is not None:
            merge_length = self.processor.image_processor.merge_size**2

        assert len(images) - (1-int(has_image)) == tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).count('<image>'), f"len(images) {len(images) - (1-int(has_image))} != count('<image>') {tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).count('<image>')} {data['id']}"

        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']

            if role == 'user':
                seq = tokenizer.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
                seq = re.sub(pattern, '', seq)

                if image_grid_thw is not None:
                    while '<image>' in seq:
                        seq = seq.replace(
                            '<image>', 
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[image_index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        image_index += 1
                    
                    seq = seq.replace('<|placeholder|>', self.processor.image_token)
                    
                ids_output = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
                input_ids.append(ids_output["input_ids"][0])
                attention_mask.append(ids_output["attention_mask"][0])
                loss_mask.extend([0] * len(ids_output["input_ids"][0]))
            elif role == 'system':
                seq = tokenizer.apply_chat_template([msg], add_generation_prompt=False, tokenize=False)
                ids_output = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
                input_ids.append(ids_output["input_ids"][0])
                attention_mask.append(ids_output["attention_mask"][0])
                loss_mask.extend([0] * len(ids_output["input_ids"][0]))
            elif role == 'assistant':
                seq = content + tokenizer.eos_token + '\n'
                ids_output = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
                input_ids.append(ids_output["input_ids"][0])
                attention_mask.append(ids_output["attention_mask"][0])
                if msg.get('is_input_only', False):
                    loss_mask.extend([0] * len(ids_output["input_ids"][0]))
                else:
                    loss_mask.extend([1] * (len(ids_output["input_ids"][0]) - 1) + [0])

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        loss_mask = torch.tensor(loss_mask, dtype=torch.bool)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        from verl.models.transformers.qwen2_vl import get_rope_index

        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )  # (3, seq_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "sequence": tokenizer.decode(input_ids),
            "multi_modal_inputs": multi_modal_inputs,
        }
