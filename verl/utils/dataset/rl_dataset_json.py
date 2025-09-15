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
from verl.utils.dataset.image_processors import pil_to_bytes

def use_image_bytes(func):
    def wrapper(self, row_dict):
        output_dict = func(self, row_dict)
        if self.config.get('use_image_bytes', False):
            output_dict['multi_modal_data']['image'] = [pil_to_bytes(image) for image in output_dict['multi_modal_data']['image']]
            output_dict['multi_modal_inputs'] = {}
        return output_dict
    return wrapper

def process_image(image: dict, max_pixels: int = 1536 * 1536, min_pixels: int = 128 * 128, max_aspect_ratio: float = 100):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    width, height = image.width, image.height

    current_ratio = max(width, height) / min(width, height)
    if current_ratio > max_aspect_ratio:
        if width > height:
            width = int(height * max_aspect_ratio)
        else:
            height = int(width * max_aspect_ratio)

    resize_factor = 1

    if (width * height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))

    elif (width * height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))

    short_side = min(width, height)
    if short_side < 28:
        resize_factor = max(resize_factor, 28 / short_side)

    width, height = int(width * resize_factor), int(height * resize_factor)
    image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFJSONDataset(RLHFDataset):
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

        self.meta_file = meta_file
        self.meta_info = json.load(open(meta_file, 'r'))

        self.input_template = config.get("input_template", None)

        # internvl dataset config
        self.apply_internvl_preprocess = config.get('apply_internvl_preprocess', False)
        if self.apply_internvl_preprocess:
            try:
                internvl_config_file = config['internvl_config_file']
            except KeyError:
                raise ValueError("`internvl_config_file` is required when `apply_internvl_preprocess` is True")

            assert os.path.exists(internvl_config_file), f"internvl_config_file {internvl_config_file} not found"
            internvl_config = json.load(open(internvl_config_file, 'r'))
            self.image_size = internvl_config.get('force_image_size', None) or internvl_config['vision_config']['image_size']
            self.pad2square = internvl_config['pad2square']
            self.min_dynamic_patch = internvl_config['min_dynamic_patch']
            self.max_dynamic_patch = internvl_config['max_dynamic_patch']
            self.use_thumbnail = internvl_config['use_thumbnail']
            patch_size = internvl_config['vision_config']['patch_size']
            downsample_ratio = internvl_config['downsample_ratio']
            self.num_image_token = int((self.image_size // patch_size)**2 * (downsample_ratio**2))

        super().__init__(data_files, tokenizer, config, processor)
        
    def _read_files_and_tokenize(self):
        self.dataframe = []
        for dataset_name, dataset_info in self.meta_info.items():
            data_list = self.tcs_loader(dataset_info['annotation'], file_type='jsonl')
            # assert len(data_list) == dataset_info['length'], f"dataset {dataset_name} has {len(data_list)} samples, but meta file says there are {dataset_info['length']} samples"

            # process repeat_time
            full_repeats = data_list * int(dataset_info['repeat_time'])
            extra_samples = random.sample(data_list, int(len(data_list) * (dataset_info['repeat_time'] - int(dataset_info['repeat_time']))))
            data_list = full_repeats + extra_samples

            for d in data_list:
                d = d.copy()
                d['data_source'] = dataset_name
                d['id'] = f"{dataset_name}-{d.get('index', 0)}"

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
            if self.input_template:
                self.dataframe = [doc for doc in self.dataframe if len(
                    tokenizer.encode(self.input_template.format(doc[self.prompt_key][0]['content']))) <= self.max_prompt_length]
            else:
                self.dataframe = [doc for doc in self.dataframe if len(
                    tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)) <= self.max_prompt_length]

            print(f'filter dataset len: {len(self.dataframe)}')


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = copy.deepcopy(self.dataframe[item])
        if self.apply_internvl_preprocess:
            try:
                return self._internvl_preprocess(row_dict)
            except UnidentifiedImageError:
                print(f"Find broken image: {row_dict['id']}")
                return self.__getitem__((item + 1) % len(self.dataframe))
            except NotImplementedError:
                print(f"Too long prompt: {row_dict['id']}")
                return self.__getitem__((item + 1) % len(self.dataframe))
        else:
            return self._default_preprocess(row_dict)

    def _build_messages(self, messages: dict, images: list):
        for message in messages:
            text = message["content"]
            content_list = []
            for _ in range(text.count("<image>")):
                image = images.pop(0)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    },
                })
            content_list.append({"type": "text", "text": text})
            message["content"] = content_list
        assert len(images) == 0

        return messages


    @use_image_bytes
    def _internvl_preprocess(self, row_dict):
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']
        chat = row_dict.pop(self.prompt_key)

        if self.input_template:
            question = chat[0]["content"]
            prompt_with_chat_template = self.input_template.format(question)
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

        raw_prompt = prompt_with_chat_template

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:
            transform = build_transform(is_train=False, input_size=self.image_size, pad2square=self.pad2square)
            output_dict['multi_modal_data'] = {'image': [process_image(self.tcs_loader(image)) for image in row_dict.pop(self.image_key)]}

            # convert image to patches
            num_image = len(output_dict['multi_modal_data']['image'])
            images, num_tiles = [], []
            for image in output_dict['multi_modal_data']['image']:
                patches = dynamic_preprocess_v3(
                    image, 
                    min_num=self.min_dynamic_patch, 
                    # max_num=max(self.max_dynamic_patch // num_image, 1),
                    max_num=max(self.max_dynamic_patch, 1),
                    image_size=self.image_size, 
                    use_thumbnail=self.use_thumbnail)
                images.extend(patches)
                num_tiles.append(len(patches))
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            num_patches = pixel_values.size(0)
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
            assert sum(num_tiles) == pixel_values.size(0), f"{sum(num_tiles)=}, {pixel_values.size(0)=}"

            # replace <image> with image tokens
            num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
            assert len(num_image_tokens) == prompt_with_chat_template.count('<image>'), f"{len(num_image_tokens)=}, {prompt_with_chat_template.count('<image>')=}"
            for i in range(len(num_image_tokens)):
                prompt_with_chat_template = prompt_with_chat_template.replace('<image>', f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens[i]}{IMG_END_TOKEN}", 1)

            # build multi_modal_inputs
            output_dict['multi_modal_inputs'] = {
                'pixel_values': pixel_values,
                'image_flags': image_flags
            }
        else:
            output_dict['multi_modal_data'] = None
            output_dict['multi_modal_inputs'] = {
                'pixel_values': torch.zeros(1, 3, self.image_size, self.image_size), 
                'image_flags': torch.zeros(1, dtype=torch.long)
            }
        
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

        if is_multi_modal:
            messages = self._build_messages(chat, copy.deepcopy(output_dict['multi_modal_data']['image']))
        else:
            messages = chat

        # encode prompts without chat template
        if self.return_raw_chat:
            output_dict['raw_prompt'] = messages

        # add index for each prompt
        output_dict["index"] = row_dict['id']

        return output_dict


    def _default_preprocess(self, row_dict):
        chat = row_dict.pop(self.prompt_key)

        if self.input_template:
            question = chat[0]["content"]
            prompt_with_chat_template = self.input_template.format(question)
        else:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(self.tcs_loader(image)) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
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

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict["prompt_length"] = row_dict['position_ids'].max().item() + 1

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict