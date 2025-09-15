import os
from typing import Optional
import json
from transformers import PreTrainedTokenizer, ProcessorMixin
import torch
import random
random.seed(42)
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
from verl.utils.dataset.utils import dynamic_preprocess_v4, build_transform, IMG_START_TOKEN, IMG_CONTEXT_TOKEN, IMG_END_TOKEN
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


def format_prompt_general(
    tokenizer,
    prompts,
    images=None,
    **kwargs
):
    assert isinstance(prompts, list)

    if 'system_prompt' in kwargs:
        for p in prompts:
            assert p['role'] != 'system', f"system prompt already exists in prompts"
        prompts.insert(0, {'role': 'system', 'content': kwargs['system_prompt']})

    if 'format_instruction' in kwargs:
        for p in prompts:
            if p['role'] == 'user':
                break 
        p['content'] = p['content'].rstrip() + '\n' + kwargs['format_instruction']

    return tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)

def format_prompt_agentic(
    tokenizer,
    prompts,
    images=None,
    **kwargs
):
    assert isinstance(prompts, list)
    resource_type = kwargs.get("resource_type", "Image")

    if 'system_prompt' in kwargs:
        for p in prompts:
            assert p['role'] != 'system', f"system prompt already exists in prompts"
        prompts.insert(0, {'role': 'system', 'content': kwargs['system_prompt']})

    if images is not None:
        if isinstance(images, str):
            images = [images]
        context = []
        for i, img_path in enumerate(images):
            if resource_type.lower() == "image":
                context.append(f"![{resource_type}{i+1}]({img_path})")
            else:
                context.append(f"[{resource_type}{i+1}]({img_path})")

        for p in prompts:
            if p['role'] == 'user':
                break

        assert len(context) == p['content'].count('<image>'), \
            f"Number of images ({len(context)}) does not match the number of <image> placeholders ({p['content'].count('<image>')}) in the prompt."
        for resource_path in context:
            p['content'] = p['content'].replace('<image>', resource_path, 1)
        
    if 'format_instruction' in kwargs:
        for p in prompts:
            if p['role'] == 'user':
                break 
        p['content'] = p['content'].rstrip() + '\n' + kwargs['format_instruction']

    return tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)


def format_prompt_agentic_legacy(
    tokenizer,
    prompts,
    images=None,
    **kwargs
):
    assert isinstance(prompts, list)
    resource_type = kwargs.get("resource_type", "Image")

    if 'system_prompt' in kwargs:
        for p in prompts:
            assert p['role'] != 'system', f"system prompt already exists in prompts"
        prompts.insert(0, {'role': 'system', 'content': kwargs['system_prompt']})

    if images is not None:
        if isinstance(images, str):
            images = [images]
        context = []
        for i, img_path in enumerate(images):
            if resource_type.lower() == "image":
                context.append(f"![{resource_type}{i}]({img_path})")
            else:
                context.append(f"[{resource_type}{i}]({img_path})")

        for p in prompts:
            if p['role'] == 'user':
                break
        p['content'] = '\n'.join(context) + '\n' + p['content'].rstrip()

    if 'format_instruction' in kwargs:
        for p in prompts:
            if p['role'] == 'user':
                break 
        p['content'] = p['content'].rstrip() + '\n' + kwargs['format_instruction']

    return tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)



format_prompt_fns = {
    'general': format_prompt_general,
    'agentic': format_prompt_agentic,
    # 'agentic': format_prompt_agentic_legacy
}


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

        filename = os.path.basename(self.meta_file)
        if "train" in filename:
            self.is_train = True
        elif "test" in filename:
            self.is_train = False
        else:
            raise Exception(f"filename({filename}) not contain 'train' or 'test'")

        super().__init__(data_files, tokenizer, config, processor)


    def _read_files_and_tokenize(self):
        self.dataframe = []
        for dataset_name, dataset_info in self.meta_info.items():
            data_list = self.tcs_loader(dataset_info['annotation'], file_type='jsonl')
            assert len(data_list) == dataset_info['length'], f"dataset {dataset_name} has {len(data_list)} samples, but meta file says there are {dataset_info['length']} samples"

            # process repeat_time
            full_repeats = data_list * int(dataset_info['repeat_time'])
            extra_samples = random.sample(data_list, int(len(data_list) * (dataset_info['repeat_time'] - int(dataset_info['repeat_time']))))
            data_list = full_repeats + extra_samples

            for i, d in enumerate(data_list):
                d = d.copy()
                d['data_source'] = dataset_name
                d['id'] = f"{dataset_name}-{d.get('index', i)}"

                # process image path
                if dataset_info['root'] is not None and self.image_key in d:
                    image = d[self.image_key]
                    if not isinstance(image, list):
                        image = [image]
                    image = [os.path.join(dataset_info['root'], i) for i in image]
                    d[self.image_key] = image

                assert 'input_template' in dataset_info, f"input_template not found in dataset_info"
                d['input_template'] = dataset_info['input_template']
                assert 'reward_fn' in dataset_info, f"reward_fn not found in dataset_info"
                d['reward_model']['reward_fn'] = dataset_info['reward_fn']
                
                self.dataframe.append(d)

        print(f'dataset len: {len(self.dataframe)}')

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = copy.deepcopy(self.dataframe[item])
        preprocess_fn = self._internvl_preprocess if self.apply_internvl_preprocess else self._default_preprocess
        try:
            return preprocess_fn(row_dict)
        except Exception as e:
            print(f"Error: {e}\ndata id: {row_dict['id']}")
            return self.__getitem__((item + 1) % len(self.dataframe))

    def _build_messages(self, messages: dict, images: list):
        for message in messages:
            content = message["content"]
            content_list = []
            for segment in re.split("(<image>)", content):
                if segment == "":
                    continue
                elif segment == "<image>":
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
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        assert len(images) == 0

        return messages


    @use_image_bytes
    def _internvl_preprocess(self, row_dict):
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']
        chat = row_dict.pop(self.prompt_key)

        format_prompt_fn = format_prompt_fns[row_dict['input_template']['name']]
        input_template_kwargs = row_dict['input_template'].get('arguments', {})
        prompt_with_chat_template = format_prompt_fn(self.tokenizer, chat, row_dict.get(self.image_key, None), **input_template_kwargs)
        raw_prompt = prompt_with_chat_template

        is_multi_modal = self.image_key in row_dict and "<image>" in prompt_with_chat_template
        if is_multi_modal: 
            # transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square)
            output_dict['multi_modal_data'] = {'image': [process_image(self.tcs_loader(image)) for image in row_dict.pop(self.image_key)]}

            # convert image to patches
            num_image = len(output_dict['multi_modal_data']['image'])
            images, num_tiles = [], []
            for image in output_dict['multi_modal_data']['image']:
                patches = dynamic_preprocess_v4(
                    image, 
                    min_num=self.min_dynamic_patch, 
                    max_num=max(self.max_dynamic_patch // num_image, 1),
                    image_size=self.image_size, 
                    use_thumbnail=self.use_thumbnail)
                images.extend(patches)
                num_tiles.append(len(patches))
            # pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(images)
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
        # else:
        #     raw_prompt = prompt_with_chat_template
        
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
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']

        chat = row_dict.pop(self.prompt_key)
        images = row_dict.get(self.image_key, None)

        format_prompt_fn = format_prompt_fns[row_dict['input_template']['name']]
        input_template_kwargs = row_dict['input_template']['arguments']
        prompt_with_chat_template = format_prompt_fn(self.tokenizer, chat, images, **input_template_kwargs)
        raw_prompt = prompt_with_chat_template

        is_multi_modal = self.image_key in row_dict and "<image>" in prompt_with_chat_template
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            output_dict['multi_modal_data'] = {'image': [process_image(self.tcs_loader(image)) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(output_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            output_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

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