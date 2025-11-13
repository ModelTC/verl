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
from verl.utils.dataset.utils import AUDIO_START_TOKEN, AUDIO_CONTEXT_TOKEN, AUDIO_END_TOKEN
from verl.utils.dataset.image_processors import pil_to_bytes
from verl.utils.dataset.audio_processors import librosa_audio_to_base64
import datasets

from transformers import WhisperFeatureExtractor
import librosa
import numpy as np


from verl.utils.dataset.rl_dataset_json_v2 import RLHFJSONDataset, format_prompt_fns, process_image, use_image_bytes

class SenseOmniDataset(RLHFJSONDataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        
        self.audio_key = config.get('audio_key', 'audio')

        # omni model setting
        self.apply_omnimodel = config.get('apply_omnimodel', False)

        try:
            internvl_config_file = config['internvl_config_file']
        except KeyError:
            raise ValueError("`internvl_config_file` is required when `apply_internvl_preprocess` is True")
        assert os.path.exists(internvl_config_file), f"internvl_config_file {internvl_config_file} not found"
        internvl_config = json.load(open(internvl_config_file, 'r'))
        
        self.audio_feature_extractor = WhisperFeatureExtractor(**internvl_config['audio_preprocessor_config'])

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
                
                # process audio path
                if self.audio_key in d:
                    audio = d[self.audio_key]
                    if not isinstance(audio, list):
                        audio = [audio]
                    if dataset_info['audio_path'] is not None:
                        audio = [os.path.join(dataset_info['audio_path'], i) for i in audio]
                    d[self.audio_key] = audio

                assert 'input_template' in dataset_info, f"input_template not found in dataset_info"
                d['input_template'] = dataset_info['input_template']

                if (
                    'reward_fn' in dataset_info and
                    'reward_fn' not in d['reward_model']
                ):
                    d['reward_model']['reward_fn'] = dataset_info['reward_fn']

                assert 'reward_fn' in d['reward_model'], \
                    f"reward_model.reward_fn is not defined for {d['id']}"

                d['max_turns'] = dataset_info.get('max_turns', -1)
                d['single_response_length'] = dataset_info.get('single_response_length', -1)

                self.dataframe.append(d)

        print(f'dataset len: {len(self.dataframe)}')

        # self.dataframe = datasets.Dataset.from_list(self.dataframe)
        # self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = copy.deepcopy(self.dataframe[item])
        preprocess_fn = self.sense_omni_preprocess
        try:
            output_dict = preprocess_fn(row_dict)
            output_dict['max_turns'] = row_dict['max_turns']
            output_dict['single_response_length'] = row_dict['single_response_length']
            return output_dict
        except Exception as e:
            print(f"Error: {e}\ndata id: {row_dict['id']}")
            return self.__getitem__((item + 1) % len(self.dataframe))


    def _build_messages(self, messages: dict, images: Optional[List] = None, audios: Optional[List] = None):
        for message in messages:
            content = message["content"]
            content_list = []
            for segment in re.split("(<image>|<audio>)", content):
                if segment == "":
                    continue
                elif segment == "<image>":
                    image = images.pop(0)
                    wh = image.size
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                    
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}",
                            "w": wh[0],
                            "h": wh[1]
                        },
                    })
                    content_list.append({"type": "text", "text": "<image>"})  # TODO 

                elif segment == "<audio>":
                    audio = audios.pop(0)
                    audio_base64 = librosa_audio_to_base64(audio)
                    audio_process_values = self.processor.process_audio(audio)
                    audio_token_num = audio_process_values['audio_token_num'].item()
                    content_list.append({
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{audio_base64}",
                            "audio_token_num": audio_token_num
                        },
                    })
                    content_list.append({"type": "text", "text": "<audio>"})  # TODO audio place holder
                    
                else:
                    content_list.append({"type": "text", "text": segment})

            message["content"] = content_list

        assert len(images) == 0

        return messages


    @use_image_bytes
    def sense_omni_preprocess(self, row_dict):
        output_dict = {}
        output_dict['reward_model'] = row_dict['reward_model']
        output_dict['data_source'] = row_dict['data_source']
        chat = row_dict.pop(self.prompt_key)

        format_prompt_fn = format_prompt_fns[row_dict['input_template']['name']]
        input_template_kwargs = row_dict['input_template'].get('arguments', {})
        prompt_with_chat_template = format_prompt_fn(self.tokenizer, chat, row_dict.get(self.image_key, None), **input_template_kwargs) # will change chat inplace
        raw_prompt = prompt_with_chat_template
        # raw_prompt = raw_prompt.replace('<image>', '<img></img>')

        if self.image_key in row_dict and "<image>" in prompt_with_chat_template: 
            # transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square)
            output_dict['multi_modal_data'] = {'image': [process_image(self.tcs_loader(image)) for image in row_dict.pop(self.image_key)]}

            # convert image to patches
            num_image = len(output_dict['multi_modal_data']['image'])
            image_inputs = self.processor.image_processor(
                output_dict['multi_modal_data']['image'],
                min_num=self.min_dynamic_patch,
                max_num=max(self.max_dynamic_patch // num_image, 1),
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail
            )
            num_patches = image_inputs['image_num_patches']
            pixel_values = image_inputs['pixel_values_flat']
            image_flags = image_inputs['image_flags']
            assert sum(num_patches) == pixel_values.size(0), f"{sum(num_patches)=}, {pixel_values.size(0)=}"

            # replace <image> with image tokens
            num_image_tokens = [self.num_image_token * num_patch for num_patch in num_patches]
            assert len(num_image_tokens) == prompt_with_chat_template.count('<image>'), f"{len(num_image_tokens)=}, {prompt_with_chat_template.count('<image>')=}"
            for i in range(len(num_image_tokens)):
                prompt_with_chat_template = prompt_with_chat_template.replace('<image>', f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_tokens[i]}{IMG_END_TOKEN}", 1)

            # build multi_modal_inputs
            output_dict['multi_modal_inputs'] = {
                'pixel_values': pixel_values,
                'image_flags': image_flags
            }
        else:
            output_dict['multi_modal_data'] = {'image': []}
            output_dict['multi_modal_inputs'] = {}

        if self.audio_key in row_dict and "<audio>" in prompt_with_chat_template:
            output_dict['multi_modal_data']['audio'] = [self.tcs_loader(audio, file_type='audio') for audio in row_dict.pop(self.audio_key)]

            num_audio = len(output_dict['multi_modal_data']['audio'])
            audio_values = []
            audio_len_after_cnn = []
            audio_token_num = []
            for audio in output_dict['multi_modal_data']['audio']:
                audio_process_values = self.processor.process_audio(audio)
                audio_values.append(audio_process_values['input_features'])
                audio_len_after_cnn.append(audio_process_values['audio_len_after_cnn'])
                audio_token_num.append(audio_process_values['audio_token_num'])

            audio_flags= torch.tensor([1] * num_audio, dtype=torch.long)

            # replace <audio> with audio tokens
            assert len(audio_token_num) == prompt_with_chat_template.count('<audio>'), f"{len(audio_token_num)=}, {prompt_with_chat_template.count('<audio>')=}"
            audio_tokens_list = list()
            for index in range(len(audio_token_num)):
                audio_token_num_i = audio_token_num[index]
                audio_tokens = AUDIO_START_TOKEN + AUDIO_CONTEXT_TOKEN * audio_token_num_i + AUDIO_END_TOKEN
                audio_tokens_list.append(audio_tokens)

            audio_tokens_iter = iter(audio_tokens_list)

            prompt_with_chat_template = re.sub(r"<audio>", lambda match:next(audio_tokens_iter), prompt_with_chat_template)

        else:
            output_dict['multi_modal_data']['audio'] = []

            audio = np.zeros((48000,), dtype=np.float32) # NOTE: too long audio will be truncated
            audio_process_values = self.processor.process_audio(audio)
            audio_values = [audio_process_values['input_features']]
            audio_len_after_cnn = [audio_process_values['audio_len_after_cnn']]
            audio_token_num = [audio_process_values['audio_token_num']]
            num_audio = 1
            audio_flags= torch.tensor([0] * num_audio, dtype=torch.long)

        audio_values = torch.cat(audio_values, dim=0)  # [num_audio, 128, 3000]
        audio_len_after_cnn = torch.stack(audio_len_after_cnn, dim=0)  # [num_audio]
        audio_token_num = torch.stack(audio_token_num, dim=0)  # [num_audio]

        # build multi_modal_inputs
        audio_ret = dict(
            audio_values=audio_values,
            audio_len_after_cnn=audio_len_after_cnn,
            audio_token_num=audio_token_num,
            audio_flags=audio_flags,
        )

        # build multi_modal_inputs
        output_dict['multi_modal_inputs'] = output_dict['multi_modal_inputs'] | audio_ret


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


        messages = self._build_messages(chat, copy.deepcopy(output_dict['multi_modal_data']['image']), copy.deepcopy(output_dict['multi_modal_data']['audio']))


        # encode prompts without chat template
        if self.return_raw_chat:
            output_dict['raw_prompt'] = messages

        # add index for each prompt
        output_dict["index"] = row_dict['id']

        return output_dict
