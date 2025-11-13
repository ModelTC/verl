# --------------------------------------------------------
# SenseOmni
# Copyright (c) 2025 SenseOmni Team
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union
import re
import json
import math
import librosa
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (GenerationConfig, Qwen3ForCausalLM, WhisperFeatureExtractor)
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM, Qwen3MoeForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_senseomni import SenseOmniConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .modeling_whisper import AudioWhisperModel
from .conversation import get_conv_template
from verl.utils.dataset.audio_processors import base64_to_numpy_audio
from verl.utils.dataset.image_processors import base64_to_pil

logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
AUDIO_START_TOKEN = '<audio>'
AUDIO_END_TOKEN = '</audio>'
AUDIO_CONTEXT_TOKEN = '<AUDIO_CONTEXT>'


class SenseOmniModel(PreTrainedModel):
    config_class = SenseOmniConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'AudioWhisperModel', 'Qwen3DecoderLayer']

    def __init__(self, config: SenseOmniConfig, vision_model=None, language_model=None, audio_model=None, use_flash_attn=True):
        super().__init__(config)

        image_size = config.force_image_size or config.vision_config.image_size
        self.image_size = image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        
        # 音频特征提取
        self.audio_feature_extractor = WhisperFeatureExtractor(**config.audio_preprocessor_config)
        self.transform = self.build_transform(input_size=image_size)

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if audio_model is not None:
            self.audio_model = audio_model
        else:
            self.audio_model = AudioWhisperModel(config.audio_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        audio_hidden_size = config.audio_config.d_model
        llm_hidden_size = config.llm_config.hidden_size

        # vision mlp 
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # audio mlp
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(audio_hidden_size),
            nn.Linear(audio_hidden_size, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.audio_context_token_id = None
        self.neftune_alpha = None
    
    def freeze(self, config):
        modules = []
        if getattr(config, "freeze_vision_model", False):
            print(f'!!! freeze vision model')
            modules.append(self.vision_model)
        if getattr(config, "freeze_audio_model", False):
            print(f'!!! freeze audio model')
            modules.append(self.audio_model)
        if getattr(config, "freeze_mlp", False):
            print(f'!!! freeze mlp')
            modules.append(self.mlp1)
            modules.append(self.mlp2)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            audio_values: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            audio_flags: Optional[torch.LongTensor] = None,
            audio_len_after_cnn: Optional[torch.LongTensor] = None,
            audio_token_num: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cu_seqlens: Optional[torch.IntTensor] = None,
            loss_weight = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        # input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.shape[0])
            input_embeds[:n_token] = input_embeds[:n_token] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        if torch.distributed.get_rank() == 0:
            audio_batch_size = audio_values.shape[0]
            print(f'audio batch size: {audio_batch_size}, audios per sample: {audio_batch_size / B}')

        audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)  # (audio_num, n_frame, C)

        # v1
        # output_audios = []
        # for i in range(len(audio_token_num)):
        #     if audio_flags[i] > 0:
        #         token_num = int(audio_token_num[i].item())
        #         audio = audio_embeds[i][:token_num]   # 提取有效的token
        #         output_audios.append(audio)

        # if len(output_audios):
        #     output_audios = torch.cat(output_audios, dim=0)
        #     audio_selected = (input_ids == self.audio_context_token_id)
        #     input_embeds[audio_selected] = input_embeds[audio_selected] * 0.0 + output_audios.reshape(-1, C)
        
        
        # v2: for zero3
        # 创建一个 mask 来选择 audio_flags > 0 的样本
        valid_audio_mask = (audio_flags > 0)  # (audio_num,)
        # any(valid_audio_mask)
        valid_audio_indices = valid_audio_mask.nonzero(as_tuple=True)[0]  # 提取有效的索引
        if valid_audio_indices.numel() == 0:
            audio_embeds = audio_embeds.reshape(-1, C)
            selected = (input_ids == self.audio_context_token_id)
            n_token = min(input_embeds.shape[0], audio_embeds.shape[0])
            input_embeds[:n_token] = input_embeds[:n_token] + audio_embeds[:n_token] * 0.0 
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'missing audio input for this train sample')
            
            pass
        else:
            # 根据 valid_audio_indices 提取有效的 audio_embeds 和 audio_token_num
            valid_audio_embeds = audio_embeds[valid_audio_indices]  # 只选择有效的样本  (audio_sample_num, n_frame, C)
            valid_audio_token_num = audio_token_num[valid_audio_indices]  # 对应的 token_num  (audio_sample_num)

            # 对每个样本提取有效的 token
            max_token_num = valid_audio_token_num.max().item()  # 找到最大 token 数量
            valid_audio_embeds = valid_audio_embeds[:, :max_token_num, :]  # 截取到最大 token 数量
            mask = torch.arange(max_token_num, device=valid_audio_token_num.device).unsqueeze(0) < valid_audio_token_num.unsqueeze(1)  # (audio_sample_num, max_token_num)
            output_audios = valid_audio_embeds[mask].reshape(-1, audio_embeds.size(-1))  # 合并有效 token

            # 将处理后的 audio 数据嵌入到 input_embeds
            audio_selected = (input_ids == self.audio_context_token_id)
            input_embeds[audio_selected] = input_embeds[audio_selected] * 0.0 + output_audios

        input_embeds = input_embeds.reshape(B, N, C)
        # import pdb; pdb.set_trace()
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask if cu_seqlens is None else cu_seqlens,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if cu_seqlens is None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                if ignore_flag:
                    loss = loss * 0.0
            else:
                assert labels.shape[0] == 1 and loss_weight is not None
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits, shift_labels)
                shift_weights = torch.FloatTensor(loss_weight[0][1:]).contiguous().to(shift_logits.device)
                shift_weights_sum = shift_weights.sum()
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)
                # import pdb 
                # pdb.set_trace()
                loss = loss * shift_weights
                if shift_weights_sum < 1:
                    print(f'warning: shift_weights_sum too small: {shift_weights_sum}')
                loss = loss.sum() / (shift_weights_sum + 1e-6)  # for very extramly case that shift_weights_sum=0, which should never happen
                if ignore_flag:
                    loss = loss * 0.0
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'calc logit finished! logits.shape: {logits.shape}, loss: {loss}')
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

        return transform
    
    def find_closest_aspect_ratio(self, image, min_num=1, max_num=6, image_size=448):
        assert min_num == 1
        original_width, original_height = image.size
        log_ratio = math.log(original_width / original_height)
        ratio = original_width * original_height / (image_size * image_size)
        multiple = min(math.ceil(ratio), max_num)
        if multiple <= 1:
            return [1, 1]
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i > max_num:
                continue
            candidate_split_grids_nums.append(i)
        
        candidate_grids = []
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1
        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        return best_grid

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        target_aspect_ratio = self.find_closest_aspect_ratio(image, min_num, max_num, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image, input_size=448, max_num=12):
        if image.startswith("data:image/") and ";base64," in image:
            image_base64 = image.split(";base64,")[1]
            image = base64_to_pil(image_base64)
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        return images

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.training and self.neftune_alpha is not None:
            vit_embeds = self.noised_embed(vit_embeds, self.neftune_alpha)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)#.to(pixel_values.device)
        return vit_embeds

    def get_T_after_cnn(self, L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def process_audio(self, audio, return_tensors, sampling_rate=16000):
        L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
        mel_len = L // 160
        audio_len_after_cnn = self.get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        inputs = self.audio_feature_extractor(audio, return_tensors=return_tensors, sampling_rate=sampling_rate)
        inputs['audio_len_after_cnn'] = torch.tensor(audio_len_after_cnn, dtype=torch.long)
        inputs['audio_token_num'] = torch.tensor(audio_token_num, dtype=torch.long)
        return inputs

    def load_audio(self, audio_file, sampling_rate=16000):
        if audio_file.startswith("data:audio/") and ";base64," in audio_file:
            audio_base64 = audio_file.split(";base64,")[1]
            audio_values, _ = base64_to_numpy_audio(audio_base64)
        else:
            audio_values, _ = librosa.load(audio_file, sr=sampling_rate) # sample rate should be 16000

        audio_process_values = self.process_audio(audio_values, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = audio_process_values['input_features']
        audio_len_after_cnn = audio_process_values['audio_len_after_cnn']
        audio_token_num = audio_process_values['audio_token_num']

        audio_input_dict = {'audio_values': input_features,
                        'audio_len_after_cnn': audio_len_after_cnn,
                        'audio_token_num': audio_token_num,
                        }
        return audio_input_dict
    
    def extract_audio_feature(self, audio_values, audio_len_after_cnn):

        audio_values = audio_values.squeeze(1)
        max_len_in_batch = int(torch.max(audio_len_after_cnn).item())
        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch]).to(dtype=audio_values.dtype, device=audio_values.device)
        for index in range(len(audio_values)):
            padding_mask[index, :int(audio_len_after_cnn[index].item())] = 0

        last_hidden_state = self.audio_model(audio_values, padding_mask, audio_len_after_cnn)  # (bs, max_token_num, 1280)

        audio_embeds = self.mlp2(last_hidden_state)

        return audio_embeds
    
    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
        
    def load_video(self, video_path, bound=None, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        frames = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            frames.append(img)
        return frames


    @torch.no_grad()
    def generate(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.FloatTensor,
            attention_mask: torch.LongTensor,
            visual_features: Optional[torch.FloatTensor] = None,
            audio_values: Optional[torch.FloatTensor] = None,
            audio_len_after_cnn: Optional[bool] = None,
            audio_token_num: Optional[bool] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:
        
        assert self.img_context_token_id is not None
        assert self.audio_context_token_id is not None

        vit_embeds = None
        if visual_features is not None:
            vit_embeds = visual_features
        elif pixel_values is not None:
            vit_embeds = self.extract_feature(pixel_values)
        
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        
        input_ids = input_ids.reshape(B * N)

        if vit_embeds is not None:
            selected = (input_ids == self.img_context_token_id)
            input_embeds[selected] = vit_embeds.reshape(-1, C)

        if audio_values is not None and audio_len_after_cnn is not None and audio_token_num is not None:
            audio_embeds = self.extract_audio_feature(audio_values, audio_len_after_cnn)
            output_audios = []
            for i in range(len(audio_token_num)):
                token_num = int(audio_token_num[i].item())
                audio = audio_embeds[i][:token_num]
                output_audios.append(audio)
            output_audios = torch.cat(output_audios, dim=0)
            selected = (input_ids == self.audio_context_token_id)
            input_embeds[selected] = output_audios.reshape(-1, C)
        
        input_embeds = input_embeds.reshape(B, N, C)
        
        print(generate_kwargs)
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        
        return outputs

    def chat(self, tokenizer, generation_config, messages, max_patch_num=12, frame=8, generate_audio=False, speaker="Taozi", print_flag=True):
        if messages is None or len(messages) == 0:
            raise RuntimeError('no messages')
        role_transfer_dict = {
            'system': ['user'],
            'user': ['assistant'],
            'assistant': ['user'],
        }

        first_role = ['system', 'user']
        last_role = ['user']
        if messages[-1]['role'] not in last_role:
            raise RuntimeError(f"last role error, expect {last_role}, but got {messages[-1]}")
        
        current_role = None
        dynamic_images = list()
        dynamic_nums = list()
        audio_values = list()
        audio_len_after_cnn = list()
        audio_token_num = list()
        template = get_conv_template(self.template)
        for index in range(len(messages)):
            text = ''
            audios = list()
            images = list()
            message = messages[index]
            if index == 0:
                if message['role'] not in first_role:
                    raise RuntimeError(f'first role error expect {first_role}, but got {message}')
            else:
                if message['role'] not in current_role:
                    raise RuntimeError(f'role error expect {current_role}, but got {message}')
            current_role = message['role']
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item['type'] == 'text':
                        if item.get('text', None) is None:
                            continue
                        text += item['text']
                    elif item['type'] == 'audio':
                        if item.get('audio', None) is None:
                            continue
                        if type(item['audio']) is list:
                            assert len(item['audio']) == 1, f'only support 1 audio file in round, but got {item["audio"]}'
                            audio = item['audio'][0]
                        else:
                            audio = item['audio']
                        audios.append(audio)

                    elif item['type'] == 'audio_url':  # audio base64
                        audio = item['audio_url']['url']
                        if type(item['audio_url']['url']) is list:
                            assert len(item['audio_url']['url']) == 1, f'only support 1 audio file in round, but got {item["audio_url"]["url"]}'
                            audio = item['audio_url']['url'][0]
                        else:
                            audio = item['audio_url']['url']
                        audios.append(audio)

                    elif item['type'] == 'image':
                        if item.get('image', None) is None:
                            continue
                        if type(item['image']) is not list:
                            images.append(item['image'])
                        else:
                            images.extend(item['image'])

                    elif item['type'] == 'image_url':  # image base64
                        if item.get('image_url', None) is None:
                            continue
                        if type(item['image_url']['url']) is not list:
                            images.append(item['image_url']['url'])
                        else:
                            images.extend(item['image_url']['url'])

                    elif item['type'] == 'video':
                        if item.get('video', None) is None:
                            continue
                        if type(item['video']) is list:
                            assert len(item['video']) == 1, f'only support 1 video file in round, but got {item["video"]}'
                            video = item['video'][0]
                        else:
                            video = item['video']
                        frames = self.load_video(video, num_segments=frame)
                        images.extend(frames)
            else:
                assert isinstance(message["content"], str), message["content"]
                text = message["content"]

            if len(audios) != 0:
                assert len(audios) == 1, f'only support 1 audio file in round, but got {audios}'
                if '<audio>' in text:
                    matches = re.findall(r"<audio>", text)
                    assert len(matches) == len(audios), f'<audio> error {text} {len(audios)}' + text
                    text = re.sub(r'(<audio>)(?!\n)', r'\1\n', text)
                else:
                    text = '<audio>\n'*len(audios) + text

                audio_path = audios[0]
                audio_input_dict = self.load_audio(audio_path)
                assert audio_input_dict['audio_token_num'].item() != 0, f'audio_token_num of {audio_path} is 0.'
                audio_values.append(audio_input_dict['audio_values'])
                audio_len_after_cnn.append(audio_input_dict['audio_len_after_cnn'])
                audio_token_num.append(audio_input_dict['audio_token_num'])

            if images is not None:
                if '<image>' in text:
                    matches = re.findall(r"<image>", text)
                    assert len(matches) == len(images), f'<image> error {text} {len(images)}' + text
                    text = re.sub(r'(<image>)(?!\n)', r'\1\n', text)
                else:
                    text = '<image>\n'*len(images) + text

                for image in images:
                    dynamic_image = self.load_image(image, max_num=max_patch_num)
                    dynamic_images += dynamic_image
                    dynamic_nums.append(len(dynamic_image))
            
            if message['role'] == 'system':
                template.set_system_message(text)
            elif message['role'] == 'user':
                template.append_message(template.roles[0], text)
            elif message['role'] == 'assistant':
                template.append_message(template.roles[1], text)
            else:
                raise ValueError('unexpected role')

            current_role = role_transfer_dict[current_role]

        template.append_message(template.roles[1], None)
        
        if len(audio_values) != 0:
            audio_values = torch.cat(audio_values, dim=0).to(dtype=self.dtype).cuda()  # [num_audio, 128, 3000]
            audio_len_after_cnn = torch.stack(audio_len_after_cnn, dim=0)  # [num_audio]
            audio_token_num = torch.stack(audio_token_num, dim=0)  # [num_audio]
        else:
            audio_values = None
            audio_len_after_cnn = None
            audio_token_num = None

        if len(dynamic_images) != 0:
            pixel_values = [self.transform(image) for image in dynamic_images]
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
        else:
            pixel_values = None
            dynamic_nums = None
        
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        audio_context_token_id = tokenizer.convert_tokens_to_ids(AUDIO_CONTEXT_TOKEN)
        self.audio_context_token_id = audio_context_token_id

        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

        query = template.get_prompt()

        if audio_values is not None:
            if print_flag:
                print(f'audio num: {len(audio_token_num)}')
            audio_tokens_list = list()
            for index in range(len(audio_token_num)):
                audio_token_num_i = audio_token_num[index]
                if print_flag:
                    print(f'audio_token_num: {audio_token_num_i}')
                audio_tokens = AUDIO_START_TOKEN + AUDIO_CONTEXT_TOKEN * audio_token_num_i + AUDIO_END_TOKEN
                audio_tokens_list.append(audio_tokens)

            audio_tokens_iter = iter(audio_tokens_list)

            query = re.sub(r"<audio>", lambda match:next(audio_tokens_iter), query)

        if pixel_values is not None:
            if print_flag:
                print(f'image num: {len(dynamic_nums)}')
            image_tokens_list = list()
            total_dynamic_num = 0
            for index in range(len(dynamic_nums)):
                dynamic_num = dynamic_nums[index]
                total_dynamic_num += dynamic_num
                if print_flag:
                    print(f'dynamic ViT batch size: {dynamic_num}')
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * dynamic_num + IMG_END_TOKEN
                image_tokens_list.append(image_tokens)
            assert total_dynamic_num == pixel_values.shape[0], f'dynamic num not equal, {total_dynamic_num}, {pixel_values.shape[0]}'

            image_tokens_iter = iter(image_tokens_list)

            query = re.sub(r"<image>", lambda match:next(image_tokens_iter), query)

        model_inputs = tokenizer(query, return_tensors='pt', add_special_tokens=False)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            audio_values=audio_values,
            audio_len_after_cnn=audio_len_after_cnn,
            audio_token_num=audio_token_num,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        response = response.split("<|im_end|>")[0].replace('<|endoftext|>', '').strip()
        query_to_print = query
        if pixel_values is not None:
            query_to_print = query_to_print.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
        if audio_values is not None:
            query_to_print = query_to_print.replace(AUDIO_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{AUDIO_START_TOKEN}{AUDIO_END_TOKEN}', '<audio>')
        if print_flag:
            print('query: ' + json.dumps(query_to_print, ensure_ascii=False))
            print('response: ' + response)

        return response