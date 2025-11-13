# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# --------------------------------------------------------
# SenseOmni
# Copyright (c) 2025 SenseOmni Team
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import copy
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional

import torch
from torch import nn
from transformers.utils import logging

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.internvl import InternVisionModel
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory, merge_multimodal_embeddings)
from vllm.model_executor.models.whisper import (WhisperEncoder,
                                                WhisperFeatureExtractor)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig,
                                    MultiModalKwargsOptionalItems,
                                    NestedTensors)
from vllm.multimodal.parse import (MultiModalDataItems, MultiModalDataParser,
                                   MultiModalKwargsItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo,
                                        MultiModalPromptUpdates,
                                        PlaceholderFeaturesInfo,
                                        PromptReplacement, PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import (BaseDummyInputsBuilder,
                                       MultiModalDataDict)
# from .conversation import get_conv_template
from vllm.sequence import IntermediateTensors
# from vllm.transformers_utils.processors.processing_senseomni import (
#     SenseOmniProcessor)
from transformers import ProcessorMixin

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, WeightsMapper

logger = logging.get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
AUDIO_START_TOKEN = "<audio>"
AUDIO_END_TOKEN = "</audio>"
AUDIO_CONTEXT_TOKEN = "<AUDIO_CONTEXT>"

has_flash_attn = True

def get_image_repl(
    processor,
    feature_size: int,
) -> PromptUpdateDetails[str]:
    repl_features = IMG_CONTEXT_TOKEN * feature_size
    repl_full = IMG_START_TOKEN + repl_features + IMG_END_TOKEN

    return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT_TOKEN)

def get_audio_repl(processor, audio_token_num: int):

    repl_features = AUDIO_CONTEXT_TOKEN * audio_token_num
    repl_full = AUDIO_START_TOKEN + repl_features + AUDIO_END_TOKEN
    return PromptUpdateDetails.select_text(repl_full, AUDIO_CONTEXT_TOKEN)

class SenseOmniProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None, "audio": None}

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        return self.ctx.get_hf_processor(ProcessorMixin, **kwargs)

    def get_feature_extractor(
        self,
        *,
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> WhisperFeatureExtractor:
        audio_processor_config = self.ctx.get_hf_config().audio_preprocessor_config
        cls_str = audio_processor_config.get("feature_extractor_type")
        assert cls_str in ["WhisperFeatureExtractor"]
        whisper_feature_extractor = WhisperFeatureExtractor(
            **audio_processor_config)
        return whisper_feature_extractor

    def get_image_size_with_most_features(self):
        # Get the SenseOmni config to access vision config
        hf_config = self.ctx.get_hf_config()
        image_size = hf_config.vision_config.image_size
        if isinstance(image_size, int):
            return (image_size, image_size)
        return image_size


class AudioWhisperModel(nn.Module):
    """
    overwrite forward to support attention_mask
    overwrite from_pretrained to support split encoder parameters
    from pretrained WhisperModel
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__()
        self.encoder = WhisperEncoder(vllm_config=vllm_config)
        self.avg_pooler = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, input_features):
        hidden_states = self.encoder(input_features)
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = self.avg_pooler(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2)
        return hidden_states

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2."
    }, )

    # copy from WhisperModel.load_weights in whisper.py
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        weights = self.hf_to_vllm_mapper.apply(weights)
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class SenseOmniMultimodalMlp(nn.Module):

    def __init__(
        self,
        in1_size,
        in2_size,
        out_size,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in1_size)
        self.linear1 = ColumnParallelLinear(in1_size,
                                            in2_size,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}")
        self.act = nn.GELU()
        self.linear2 = RowParallelLinear(
            in2_size,
            out_size,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}",
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.linear1(x)
        x = self.act(x)
        x, _ = self.linear2(x)
        return x


class SenseOmniMultiModalProcessor(
        BaseMultiModalProcessor[SenseOmniProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:

        image_token_num = hf_inputs.get("image_token_num", torch.empty(0))
        # num_images = len(image_num_patches)

        return dict(
            # pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_token_num),
            image_token_num=MultiModalFieldConfig.batched("image"),
            audio_values=MultiModalFieldConfig.batched("audio"),
            audio_len_after_cnn=MultiModalFieldConfig.batched("audio"),
            audio_token_num=MultiModalFieldConfig.batched("audio"),
        )

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    # copy from _maybe_apply_prompt_updates in processing.py
    # and delete is_update_applied code because this will replace
    # failed to apply prompt updates with empty mm_items
    # def _maybe_apply_prompt_updates(
    #     self,
    #     mm_items: MultiModalDataItems,
    #     prompt_ids: list[int],
    #     mm_kwargs: MultiModalKwargsOptionalItems,
    #     mm_prompt_updates: MultiModalPromptUpdates,
    #     is_update_applied: bool,
    # ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
    #     mm_item_counts = mm_items.get_all_counts()
    #     self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

    #     (
    #         prompt_ids,
    #         prompt,
    #         mm_placeholders,
    #     ) = self._apply_prompt_updates(
    #         prompt_ids,
    #         mm_prompt_updates,
    #     )

    #     print(f'!!! {prompt=}')
    #     self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

    #     return prompt_ids, prompt, mm_placeholders

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor: SenseOmniProcessor = self.info.get_hf_processor(
            **hf_processor_mm_kwargs)

        def get_replacement_image(item_idx: int):
            image_token_num = out_mm_kwargs["image"][item_idx][
                "image_token_num"].data.item()
            return get_image_repl(processor, feature_size=int(image_token_num))

        def get_replacement_audio(item_idx: int):
            audio_token_num = out_mm_kwargs["audio"][item_idx][
                "audio_token_num"].data.item()
            return get_audio_repl(processor, audio_token_num=int(audio_token_num))

        return [
            PromptReplacement(
                modality="image",
                target="<image>",  # Never match the prompt (see below note)
                replacement=get_replacement_image,
            ),
            PromptReplacement(
                modality="audio",
                target="<audio>",  # Never match the prompt (see below note)
                replacement=get_replacement_audio,
            ),
        ]


class SenseOmniDummyInputsBuilder(
        BaseDummyInputsBuilder[SenseOmniProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        return "<audio>" * num_audios + "<image>" * num_images

    def get_dummy_mm_data(self, seq_len: int,
                          mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        feature_extractor = self.info.get_feature_extractor()
        target_audio_length = (min(
            feature_extractor.chunk_length,
            30,
        ) * feature_extractor.sampling_rate)
        target_width, target_height = (
            self.info.get_image_size_with_most_features())
        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
            "audio":
            self._get_dummy_audios(length=target_audio_length,
                                   num_audios=num_audios),
        }


@MULTIMODAL_REGISTRY.register_processor(
    SenseOmniMultiModalProcessor,
    info=SenseOmniProcessingInfo,
    dummy_inputs=SenseOmniDummyInputsBuilder,
)
class SenseOmniForConditionalGeneration(nn.Module, SupportsPP,
                                        SupportsMultiModal):
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "AudioWhisperModel",
        "Qwen3DecoderLayer",
    ]

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # MLP 层的映射: Sequential 索引 -> 命名层
            "mlp1.0.": "mlp1.layer_norm.",
            "mlp1.1.": "mlp1.linear1.",
            "mlp1.3.": "mlp1.linear2.",
            "mlp2.0.": "mlp2.layer_norm.",
            "mlp2.1.": "mlp2.linear1.",
            "mlp2.3.": "mlp2.linear2.",
        }, )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """
        返回用户在 prompt 中使用的占位符字符串。

        SenseOmni 使用的占位符格式：
        - 图像：<image>
        - 音频：<audio>

        这些占位符会在处理时被替换为完整的 token 序列。
        """
        if modality.startswith("image"):
            return "<image>"
        if modality.startswith("audio"):
            return "<audio>"

        raise ValueError(f"Unsupported modality: {modality}. "
                         "Only 'image' and 'audio' are supported.")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        senseomni_config = vllm_config.model_config.hf_config
        self.vision_config = senseomni_config.vision_config
        self.audio_config = senseomni_config.audio_config
        self.text_config = senseomni_config.text_config

        # senseomni_config.vision_config.image_size = senseomni_config.vision_config.image_size[0]
        # senseomni_config.vision_config.patch_size = senseomni_config.vision_config.patch_size[0]
        patch_size = self.vision_config.patch_size
        image_size = (senseomni_config.force_image_size
                      or self.vision_config.image_size)
        self.select_layer = senseomni_config.select_layer
        self.template = senseomni_config.template
        self.num_image_token = int((image_size // patch_size)**2 *
                                   (senseomni_config.downsample_ratio**2))
        self.downsample_ratio = senseomni_config.downsample_ratio
        self.ps_version = senseomni_config.ps_version
        use_flash_attn = has_flash_attn
        self.vision_config.use_flash_attn = use_flash_attn
        self.text_config._attn_implementation = ("flash_attention_2" if
                                                 use_flash_attn else "eager")

        self.senseomni_config = senseomni_config
        # 音频特征提取

        logger.info("num_image_token: %s", self.num_image_token)
        logger.info("ps_version: %s", self.ps_version)

        # for internvl compatibility, image_size (h, w) and patch_size (p, p)
        # are stored in vision_config, but InternVLVisionModel expects
        # image_size and patch_size are scalar values
        self.vision_config = copy.deepcopy(self.vision_config)
        self.vision_config.image_size = image_size
        self.vision_config.patch_size = patch_size
        if self.select_layer == -1:
            self.vision_model = InternVisionModel(self.vision_config)
        else:
            self.vision_model = InternVisionModel(
                self.vision_config,
                num_hidden_layers_override=self.select_layer)
        self.vision_model = InternVisionModel(self.vision_config)

        audio_vllm_config = copy.deepcopy(vllm_config)
        audio_vllm_config.model_config.hf_config = self.audio_config
        self.audio_model = AudioWhisperModel(audio_vllm_config)

        text_vllm_config = copy.deepcopy(vllm_config)
        text_vllm_config.model_config.hf_config = self.text_config

        if self.text_config.architectures[0] == "LlamaForCausalLM":
            self.language_model = LlamaForCausalLM(
                vllm_config=text_vllm_config)
        elif self.text_config.architectures[0] == "Qwen2ForCausalLM":
            self.language_model = Qwen2ForCausalLM(
                vllm_config=text_vllm_config)
        elif self.text_config.architectures[0] == "Qwen3ForCausalLM":
            self.language_model = Qwen3ForCausalLM(
                vllm_config=text_vllm_config)
        elif self.text_config.architectures[0] == "Qwen3MoeForCausalLM":
            self.language_model = Qwen3MoeForCausalLM(
                vllm_config=text_vllm_config)
        else:
            raise NotImplementedError(
                f"{self.llm_config.architectures[0]} is not implemented.")

        vit_hidden_size = self.vision_config.hidden_size
        audio_hidden_size = self.audio_config.d_model
        llm_hidden_size = self.text_config.hidden_size
        self.mlp1 = SenseOmniMultimodalMlp(
            vit_hidden_size * int(1 / self.downsample_ratio)**2,
            llm_hidden_size,
            llm_hidden_size,
            prefix=f"{prefix}.mlp1",
        )
        self.mlp2 = SenseOmniMultimodalMlp(
            audio_hidden_size,
            llm_hidden_size,
            llm_hidden_size,
            prefix=f"{prefix}.mlp2",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    llm_hidden_size))

    def _parse_and_validate_image_input(self, **kwargs: object) -> dict:
        pixel_values = kwargs.pop("pixel_values", None)
        image_token_num = kwargs.pop("image_token_num", None)
        return {
            "pixel_values": pixel_values,
            "image_token_num": image_token_num,
        }

    def _parse_and_validate_audio_input(self, **kwargs: object) -> dict:
        audio_values = kwargs.pop("audio_values", None)
        audio_len_after_cnn = kwargs.pop("audio_len_after_cnn", None)
        audio_token_num = kwargs.pop("audio_token_num", None)

        return {
            "audio_values": audio_values,
            "audio_len_after_cnn": audio_len_after_cnn,
            "audio_token_num": audio_token_num,
        }

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("audio_values") and "audio" not in modalities:
                modalities["audio"] = self._parse_and_validate_audio_input(
                    **kwargs)
            if input_key in ("pixel_values", ) and "image" not in modalities:
                modalities["image"] = self._parse_and_validate_image_input(
                    **kwargs)

        return modalities

    def _process_audio_input(self, audio_inputs):
        audio_len_after_cnn = audio_inputs[
            "audio_len_after_cnn"]  # value: tensor([[1500]])
        audio_token_num = audio_inputs[
            "audio_token_num"]  # value: tensor([[750]])
        audio_values = audio_inputs["audio_values"]

        # NB: skip the attention mask, just use the audio_values to get the
        # audio embeddings
        # audio_len_after_cnn = torch.tensor(
        #     [[115]],
        #     device=audio_len_after_cnn.device,
        #     dtype=audio_len_after_cnn.dtype,
        # )
        if audio_len_after_cnn is not None:
            input_mel_len = audio_len_after_cnn * 2
            max_mel_len_in_batch = int(input_mel_len.max())
            audio_values = audio_values[:, :, :, :max_mel_len_in_batch]

        last_hidden_state = self.audio_model(
            audio_values)  # (1, 1, 128, 3000) -> (1, 750, 1280)

        audio_embeds = self.mlp2(last_hidden_state)  # (1,750,2560)
        audio_embeds = audio_embeds.flatten(
            0, 1)  # merge the first two dimensions
        audio_token_num = audio_token_num.reshape(-1)
        pad_token_num = audio_embeds.shape[0] - audio_token_num.sum()
        audio_embeds = audio_embeds.split(
            audio_token_num.tolist() + [pad_token_num])[:len(audio_token_num)]


        # audio_features.split(audio_token_num.tolist())
        return audio_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been "
                "swapped back, which results in a transposed image.",
                stacklevel=2,
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _process_image_input(self, image_inputs):
        pixel_values = image_inputs["pixel_values"]
        image_token_num = image_inputs["image_token_num"]
        image_token_num = image_token_num.reshape(-1)
        if pixel_values.ndim == 5:
            pixel_values = pixel_values.flatten(0, 1)
        elif pixel_values.ndim == 6:
            pixel_values = pixel_values.flatten(0, 2)
        last_hidden_state = self.vision_model(pixel_values)  # [5, 1025, 1024]
        last_hidden_state = last_hidden_state[:, 1:, :]  # [5, 1024, 1024]
        h = w = int(last_hidden_state.shape[1]**0.5)  # h, w = 32, 32
        last_hidden_state = last_hidden_state.reshape(
            last_hidden_state.shape[0], h, w, -1)  # [5, 32, 32, 1024]
        last_hidden_state = self.pixel_shuffle(
            last_hidden_state,
            scale_factor=self.downsample_ratio)  # [5, 16, 16, 4096]
        last_hidden_state = last_hidden_state.reshape(
            last_hidden_state.shape[0], -1,
            last_hidden_state.shape[-1])  # [5, 256, 4096]
        vit_embeds = self.mlp1(
            last_hidden_state)  # .to(pixel_values.device)   # [5, 256, 2560]

        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        assert vit_embeds.shape[0] == image_token_num.sum(), (
            f"the number of image tokens {image_token_num.sum()} is not "
            f"equal to the number of image embeddings {vit_embeds.shape[0]}")
        vit_embeds = vit_embeds.split(image_token_num.tolist())
        return vit_embeds

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(
            **kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def get_input_embeddings(self, input_ids, multimodal_embeddings):
        # get the text embeddings
        input_embeds = self.language_model.get_input_embeddings(input_ids)
        # if have multimodal embeddings, merge them with the text embeddings
        if multimodal_embeddings is not None and len(
                multimodal_embeddings) > 0:
            input_embeds = merge_multimodal_embeddings(
                input_ids,
                input_embeds,
                multimodal_embeddings,
                placeholder_token_id=[
                    self.senseomni_config.img_context_token_id,
                    self.senseomni_config.audio_context_token_id,
                ],
            )
        return input_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        positions: Optional[torch.LongTensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights,
                                             mapper=self.hf_to_vllm_mapper)

        return loaded_weights