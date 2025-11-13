# --------------------------------------------------------
# SenseOmni
# Copyright (c) 2025 SenseOmni Team
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import AutoConfig, LlamaConfig, Qwen3MoeConfig, Qwen2Config, Qwen3Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_intern_vit import InternVisionConfig
from .configuration_whisper import WhisperConfig


logger = logging.get_logger(__name__)


class SenseOmniConfig(PretrainedConfig):
    model_type = 'senseomni'
    is_composition = False
    has_no_defaults_at_init = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            audio_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            img_context_token_id=151669,
            eos_token_id=151645,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen3ForCausalLM']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        if audio_config is None:
            audio_config = {}
            logger.info('audio_config is None. Initializing the Audioconfig with default values.')

        self.vision_config = InternVisionConfig(**vision_config)
        self.audio_config = WhisperConfig(**audio_config)

        if llm_config.get('architectures')[0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen3ForCausalLM':
            self.llm_config = Qwen3Config(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen3MoeForCausalLM':
            self.llm_config = Qwen3MoeConfig(**llm_config)
        else:
            self.llm_config = Qwen3Config(**llm_config) # NOTE(yzx): default to qwen3

        self.text_config = self.llm_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        self.img_context_token_id = img_context_token_id
        self.eos_token_id = eos_token_id

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['audio_config'] = self.audio_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()

        if self.text_config is not None:
            output['text_config'] = self.text_config.to_dict()

        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['img_context_token_id'] = self.img_context_token_id
        output['eos_token_id'] = self.eos_token_id

        return output
    
AutoConfig.register("senseomni", SenseOmniConfig)