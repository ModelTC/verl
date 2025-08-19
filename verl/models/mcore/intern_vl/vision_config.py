import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig


def get_mlp_config(config: TransformerConfig, hf_config) -> TransformerConfig:
    # Given a Transformer Config from decoder, build vision encoder config
    # diff: out_hidden_size & intermediate_size

    # mlp: hidden_size -> intermediate_size -> embed_dim, silu
    # NOTE: here we provide a workaround to solve the wrong layer amount when VPP of decoder is on
    config.ffn_hidden_size = hf_config.llm_config.hidden_size

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        config.num_layers = hf_config.vision_config.num_hidden_layers * parallel_state.get_virtual_pipeline_model_parallel_world_size()  # depth
    else:
        config.num_layers = hf_config.vision_config.num_hidden_layers
    config.num_attention_heads = hf_config.vision_config.num_attention_heads
    config.add_bias_linear = True  # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True  # qkv_proj in attn has bias
    config.hidden_size = hf_config.vision_config.hidden_size * int(1 / hf_config.downsample_ratio) ** 2
    config.downsample_ratio = hf_config.downsample_ratio
    config.ps_version = hf_config.ps_version
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    config.gated_linear_unit = False # no gated
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    config.normalization = 'LayerNorm' # use RMSNorm
    config.activation_func = F.gelu
    config.seq_length = 1

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = hf_config.vision_config.patch_size
    config.in_channels = hf_config.vision_config.num_channels # 3
    config.spatial_merge_size = 2

    config.fullatt_block_indexes = [7, 15, 23, 31]
    config._qwen2_5_vl_window_size = 112
    # extra
    config.image_size = hf_config.vision_config.image_size
    config.select_layer = hf_config.select_layer
    return config


def get_vision_model_config(config: TransformerConfig, hf_config) -> TransformerConfig:
    # Given a Transformer Config from decoder, build vision encoder config
    # diff: out_hidden_size & intermediate_size

    # mlp: hidden_size -> intermediate_size -> embed_dim, silu
    # NOTE: here we provide a workaround to solve the wrong layer amount when VPP of decoder is on
    config.ffn_hidden_size = hf_config.vision_config.intermediate_size

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        config.num_layers = hf_config.vision_config.num_hidden_layers * parallel_state.get_virtual_pipeline_model_parallel_world_size()  # depth
    else:
        config.num_layers = hf_config.vision_config.num_hidden_layers
    config.num_attention_heads = hf_config.vision_config.num_attention_heads
    config.add_bias_linear = True  # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True  # qkv_proj in attn has bias
    config.hidden_size = hf_config.vision_config.hidden_size
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    config.gated_linear_unit = False # no gated
    config.activation_func = F.gelu # quick_gelu # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    config.seq_length = 1

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.temporal_patch_size = 2
    config.patch_size = hf_config.vision_config.patch_size
    config.in_channels = hf_config.vision_config.num_channels # 3
    config.spatial_merge_size = 2

    config.fullatt_block_indexes = [7, 15, 23, 31]
    config._qwen2_5_vl_window_size = 112
    # extra
    config.image_size = hf_config.vision_config.image_size
    config.select_layer = hf_config.select_layer
    config.normalization = "LayerNorm"
    config.initializer_factor = hf_config.vision_config.initializer_factor
    return config