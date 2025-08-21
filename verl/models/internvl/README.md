本目录说明：

本目录下的文件需要拷贝到 InternVL HuggingFace 模型的目录下（即包含 `config.json` 的目录）。

**操作步骤：**
1. 将本目录下的所有文件复制到你的 InternVL HuggingFace 模型目录下。
2. 修改该目录下的 `config.json` 文件。例如：
```json
{
    "_commit_hash": null,
    "architectures": [
        "InternVLChatModel"
    ],
    # 需要添加 auto map
    "auto_map": {
        "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
        "AutoModel": "modeling_internvl_chat.InternVLChatModel",
        "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
    },
    "boi_contrastive_loss_weight": 0.0,
    "downsample_ratio": 0.5,
    "dynamic_image_size": true,
    "force_image_size": 448,
    "image_fold": null,
    # 需要添加 img_context_token_id
    "img_context_token_id": 151669,
    # 需要添加 eos_token_id
    "eos_token_id": 151645,
    "llm_config": {
        "architectures": [
            "Qwen3MoeForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "decoder_sparse_step": 1,
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 40960,
        "max_window_layers": 94,
        "mlp_only_layers": [],
        "model_type": "qwen3_moe",
        "moe_intermediate_size": 1536,
        "norm_topk_prob": true,
        "num_attention_heads": 64,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 94,
        "num_key_value_heads": 4,
        "output_router_logits": false,
        "rms_norm_eps": 1e-06,
        "rope_scaling": null,
        "rope_theta": 1000000.0,
        "router_aux_loss_coef": 0.001,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.51.0",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    },
    "max_dynamic_patch": 12,
    "min_dynamic_patch": 1,
    "model_type": "internvl_chat",
    "pad2square": false,
    "ps_version": "v3",
    "select_layer": -1,
    "template": "internlm2-chat-v3",
    "torch_dtype": "bfloat16",
    "transformers_version": null,
    "use_backbone_lora": 0,
    "use_img_start_end_token": true,
    "use_llm_lora": 0,
    "use_thumbnail": true,
    "vision_config": {
        "_name_or_path": "pretrained/intern_vit_6b_448px_v1_2/",
        "add_cross_attention": false,
        "architectures": [
            "InternVisionModel"
        ],
        "auto_map": {
            "AutoConfig": "configuration_intern_vit.InternVisionConfig",
            "AutoModel": "modeling_intern_vit.InternVisionModel"
        },
        "attention_dropout": 0.0,
        "drop_path_rate": 0.1,
        "dropout": 0.0,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1536,
        "image_size": 448,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "layer_norm_eps": 1e-06,
        "model_type": "intern_vit_1b",
        "norm_type": "layer_norm",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 40,
        "patch_size": 14,
        "qk_normalization": false,
        "qkv_bias": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.37.2",
        "use_flash_attn": true
    }
}
```
