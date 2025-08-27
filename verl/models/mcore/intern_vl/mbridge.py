from .model import InternVLModel
from .vision_config import get_vision_model_config, get_mlp_config

from copy import deepcopy
from typing import Callable, Generator, Optional

import torch
from mbridge.core import VLMBridge, register_model
from mbridge.core.util import unwrap_model
from torch.nn import functional as F
from transformers import PretrainedConfig

from .vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from .model import InternVLModel
from .vision_config import get_vision_model_config, get_mlp_config

from mbridge.core.util import (
    broadcast_from_megatron_pp,
    broadcast_str_from_megatron_pp,
    unwrap_model,
)
from megatron.core import parallel_state as mpu

@register_model("internvl_chat")
class InternVLBridge(VLMBridge):
    """
    Bridge implementation for InternVL models.

    This class extends VLMBridge to provide specific configurations and
    optimizations for InternVL models, handling the conversion between
    Hugging Face InternVL format and Megatron-Core.
    """

    _DIRECT_MAPPING = {
        "language_model.embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
        "language_model.decoder.final_layernorm.weight": "language_model.model.norm.weight",
        "language_model.output_layer.weight": "language_model.lm_head.weight",
        "vision_model.patch_embed.class_embedding": "vision_model.embeddings.class_embedding",
        "vision_model.patch_embed.position_embedding": "vision_model.embeddings.position_embedding",
        "vision_model.patch_embed.patch_embedding.weight": "vision_model.embeddings.patch_embedding.weight",
        "vision_model.patch_embed.patch_embedding.bias": "vision_model.embeddings.patch_embedding.bias",

        # "vision_model.projection.encoder.linear_fc1.weight": "vision_model.projection.linear_fc1.weight",
        # "vision_model.projection.encoder.linear_fc1.bias": "vision_model.projection.linear_fc1.bias",
        # "vision_model.projection.encoder.linear_fc2.weight": "vision_model.projection.linear_fc2.weight",
        # "vision_model.projection.encoder.linear_fc2.bias": "vision_model.projection.linear_fc2.bias",
        "mlp1.linear_fc1.layer_norm_weight": "mlp1.0.weight",
        "mlp1.linear_fc1.layer_norm_bias": "mlp1.0.bias",
        "mlp1.linear_fc1.weight": "mlp1.1.weight",
        "mlp1.linear_fc1.bias": "mlp1.1.bias",
        "mlp1.linear_fc2.weight": "mlp1.3.weight",
        "mlp1.linear_fc2.bias": "mlp1.3.bias",
    }

    _ATTENTION_MAPPING = {
        # language model attention
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.bias": [
            "language_model.model.layers.{layer_number}.self_attn.q_proj.bias",
            "language_model.model.layers.{layer_number}.self_attn.k_proj.bias",
            "language_model.model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.weight": [
            "language_model.model.layers.{layer_number}.self_attn.q_proj.weight",
            "language_model.model.layers.{layer_number}.self_attn.k_proj.weight",
            "language_model.model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_proj.weight":
        ["language_model.model.layers.{layer_number}.self_attn.o_proj.weight"],
        "language_model.decoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight":
        ["language_model.model.layers.{layer_number}.input_layernorm.weight"],
        "language_model.decoder.layers.{layer_number}.self_attention.q_layernorm.weight":
        ["language_model.model.layers.{layer_number}.self_attn.q_norm.weight"],
        "language_model.decoder.layers.{layer_number}.self_attention.k_layernorm.weight":
        ["language_model.model.layers.{layer_number}.self_attn.k_norm.weight"],

        # vision model attention - InternVL specific
        "vision_model.encoder.layers.{layer_number}.self_attention.linear_proj.weight": ["vision_model.encoder.layers.{layer_number}.attn.proj.weight"],
        "vision_model.encoder.layers.{layer_number}.self_attention.linear_proj.bias": ["vision_model.encoder.layers.{layer_number}.attn.proj.bias"],
        "vision_model.encoder.layers.{layer_number}.self_attention.linear_qkv.bias": ["vision_model.encoder.layers.{layer_number}.attn.qkv.bias"],
        "vision_model.encoder.layers.{layer_number}.self_attention.linear_qkv.weight": ["vision_model.encoder.layers.{layer_number}.attn.qkv.weight"],
        # "vision_model.encoder.layers.{layer_number}.self_attention.linear_qkv.layer_norm_weight": ["vision_model.encoder.layers.{layer_number}.norm1.weight"],
    }

    _MLP_MAPPING = {
        # language model MLP
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.weight": [
            "language_model.model.layers.{layer_number}.mlp.gate_proj.weight",
            "language_model.model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.bias": [
            "language_model.model.layers.{layer_number}.mlp.gate_proj.bias",
            "language_model.model.layers.{layer_number}.mlp.up_proj.bias",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.weight": ["language_model.model.layers.{layer_number}.mlp.down_proj.weight"],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc2.bias": ["language_model.model.layers.{layer_number}.mlp.down_proj.bias"],
        "language_model.decoder.layers.{layer_number}.mlp.linear_fc1.layer_norm_weight":
        ["language_model.model.layers.{layer_number}.post_attention_layernorm.weight"],
        # "shared_experts.linear_fc1.weight": [
        #     "language_model.model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
        #     "language_model.model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        # ],
        "language_model.decoder.layers.{layer_number}.pre_mlp_layernorm": ["language_model.model.layers.{layer_number}.post_attention_layernorm.weight"],
        # "shared_experts.linear_fc2.weight": ["language_model.model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
        "language_model.decoder.layers.{layer_number}.mlp.router.weight": ["language_model.model.layers.{layer_number}.mlp.gate.weight"],
        # "shared_experts.gate_weight": ["language_model.model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc1": [
            "language_model.model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "language_model.model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "language_model.decoder.layers.{layer_number}.mlp.experts.linear_fc2":
        ["language_model.model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],

        # vision model MLP - InternVL specific
        "vision_model.encoder.layers.{layer_number}.ls1": ["vision_model.encoder.layers.{layer_number}.ls1"],
        "vision_model.encoder.layers.{layer_number}.ls2": ["vision_model.encoder.layers.{layer_number}.ls2"],
        "vision_model.encoder.layers.{layer_number}.mlp.linear_fc1.weight": ["vision_model.encoder.layers.{layer_number}.mlp.fc1.weight"],
        "vision_model.encoder.layers.{layer_number}.mlp.linear_fc1.bias": ["vision_model.encoder.layers.{layer_number}.mlp.fc1.bias"],
        "vision_model.encoder.layers.{layer_number}.mlp.linear_fc2.weight": ["vision_model.encoder.layers.{layer_number}.mlp.fc2.weight"],
        "vision_model.encoder.layers.{layer_number}.mlp.linear_fc2.bias": ["vision_model.encoder.layers.{layer_number}.mlp.fc2.bias"],
        "vision_model.encoder.layers.{layer_number}.input_layernorm.weight": ["vision_model.encoder.layers.{layer_number}.norm1.weight"],
        "vision_model.encoder.layers.{layer_number}.input_layernorm.bias": ["vision_model.encoder.layers.{layer_number}.norm1.bias"],
        "vision_model.encoder.layers.{layer_number}.pre_mlp_layernorm.weight": ["vision_model.encoder.layers.{layer_number}.norm2.weight"],
        "vision_model.encoder.layers.{layer_number}.pre_mlp_layernorm.bias": ["vision_model.encoder.layers.{layer_number}.norm2.bias"],
    }

    def _model_provider(self, post_model_creation_callbacks: list[Callable[[torch.nn.Module], None]]):
        """
        Creates and returns a model provider function.

        The returned function creates an InternVL model with the specified configuration
        when called with pre_process and post_process parameters.

        Args:
            post_model_creation_callbacks: List of callbacks to be called after model creation

        Returns:
            function: A provider function that creates and returns an InternVL model instance
        """

        share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        def provider(pre_process, post_process, vp_stage: Optional[int] = None):

            tfconfig = self.config
            hf_config = self.hf_config
            from copy import deepcopy

            from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELayerNormColumnParallelLinear, TERowParallelLinear
            from megatron.core.models.gpt.moe_module_specs import MLPSubmodules

            transformer_layer_spec = self._get_transformer_layer_spec()

            vision_transformer_config = get_vision_model_config(deepcopy(tfconfig), deepcopy(hf_config))
            vision_transformer_config.pipeline_model_parallel_size = 1
            vision_transformer_config.first_pipeline_num_layers = None
            vision_transformer_config.num_layers_in_first_pipeline_stage = None
            vision_transformer_config.num_layers_in_last_pipeline_stage = None

            mlp_config = get_mlp_config(deepcopy(tfconfig), deepcopy(hf_config))

            vision_projection_config = None
            vision_projection_layer_spec = MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

            # breakpoint()
            mlp1_layer_spec = MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,  # TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )

            model = InternVLModel(
                language_transformer_config=tfconfig,
                language_transformer_layer_spec=transformer_layer_spec,
                language_vocab_size=hf_config.llm_config.vocab_size,
                language_max_sequence_length=hf_config.llm_config.max_position_embeddings,
                vision_transformer_config=vision_transformer_config,
                vision_transformer_layer_spec=vision_transformer_layer_spec,
                language_rotary_base=hf_config.llm_config.rope_theta,
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                mlp1_layer_spec=mlp1_layer_spec,
                mlp_config=mlp_config,
                img_context_token_id=hf_config.img_context_token_id,
            )

            freeze_language_model = getattr(hf_config, "freeze_language_model", False)
            freeze_vision_model = getattr(hf_config, "freeze_vision_model", False)
            freeze_mlp = getattr(hf_config, "freeze_mlp", False)
            model.freeze(freeze_language_model, freeze_vision_model, freeze_mlp)

            for callback in post_model_creation_callbacks:
                callback(model)

            return model

        return provider

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """
        Map MCore weight names to Hugging Face weight names.

        Args:
            mcore_weights_name: MCore weight name

        Returns:
            list: Corresponding Hugging Face weight names
        """
        assert (
            "_extra_state" not in mcore_weights_name
        ), "extra_state should not be loaded"

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name or 'vision_model' in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []
        # print(key)
        mapping_names = self._ATTENTION_MAPPING[key]
        convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        split_name = name.split(".")
        layer_number = split_name[3]
        split_name[3] = "{layer_number}"
        key = ".".join(split_name)
        convert_names = []

        # print(key)
        # mapping_names = self._MLP_MAPPING[key]

        for name_key, mapping_names in self._MLP_MAPPING.items():
            if name_key in key:
                # mapping_names = mapping_names

                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend([x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names])
                else:
                    convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])

                break

        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _weight_to_hf_format(self, mcore_weights_name: str, mcore_weights: torch.Tensor) -> tuple[list[str], list[torch.Tensor]]:
        """
        Export MCore weights to Hugging Face format.

        Takes MCore weight names and tensor, outputs Hugging Face weight names and tensors.
        Due to MCore's runtime optimizations involving weight merging, output can be a list.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor

        Returns:
            tuple: (hf_names, hf_weights) - lists of Hugging Face weight names and tensors

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)

        if 'vision_model' in mcore_weights_name:
            if 'self_attention.linear_qkv.' in mcore_weights_name:
                hf_names = hf_names * 3

        if len(hf_names) == 1:
            return [hf_names[0]], [mcore_weights]
        if ("self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name):
            # split qkv
            assert len(hf_names) == 3
            # split qkv
            num_key_value_heads = getattr(self.hf_config.llm_config, "num_key_value_heads", 32)
            hidden_dim = getattr(self.hf_config.llm_config, "hidden_size", 4096)
            num_attention_heads = getattr(self.hf_config.llm_config, "num_attention_heads", 32)
            head_dim = getattr(self.hf_config.llm_config, "head_dim", hidden_dim // num_attention_heads)
            if "vision_model" in mcore_weights_name:
                num_attention_heads = getattr(self.hf_config.vision_config, "num_attention_heads", 16)
                num_key_value_heads = getattr(self.hf_config.vision_config, "num_attention_heads", 16)
                hidden_dim = getattr(self.hf_config.vision_config, "hidden_size", 1536)
                head_dim = getattr(self.hf_config.vision_config, "head_dim", hidden_dim // num_attention_heads) # 1536 / 16 = 96

            out_shape = ([num_key_value_heads, -1, hidden_dim] if ".bias" not in mcore_weights_name else [num_key_value_heads, -1])
            qkv = mcore_weights.view(*out_shape)
            q_len = head_dim * num_attention_heads // num_key_value_heads
            k_len = head_dim
            v_len = head_dim
            single_out_shape = ([-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1])
            q = qkv[:, :q_len].reshape(*single_out_shape)
            k = qkv[:, q_len:q_len + k_len].reshape(*single_out_shape)
            v = qkv[:, q_len + k_len:].reshape(*single_out_shape)

            if 'vision_model' in mcore_weights_name:
                qkv = torch.cat([q, k, v], dim=0)
                return [hf_names[0]], [qkv]
            else:
                return hf_names, [q, k, v]

        elif ("linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name):
            # split gate_proj and up_proj
            assert len(hf_names) == 2
            gate, up = mcore_weights.chunk(2)
            return hf_names, [gate, up]
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list[torch.Tensor]) -> torch.Tensor:
        """
        Import Hugging Face weights to MCore format.

        Takes Hugging Face weight names and tensors, outputs MCore weight tensor.
        Due to MCore's runtime optimizations involving weight merging, input is a list.

        Args:
            mcore_weights_name: MCore weight name
            hf_weights: List of Hugging Face weight tensors

        Returns:
            torch.Tensor: MCore weight tensor

        Raises:
            NotImplementedError: If the parameter name is unsupported
        """
        if 'vision_model' in mcore_weights_name:
            if 'self_attention.linear_qkv.' in mcore_weights_name:
                hf_weights = hf_weights[0].chunk(3)

        if len(hf_weights) == 1:
            return hf_weights[0]
        if ("self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name):
            # merge qkv
            assert len(hf_weights) == 3
            num_key_value_heads = getattr(self.hf_config.llm_config, "num_key_value_heads", 32)
            hidden_dim = getattr(self.hf_config.llm_config, "hidden_size", 4096)
            num_attention_heads = getattr(self.hf_config.llm_config, "num_attention_heads", 32)
            head_dim = getattr(self.hf_config.llm_config, "head_dim", hidden_dim // num_attention_heads)
            if "vision_model" in mcore_weights_name:
                num_attention_heads = getattr(self.hf_config.vision_config, "num_attention_heads", 16)
                num_key_value_heads = getattr(self.hf_config.vision_config, "num_attention_heads", 16)
                hidden_dim = getattr(self.hf_config.vision_config, "hidden_size", 1536)
                head_dim = getattr(self.hf_config.vision_config, "head_dim", hidden_dim // num_attention_heads)

            group_dim = head_dim * num_attention_heads // num_key_value_heads
            q, k, v = hf_weights
            # q k v might be tp split
            real_num_key_value_heads = q.shape[0] // group_dim
            q = q.view([
                real_num_key_value_heads,
                group_dim,
                -1,
            ])
            k = k.view([real_num_key_value_heads, head_dim, -1])
            v = v.view([real_num_key_value_heads, head_dim, -1])
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qkv
        elif ("linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name):
            # merge gate_proj and up_proj
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")

    def _weight_name_mapping_mcore_local_to_global(self, model: torch.nn.Module, consider_ep: bool = True) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """

        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "language_model") and hasattr(model.language_model, "decoder"):
            for idx, layer in enumerate(model.language_model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [k for k in model.state_dict().keys() if "_extra_state" not in k]

        # print(model)
        # import ipdb
        # ipdb.set_trace()
        ret = {}
        for param_name in all_param_names:
            keyword = "language_model.decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(f"layers.{layer_idx}.", f"layers.{global_layer_idx}.")
            else:
                ret[param_name] = param_name

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {i: i + num_experts_per_rank * self.mpu.ep_rank for i in range(num_experts_per_rank)}
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    name_prefix, local_expert_id = v.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[int(local_expert_id)]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret


    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merge weights across tensor parallel ranks.
        In mcore format

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: List of MCore weight tensors from different TP ranks
            param: Parameter tensor

        Returns:
            torch.Tensor: Merged weight tensor
        """
        if self.mpu.tp_size == 1:
            assert len(mcore_weights) == 1
            return mcore_weights[0]
        if "mlp.experts.linear_fc" in mcore_weights_name:
            assert len(mcore_weights) == self.mpu.etp_size
        else:
            assert len(mcore_weights) == self.mpu.tp_size
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return torch.cat(mcore_weights, dim=0)
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):

            if 'vision_model' in mcore_weights_name or 'mlp1' in mcore_weights_name:
                ret = torch.cat(mcore_weights, dim=0)
            else:
                # if the tensor is gate and proj
                gate_lst = []
                up_lst = []
                for mcore_weight in mcore_weights:
                    gate, up = mcore_weight.chunk(2)
                    gate_lst.append(gate)
                    up_lst.append(up)
                gate = torch.cat(gate_lst, dim=0)
                up = torch.cat(up_lst, dim=0)
                ret = torch.cat((gate, up), dim=0)

        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = torch.cat(mcore_weights, dim=1)
        else:
            assert (
                hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
            )
            ret = torch.cat(mcore_weights, dim=param.partition_dim)

        return ret

    def _weight_split_across_tp(
        self,
        mcore_weights_name: str,
        mcore_weights: torch.Tensor,
        param: torch.Tensor,
        tp_split_size: int,
    ) -> list[torch.Tensor]:
        """
        Split weight tensor across tensor parallel ranks.

        Args:
            mcore_weights_name: MCore weight name
            mcore_weights: MCore weight tensor
            param: Parameter tensor

        Returns:
            list: List of weight tensors split for each TP rank
        """
        if tp_split_size == 1:
            return [mcore_weights]

        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
        ):
            return mcore_weights.chunk(tp_split_size)
        elif (
            "linear_fc1.weight" in mcore_weights_name
            or "linear_fc1.bias" in mcore_weights_name
        ):

            if 'vision_model' in mcore_weights_name or 'mlp1' in mcore_weights_name:
                ret = mcore_weights.chunk(tp_split_size)
            else:
                gate, up = mcore_weights.chunk(2)
                gates = gate.chunk(tp_split_size)
                ups = up.chunk(tp_split_size)
                ret = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
            ret = mcore_weights.chunk(tp_split_size, dim=1)
        else:
            if param.shape == mcore_weights.shape:
                return [mcore_weights for _ in range(tp_split_size)]
            assert len(param.shape) == len(mcore_weights.shape)
            for partition_dim, (s1, s2) in enumerate(
                zip(param.shape, mcore_weights.shape)
            ):
                if s1 != s2:
                    break

            ret = mcore_weights.chunk(tp_split_size, dim=partition_dim)
        return ret


    def _build_config(self):
        """
        Build the configuration for InternVL models.

        Configures InternVL-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for InternVL models
        """
        hf_config = self.hf_config
        if hf_config.llm_config.architectures[0] == "Qwen2ForCausalLM":
            return self._build_base_config(
                hf_config=hf_config.llm_config,
                dtype=self.dtype,
                add_bias_linear=False,
                # qwen specific
                add_qkv_bias=True,
                # mrope_section=hf_config.rope_scaling["mrope_section"],
            )

        elif hf_config.llm_config.architectures[0] == "Qwen3ForCausalLM":
            return self._build_base_config(
                hf_config=hf_config.llm_config,
                dtype=self.dtype,
                add_bias_linear=False,
            )

        elif hf_config.llm_config.architectures[0] == "Qwen3MoeForCausalLM":
            return self._build_base_config(
                hf_config=hf_config.llm_config,
                dtype=self.dtype,
                use_cpu_initialization=False,
                add_bias_linear=False,
                layernorm_epsilon=hf_config.llm_config.rms_norm_eps,
                # MoE specific
                moe_ffn_hidden_size=hf_config.llm_config.moe_intermediate_size,
                moe_router_bias_update_rate=0.001,
                moe_router_topk=hf_config.llm_config.num_experts_per_tok,
                num_moe_experts=hf_config.llm_config.num_experts,
                moe_aux_loss_coeff=hf_config.llm_config.router_aux_loss_coef,
                # moe_aux_loss_coeff=0.0,
                moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
                moe_grouped_gemm=True,
                moe_router_score_function="softmax",
                # Other optimizations
                persist_layer_norm=True,
                bias_activation_fusion=True,
                bias_dropout_fusion=True,
                # Qwen specific
                moe_router_pre_softmax=False,
                qk_layernorm=True,
            )

    def _build_base_config(self, hf_config: PretrainedConfig, dtype: torch.dtype, **kwargs):
        """
        Build the configuration for InternVL models.

        Configures InternVL-specific parameters such as attention bias settings.

        Returns:
            TransformerConfig: Configuration object for InternVL models
        """

        overlap_p2p_comm = self.mpu.vpp_size is not None and self.mpu.pp_size > 1
        batch_p2p_comm = False
        base_config = {
            # Model architecture parameters
            "num_layers": hf_config.num_hidden_layers,
            "hidden_size": hf_config.hidden_size,
            "num_attention_heads": hf_config.num_attention_heads,
            "num_query_groups": hf_config.num_key_value_heads,
            "ffn_hidden_size": hf_config.intermediate_size,
            "attention_dropout": hf_config.attention_dropout,
            "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
            "kv_channels": getattr(hf_config, "head_dim", None),
            "layernorm_epsilon": hf_config.rms_norm_eps,
            # Activation and normalization
            "activation_func": F.silu,
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            # Data types
            "pipeline_dtype": dtype,
            "params_dtype": dtype,
            "bf16": dtype is torch.bfloat16,
            # Parallel configuration
            "tensor_model_parallel_size": self.mpu.tp_size,
            "pipeline_model_parallel_size": self.mpu.pp_size,
            "expert_model_parallel_size": self.mpu.ep_size,
            "expert_tensor_parallel_size": self.mpu.etp_size,
            "virtual_pipeline_model_parallel_size": self.mpu.vpp_size,
            "context_parallel_size": self.mpu.cp_size,
            "sequence_parallel": self.mpu.tp_size > 1,
            # Common settings
            "variable_seq_lengths": True,
            "masked_softmax_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "add_bias_linear": False,
            "use_cpu_initialization": False,
            "overlap_p2p_comm": overlap_p2p_comm,
            "batch_p2p_comm": batch_p2p_comm,
            "qk_layernorm": getattr(hf_config, "qk_normalization", True),
        }

        # Update with any provided overrides
        base_config.update(kwargs)
        base_config.update(self.extra_args)

        return self.TransformerConfigClass(**base_config)

    def export_weights(
        self, models: list[torch.nn.Module]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        models = [unwrap_model(model) for model in models]

        def get_model_chunk_generator():
            for model in models:
                existing_keys = set()
                for name, param in model.named_parameters():
                    existing_keys.add(name)
                    yield name, param

                # note
                # there is a bug in megatron GPTModel
                # decoder.layers[n].mlp.router.expert_bias" in GPTModel is not registered in named_parameter, but in state_dict().
                # for now we patch it by adding those keys to extra_keys.
                extra_keys = [
                    x
                    for x in model.state_dict().keys()
                    if "_extra_state" not in x
                    and "expert_bias" in x
                    and x not in existing_keys
                ]
                for name in extra_keys:
                    yield name, model.state_dict()[name].to(torch.cuda.current_device())

        weights_names = []
        for vpp_rank, model in enumerate(models):
            existing_keys = set()
            for name, param in model.named_parameters():
                existing_keys.add(name)
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))
            extra_keys = [
                x
                for x in model.state_dict().keys()
                if "_extra_state" not in x
                and "expert_bias" in x
                and x not in existing_keys
            ]
            for name in extra_keys:
                weights_names.append((self.mpu.pp_rank, vpp_rank, name))

        weights_names_all_pp = [None] * self.mpu.pp_size
        torch.distributed.all_gather_object(
            object_list=weights_names_all_pp, obj=weights_names, group=self.mpu.pp_group
        )
        weights_names_all_pp = sum(weights_names_all_pp, [])
        model_chunk_generator = get_model_chunk_generator()
        local_to_global_maps = [
            self._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)
            for model in models
        ]
        for iter_pp_rank, iter_vpp_rank, iter_name in weights_names_all_pp:
            local_to_global_map = local_to_global_maps[iter_vpp_rank]
            if iter_pp_rank == self.mpu.pp_rank:
                try:
                    name, param = next(model_chunk_generator)
                except StopIteration:
                    name, param = None, None
                name = local_to_global_map[iter_name]
            else:
                name, param = None, None

            name = broadcast_str_from_megatron_pp(name)
            broad_pp_param = broadcast_from_megatron_pp(param)

            # EP
            if ".mlp.experts.linear_fc" in name and self.mpu.ep_size > 1:
                num_experts = self.config.num_moe_experts
                num_experts_per_rank = num_experts // self.mpu.ep_size
                infer_params = [
                    torch.empty_like(broad_pp_param) for _ in range(self.mpu.ep_size)
                ]
                torch.distributed.all_gather(
                    infer_params, broad_pp_param, group=self.mpu.ep_group
                )

                name_prefix, local_expert_id = name.split(".weight")
                local_expert_id = int(local_expert_id)
                global_expert_ids = [
                    num_experts_per_rank * ep_rank + local_expert_id
                    for ep_rank in range(self.mpu.ep_size)
                ]
                global_expert_names = [
                    f"{name_prefix}.weight{expert_id}"
                    for expert_id in global_expert_ids
                ]

                for name, param in zip(global_expert_names, infer_params):
                    if self.mpu.etp_size > 1:
                        # gather etp
                        etp_params = [
                            torch.empty_like(param) for _ in range(self.mpu.etp_size)
                        ]
                        torch.distributed.all_gather(
                            etp_params, param, group=self.mpu.etp_group
                        )
                        params = etp_params
                    else:
                        params = [param]

                    merge_params = self._weight_merge_across_tp(
                        name, params, broad_pp_param
                    )
                    converted_names, converted_params = self._weight_to_hf_format(
                        name, merge_params
                    )
                    yield from zip(converted_names, converted_params)
                continue

            # TP
            if (
                hasattr(broad_pp_param, "tensor_model_parallel")
                and broad_pp_param.tensor_model_parallel
            ):
                if mpu._VISION_TP_SIZE != mpu._LLM_TP_SIZE:
                    if self.mpu.pp_rank == 0:
                        self.mpu.tp_group = mpu._LLM_TP_GROUP
                        self.mpu.tp_size = mpu._LLM_TP_SIZE
                    
                    if name.startswith("vision_model") or name.startswith("mlp1") or "word_embeddings" in name:
                        infer_params = broad_pp_param
                    else:
                        # allocate a new tensor with proper size
                        if self.mpu.tp_size <= 1:
                            infer_params = [broad_pp_param]
                        else:
                            infer_params = [
                                torch.empty_like(broad_pp_param)
                                for _ in range(self.mpu.tp_size)
                            ]
                            torch.distributed.all_gather(
                                infer_params, broad_pp_param, group=self.mpu.tp_group
                            )
                        infer_params = self._weight_merge_across_tp(
                            name, infer_params, broad_pp_param
                        )
                else:
                    # allocate a new tensor with proper size
                    if self.mpu.tp_size <= 1:
                        infer_params = [broad_pp_param]
                    else:
                        infer_params = [
                            torch.empty_like(broad_pp_param)
                            for _ in range(self.mpu.tp_size)
                        ]
                        torch.distributed.all_gather(
                            infer_params, broad_pp_param, group=self.mpu.tp_group
                        )

                    infer_params = self._weight_merge_across_tp(
                        name, infer_params, broad_pp_param
                    )
            else:
                infer_params = broad_pp_param

            converted_names, converted_params = self._weight_to_hf_format(
                name, infer_params
            )

            yield from zip(converted_names, converted_params)

        self.mpu.tp_group = mpu.get_tensor_model_parallel_group()
        self.mpu.tp_size = mpu.get_tensor_model_parallel_world_size()
