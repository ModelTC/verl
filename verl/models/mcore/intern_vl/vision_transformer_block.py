import torch
from typing import Optional, Tuple, Union

from megatron.core.transformer.transformer_block import *


class InternVisionTransformerBlock(TransformerBlock):
    def _checkpointed_forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
    ):
        def custom(start: int, end: int):
            def custom_forward(inputs_embeds, output_hidden_states):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
                        inputs_embeds,
                        output_hidden_states
                    )
                return hidden_states, context
            return custom
        
        def checkpoint_handler(forward_func):
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    inputs_embeds,
                    output_hidden_states
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    inputs_embeds,
                    output_hidden_states
                )
        
        if self.config.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                inputs_embeds, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers)
                )

                layer_idx += self.config.recompute_num_layers
        elif self.config.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    layer_idx >= recompute_skip_num_layers
                    and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    inputs_embeds, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    inputs_embeds, context = custom(layer_idx, layer_idx + 1)(
                        inputs_embeds, output_hidden_states
                    )
        else:
            raise ValueError("Invalid activation recompute method.")

        return inputs_embeds
    
    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        
        # for idx, encoder_layer in enumerate(self.layers):
        #     if output_hidden_states:
        #         encoder_states = encoder_states + (hidden_states,)
        #     if True:
        #         layer_outputs, _ = tensor_parallel.checkpoint(
        #             encoder_layer,
        #             self.config.distribute_saved_activations,
        #             hidden_states,
        #         )
        #     hidden_states = layer_outputs

        def custom(start: int, end: int):
            def custom_forward(hidden_states):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    hidden_states, _ = layer(hidden_states)
                return hidden_states

            return custom_forward

        def checkpoint_handler(forward_func):
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
            )

        recompute_num_layers = 5
        assert len(self.layers) % recompute_num_layers == 0, f"{len(self.layers)=}, {recompute_num_layers=}"
        layer_idx = 0
        while layer_idx < len(self.layers):
            hidden_states = checkpoint_handler(
                custom(layer_idx, layer_idx + recompute_num_layers)
            )
            layer_idx += recompute_num_layers
            
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
            
        return hidden_states