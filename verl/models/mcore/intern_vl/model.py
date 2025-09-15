import torch
import warnings
from typing import List, Optional, Tuple, Union
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from .vit_layer_specs import MLP1

from .vision_model import InternVisionModel
from megatron.core import tensor_parallel

class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vit_embeds, dim, sp_size, sp_group, sp_rank):
        ctx.dim = dim
        ctx.sp_group = sp_group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        vit_embeds_list = [torch.zeros_like(vit_embeds) for _ in range(sp_size)]
        dist.all_gather(vit_embeds_list, vit_embeds, group=sp_group)
        vit_embeds = torch.cat(vit_embeds_list, dim=dim)
        return vit_embeds

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = torch.split(grad_output, grad_output.shape[0] // ctx.sp_size, dim=ctx.dim)
        return grad_output[ctx.sp_rank].contiguous(),  None, None, None, None


class InternVLModel(MegatronModule):
    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        mlp1_layer_spec: ModuleSpec,
        mlp_config: TransformerConfig,
        parallel_output: bool = True,
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        language_rotary_base: int = 10000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        img_context_token_id=None
    ):
        super().__init__(config=language_transformer_config)
        
        self.pre_process = pre_process
        self.post_process = post_process
        
        self.encoder_hidden_state = None
        self.select_layer = vision_transformer_config.select_layer

        self.vision_model = None
        self.mlp1 = None
        
        self.share_embeddings_and_output_weights = False
        if self.pre_process:
            self.vision_model = InternVisionModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                pre_process=True,
                post_process=True,
            )
            
            self.mlp1 = MLP1(
                config=mlp_config,
                submodules=mlp1_layer_spec,
                is_expert=False,
                input_size=None,
            )
        
        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="rope",
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )
        
        self.downsample_ratio = mlp_config.downsample_ratio
        self.ps_version = mlp_config.ps_version
        
        self.img_context_token_id = img_context_token_id
        self.SP_SIZE = mpu.get_context_parallel_world_size()
        self.SP_RANK = mpu.get_context_parallel_rank()
        self.SP_GROUP = mpu.get_context_parallel_group()
        self.all_gather = _AllGather()
        
        self.image_size = vision_transformer_config.image_size

        self.VIT_DP_SIZE = mpu._VISION_DATA_PARALLEL_SIZE if mpu._VISION_DATA_PARALLEL_SIZE is not None else 1
        self.VIT_DP_RANK = mpu._VISION_DATA_PARALLEL_RANK if mpu._VISION_DATA_PARALLEL_RANK is not None else 0
        self.VIT_DP_GROUP = mpu._VISION_DATA_PARALLEL_GROUP if mpu._VISION_DATA_PARALLEL_GROUP is not None else None

        self.LLM_TP_SIZE = mpu._LLM_TP_SIZE if mpu._LLM_TP_SIZE is not None else mpu.get_tensor_model_parallel_world_size()
        self.LLM_TP_RANK = mpu._LLM_TP_RANK if mpu._LLM_TP_RANK is not None else mpu.get_tensor_model_parallel_rank()
        
    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen2VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])
            
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
                return_dict=True) # .last_hidden_state
        # else:
        #     vit_embeds = self.vision_model(
        #         pixel_values=pixel_values,
        #         output_hidden_states=True,
        #         return_dict=True).hidden_states[self.select_layer]

        def projection(vit_embeds):
            vit_embeds = vit_embeds[:, 1:, :]

            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.mlp1(vit_embeds)
            return vit_embeds

        vit_embeds = tensor_parallel.checkpoint(
            projection,
            self.config.distribute_saved_activations,
            vit_embeds,
        )

        return vit_embeds

    def preprocess_for_sp_dp(self, pixel_values, image_flags):
        sp_size, vit_dp_size = self.SP_SIZE, self.VIT_DP_SIZE
        num_image = pixel_values.shape[0]
        if num_image % (sp_size * vit_dp_size) != 0:
            num_pad_image = (sp_size * vit_dp_size) - num_image % (sp_size * vit_dp_size) # 计算需要填充的图像数量
            pad_images = torch.zeros(num_pad_image, 3, self.image_size, self.image_size, device=pixel_values.device, dtype=pixel_values.dtype)
            pad_images_flages = torch.tensor([0] * num_pad_image, dtype=torch.long).to(dtype=image_flags.dtype, device=image_flags.device)
            pixel_values = torch.cat([pixel_values, pad_images], dim=0)
            image_flags = torch.cat([image_flags, pad_images_flages], dim=0)
        return pixel_values, image_flags
    
    
    def split_for_sp(self, tensor, size, dim, rank):
        tensor_list = torch.split(tensor, size // self.SP_SIZE, dim=dim)
        return tensor_list[rank].contiguous()

    def split_for_dp(self, tensor, size, dim, rank):
        tensor_list = torch.split(tensor, size // self.VIT_DP_SIZE, dim=dim)
        return tensor_list[rank].contiguous()
        
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        input_ids: torch.LongTensor = None,
        image_flags: Optional[torch.LongTensor] = None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        img_token_idxs=None,
    ) -> torch.Tensor:
        if self.pre_process:
            input_embeds = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()
            
            img_token_idxs = img_token_idxs.reshape(-1)
            input_ids = input_ids.reshape(-1)
            selected = (input_ids == self.img_context_token_id)
            
            if pixel_values is not None:
                if image_flags.sum() == 0:
                    pixel_values = torch.zeros(1, 3, self.image_size, self.image_size, device=input_embeds.device, dtype=input_embeds.dtype)
                    image_flags = torch.zeros(1, device=input_embeds.device, dtype=torch.long)
                else:
                    pixel_values = pixel_values[image_flags == 1]
                    image_flags = image_flags[image_flags == 1]

                pixel_values, image_flags = self.preprocess_for_sp_dp(pixel_values, image_flags)
                pixel_values = self.split_for_sp(pixel_values, pixel_values.shape[0], 0, self.SP_RANK)
                if self.VIT_DP_SIZE > 1:
                    pixel_values = self.split_for_dp(pixel_values, pixel_values.shape[0], 0, self.VIT_DP_RANK)

                pixel_values = pixel_values.to(device=input_embeds.device, dtype=input_embeds.dtype)
                image_flags = image_flags.to(device=input_embeds.device)
                    
                vit_batch_size = pixel_values.shape[0]
                vit_embeds, _ = self.extract_feature(pixel_values)

                if self.VIT_DP_SIZE > 1:
                    vit_embeds = self.all_gather.apply(vit_embeds, 0, self.VIT_DP_SIZE,
                                                       self.VIT_DP_GROUP, self.VIT_DP_RANK)

                if len(image_flags) != pixel_values.shape[0] and self.SP_SIZE > 1:
                    vit_embeds = self.all_gather.apply(vit_embeds, 0, self.SP_SIZE,
                                                       self.SP_GROUP, self.SP_RANK)
                
                vit_embeds = vit_embeds[image_flags == 1]
                
                N, B, C = input_embeds.shape
                input_embeds = input_embeds.reshape(B * N, C)
                vit_embeds = vit_embeds.reshape(-1, C)
                
                if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
                
                try:
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[img_token_idxs[selected].contiguous()]
                except Exception as e:
                    vit_embeds = vit_embeds.reshape(-1, C)
                    print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                        f'vit_embeds.shape={vit_embeds.shape}')
                    n_token = selected.sum()
                    n_visual_feature = vit_embeds.shape[0]
                    if n_visual_feature < n_token:
                        vit_embeds = torch.cat([vit_embeds, torch.zeros_like(vit_embeds[:n_token - n_visual_feature])], dim=0)

                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                
                input_embeds = input_embeds.reshape(N, B, C)
                
            if self.LLM_TP_SIZE > 1:
                # input_embeds = tensor_parallel.scatter_to_sequence_parallel_region(input_embeds)
                input_embeds = torch.split(input_embeds, input_embeds.shape[0] // self.LLM_TP_SIZE, dim=0)[self.LLM_TP_RANK]
                input_embeds = input_embeds.contiguous()
        else:
            input_embeds = None
        
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=input_embeds,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            # inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            **(extra_block_kwargs or {}),
        )

        return output

    def freeze(self, freeze_language_model: bool, freeze_vision_model: bool, freeze_mlp: bool):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_mlp (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_mlp and self.mlp1 is not None:
            modules.append(self.mlp1)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
