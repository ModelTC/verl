from typing import Optional

import torch
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import nn
from torch.nn import functional as F

from .vision_transformer_block import InternVisionTransformerBlock as TransformerBlock


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        image_size: int = 256,
        embed_dim: int = 1152,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionModel(VisionModule):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = False,
    ) -> None:
        super().__init__(config=transformer_config)
        self.transformer_config = transformer_config
        
        patch_size = transformer_config.patch_size
        image_size = transformer_config.image_size
        embed_dim = transformer_config.hidden_size
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            image_size=image_size,
            embed_dim=embed_dim,
        )
        
        self.pre_process = pre_process
        self.post_process = post_process
        transformer_config.context_parallel_size = 1
        self.encoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False, # True,
        )
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        return_dict=False,
    ) -> torch.Tensor:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.transformer_config.output_hidden_states
        )
        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')
        
        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.patch_embed(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
        )
        
        return encoder_outputs