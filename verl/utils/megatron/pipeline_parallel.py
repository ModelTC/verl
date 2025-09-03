# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import torch
from megatron.core import parallel_state as mpu
import time

from verl import DataProto
from .sequence_parallel import pad_to_sequence_parallel

from PIL import Image
from verl.utils.dataset.image_processors import bytes_to_pil, base64_to_pil, preprocess_images_internvl
from functools import partial


def compute_transformers_input_shapes(batches, meta_info):
    from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron

    # pre-compute input shapes for each micro-batch at each pp stage
    input_shapes = []
    for model_inputs in batches:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        if meta_info["sequence_parallel"]:
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # compute shapes for model_inputs
            input_shapes.append(
                torch.Size(
                    [
                        input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(),
                        1,
                        meta_info["hidden_size"],
                    ]
                )
            )
        else:
            # compute shapes for model_inputs
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info["hidden_size"]]))
    return input_shapes


def make_batch_generator(batches, vpp_size, hf_config=None):
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """

    if 'multi_modal_data' in batches[0].keys():
        batches = pixel_prefetch(batches, hf_config)


    # batches = batches[:2]
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, batches, hf_config):
        self.batches = batches
        self.hf_config = hf_config

        self.image_size = self.hf_config.force_image_size or self.hf_config.vision_config.image_size
        patch_size = self.hf_config.vision_config.patch_size
        downsample_ratio = self.hf_config.downsample_ratio
        self.preprocess_images_fn = partial(
            preprocess_images_internvl,
            image_size=self.image_size, 
            min_dynamic_patch=self.hf_config.min_dynamic_patch, 
            max_dynamic_patch=self.hf_config.max_dynamic_patch, 
            use_thumbnail=self.hf_config.use_thumbnail, 
            num_image_token=int((self.image_size // patch_size)**2 * (downsample_ratio**2))
        )


    def multi_modal_preprocess(self, data):
        if 'multi_modal_data' not in data.keys():
            return data


        def image_convert(image_item):
            if isinstance(image_item, Image.Image):
                return image_item
            elif isinstance(image_item, bytes):
                return bytes_to_pil(image_item)
            elif isinstance(image_item, str):
                return base64_to_pil(image_item)
            else:
                raise ValueError(f"Unsupported image type: {type(image_item)}")

        for item_idx in data['multi_modal_inputs_idx']:
            if 'pixel_values' in data['multi_modal_inputs'][item_idx]:
                continue
            multi_modal_data = data['multi_modal_data'][item_idx]
            if 'image' in multi_modal_data:
                images = [image_convert(image_item) for image_item in multi_modal_data['image']]
                image_inputs, _ = self.preprocess_images_fn(images=images)
                data['multi_modal_inputs'][item_idx] = image_inputs
            else:
                data['multi_modal_inputs'][item_idx] = {
                    'pixel_values': torch.zeros(1, 3, self.image_size, self.image_size),
                    'image_flags': torch.zeros(1, dtype=torch.long)
                }

        data.pop('multi_modal_data')
        return data


    def __len__(self):
        return len(self.batches)


    def __getitem__(self, index):
        batch_item = copy.deepcopy(self.batches[index])
        return self.multi_modal_preprocess(batch_item)


def pixel_prefetch(batches, hf_config):
    batches_cpu = [item.to("cpu", non_blocking=False) for item in batches]
    dataset = PrefetchDataset(batches_cpu, hf_config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    return dataloader
