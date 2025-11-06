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
from verl.utils.dataset.image_processors import bytes_to_pil, base64_to_pil
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


def make_batch_generator(batches, vpp_size, processor=None):
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
        batches = PixelPrefetch(batches, processor)


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
    def __init__(self, multi_modal_data, multi_modal_inputs_idx, multi_modal_inputs, processor):
        self.multi_modal_data = multi_modal_data
        self.multi_modal_inputs_idx = multi_modal_inputs_idx
        self.multi_modal_inputs = multi_modal_inputs

        self.processor = processor

    def __len__(self):
        return len(self.multi_modal_inputs_idx)

    def __getitem__(self, index):
        def image_convert(image_item):
            if isinstance(image_item, Image.Image):
                return image_item
            elif isinstance(image_item, bytes):
                return bytes_to_pil(image_item)
            elif isinstance(image_item, str):
                return base64_to_pil(image_item)
            else:
                raise ValueError(f"Unsupported image type: {type(image_item)}")

        outputs = {
            'multi_modal_inputs_idx': self.multi_modal_inputs_idx[index],
            'multi_modal_inputs': [{}] * len(self.multi_modal_inputs)
        }
        for item_idx in self.multi_modal_inputs_idx[index]:
            if 'pixel_values' in self.multi_modal_inputs[item_idx]:
                outputs['multi_modal_inputs'][item_idx] = copy.deepcopy(self.multi_modal_inputs[item_idx]) # NOTE: use deepcopy to avoid modifying the original data
                continue
            multi_modal_data = self.multi_modal_data[item_idx]
            if 'image' in multi_modal_data:
                images = [image_convert(image_item) for image_item in multi_modal_data['image']]
                multi_modal_inputs = self.processor.image_processor(images=images)
                multi_modal_inputs = {
                    'pixel_values': multi_modal_inputs['pixel_values_flat'],
                    'image_flags': multi_modal_inputs['image_flags']
                }
                outputs['multi_modal_inputs'][item_idx] = multi_modal_inputs

                print(multi_modal_inputs)
                # breakpoint()
            else:
                outputs['multi_modal_inputs'][item_idx] = {
                    'pixel_values': torch.zeros(1, 3, self.image_size, self.image_size),
                    'image_flags': torch.zeros(1, dtype=torch.long)
                }

        return outputs


class PixelPrefetch:

    def __init__(self, batches, processor):

        self.batches = batches

        mutli_modal_data = [item.pop('multi_modal_data').data for item in self.batches][0] # NOTE: [0] has all multi_modal_data
        multi_modal_inputs_idx = [item.pop('multi_modal_inputs_idx').data.to('cpu', non_blocking=False) for item in self.batches]
        multi_modal_inputs = [item.pop('multi_modal_inputs').data for item in self.batches][0] # NOTE: [0] has all multi_modal_inputs

        self.dataset = PrefetchDataset(mutli_modal_data, multi_modal_inputs_idx, multi_modal_inputs, processor)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )

    def __iter__(self):

        for batch_idx, batch_item in enumerate(self.dataloader):
            self.batches[batch_idx]['multi_modal_inputs_idx'] = batch_item['multi_modal_inputs_idx']
            self.batches[batch_idx]['multi_modal_inputs'] = batch_item['multi_modal_inputs']

            yield self.batches[batch_idx]
