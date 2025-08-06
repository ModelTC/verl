# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import cutlass_fp8_supported


def get_full_tensor_from_distributed(param):
    if isinstance(param, torch.distributed.tensor.DTensor):
        return param.full_tensor()
    else:
        return param

@torch.no_grad()
def weight_fuse_and_quantize(updated_params):

    device = torch.cuda.current_device()

    delete_keys = []
    new_updated_params = {}

    for name in list(updated_params.keys()):
        if 'vision_model' in name:
            continue

        if 'q_proj.weight' in name:
            q_weight = updated_params.pop(name)
            k_weight = updated_params.pop(name.replace('q_proj.weight', 'k_proj.weight'))
            v_weight = updated_params.pop(name.replace('q_proj.weight', 'v_proj.weight'))


            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            q_weight = q_weight.to(torch.device("cpu"), non_blocking=True)
            k_weight = k_weight.to(torch.device("cpu"), non_blocking=True)
            v_weight = v_weight.to(torch.device("cpu"), non_blocking=True)

            qkv_weight = get_full_tensor_from_distributed(qkv_weight)
            # qweight, weight_scale = ops.scaled_fp8_quant(qkv_weight.to(device, non_blocking=True), scale=None)
            qweight, weight_scale = ops.scaled_fp8_quant(qkv_weight.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
            new_updated_params[name.replace('q_proj.weight', 'qkv_proj.weight')] = qweight.t()
            new_updated_params[name.replace('q_proj.weight', 'qkv_proj.weight_scale')] = weight_scale

            del qkv_weight

        elif 'q_proj.bias' in name:
            q_bias = updated_params.pop(name)
            k_bias = updated_params.pop(name.replace('q_proj.bias', 'k_proj.bias'))
            v_bias = updated_params.pop(name.replace('q_proj.bias', 'v_proj.bias'))

            qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

            q_bias = q_bias.to(torch.device("cpu"), non_blocking=True)
            k_bias = k_bias.to(torch.device("cpu"), non_blocking=True)
            v_bias = v_bias.to(torch.device("cpu"), non_blocking=True)

            new_updated_params[name.replace('q_proj.bias', 'qkv_proj.bias')] = qkv_bias
        elif 'gate_proj.weight' in name:
            gate_weight = updated_params.pop(name)
            up_weight = updated_params.pop(name.replace('gate_proj.weight', 'up_proj.weight'))

            gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

            gate_weight = gate_weight.to(torch.device("cpu"), non_blocking=True)
            up_weight = up_weight.to(torch.device("cpu"), non_blocking=True)

            gate_up_weight = get_full_tensor_from_distributed(gate_up_weight)
            # qweight, weight_scale = ops.scaled_fp8_quant(gate_up_weight.to(device, non_blocking=True), scale=None)
            qweight, weight_scale = ops.scaled_fp8_quant(gate_up_weight.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
            new_updated_params[name.replace('gate_proj.weight', 'gate_up_proj.weight')] = qweight.t()
            new_updated_params[name.replace('gate_proj.weight', 'gate_up_proj.weight_scale')] = weight_scale

            del gate_up_weight

        # gate_up_prof.bias ?

        elif name.endswith("proj.weight") or name.endswith("qkv.weight") or name.endswith("fc1.weight") or name.endswith("fc2.weight"):
            if name in updated_params.keys():
                param = updated_params.pop(name)
                param = get_full_tensor_from_distributed(param)
                # qweight, weight_scale = ops.scaled_fp8_quant(param.to(device, non_blocking=True), scale=None)
                qweight, weight_scale = ops.scaled_fp8_quant(param.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
                new_updated_params[name] = qweight.t()
                new_updated_params[name.replace('.weight', '.weight_scale')] = weight_scale

    updated_params.update(new_updated_params)

    torch.cuda.empty_cache()

    return updated_params

@torch.no_grad()
def weight_fuse_and_quantize_generator(updatad_params):

    device = torch.cuda.current_device()

    weight_fuse_buffer = {}

    fuse_keys = [
        '.q_proj.weight',
        '.k_proj.weight',
        '.v_proj.weight',
    ]

    for name, param in updatad_params:

        if '.experts.' in name:
            if name.endswith("proj.weight"): # for gate, up, down
                qweight, weight_scale = ops.scaled_fp8_quant(param.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
                yield name, qweight # here is not .t()
                yield name.replace('.weight', '.weight_scale'), weight_scale

                continue

        if '.q_proj.weight' in name or '.k_proj.weight' in name or '.v_proj.weight' in name:
            weight_fuse_buffer[name] = param # .to(torch.device("cpu"), non_blocking=True)
        else:
            if name.endswith("proj.weight") or name.endswith("qkv.weight") or name.endswith("fc1.weight") or name.endswith("fc2.weight"):
                qweight, weight_scale = ops.scaled_fp8_quant(param.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
                yield name, qweight.t()
                yield name.replace('.weight', '.weight_scale'), weight_scale
            else:
                yield name, param
        
        name_prefix = '.'.join(name.split('.')[:-2])
        if name_prefix + '.q_proj.weight' in weight_fuse_buffer and name_prefix + '.k_proj.weight' in weight_fuse_buffer and name_prefix + '.v_proj.weight' in weight_fuse_buffer:

            q_weight = weight_fuse_buffer.pop(name_prefix + '.q_proj.weight')
            k_weight = weight_fuse_buffer.pop(name_prefix + '.k_proj.weight')
            v_weight = weight_fuse_buffer.pop(name_prefix + '.v_proj.weight')
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qweight, weight_scale = ops.scaled_fp8_quant(qkv_weight.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
            yield name_prefix + '.qkv_proj.weight', qweight.t()
            yield name_prefix + '.qkv_proj.weight_scale', weight_scale

            # torch.cuda.empty_cache()
        
        # if name_prefix + '.gate_proj.weight' in weight_fuse_buffer and name_prefix + '.up_proj.weight' in weight_fuse_buffer:
        #     gate_weight = weight_fuse_buffer.pop(name_prefix + '.gate_proj.weight')
        #     up_weight = weight_fuse_buffer.pop(name_prefix + '.up_proj.weight')
        #     gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)
        #     qweight, weight_scale = ops.scaled_fp8_quant(gate_up_weight.to(device, non_blocking=True), scale=None, use_per_token_if_dynamic=cutlass_fp8_supported())
        #     yield name_prefix + '.gate_up_proj.weight', qweight.t()
        #     yield name_prefix + '.gate_up_proj.weight_scale', weight_scale
            