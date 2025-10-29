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
import asyncio
import copy
import json
import time
from typing import Any, Dict, List
import random
import json
import os

import re
import aiohttp
from functools import partial

import torch
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.workers.rollout.legacy_chat_scheduler import ChatCompletionScheduler
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.profiler.performance import simple_timer

import io
import base64
import numpy as np
from PIL import Image
from sandbox_fusion import run_jupyter_async, RunJupyterRequest

from verl.utils.dataset.image_processors import adaptive_resize_image, preprocess_images_internvl, preprocess_images_qwenvl, pil_to_bytes
from verl.utils.dataset.image_processors import base64_to_pil, pil_to_base64

import copy


def repeat_each(lst, n):
    return [x for x in lst for _ in range(n)]


class ToolChatCompletionScheduler(ChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """
    def __init__(self, config, model_path, server_addresses, sandbox_address, data_config, max_turns=10, **kwargs):
        super().__init__(config, model_path, server_addresses, **kwargs)
        self.data_config = data_config
        self.sandbox_address = sandbox_address
        self.max_turns = max_turns
        self.max_prompt_length = data_config.max_prompt_length
        self.max_response_length = data_config.max_response_length

        self._chat_template_prefix_suffix = {
            "user": {
                "prefix_ids": list(self.tokenizer.encode("<|im_start|>user\n")),
                "suffix_ids": list(self.tokenizer.encode("<|im_end|>\n")),
                "suffix_ids_wo_eos": list(self.tokenizer.encode("\n")),
            },
            "assistant": {
                "prefix_ids": list(self.tokenizer.encode("<|im_start|>assistant\n")),
                "suffix_ids": list(self.tokenizer.encode("<|im_end|>\n")),
                "suffix_ids_wo_eos": list(self.tokenizer.encode("\n")),
            }
        }

        if type(self) is ToolChatCompletionScheduler:
            raise TypeError("ToolChatCompletionScheduler is an abstract class and should not be instantiated directly.")

    def _message_to_token(self, message):
        if isinstance(message['content'], str):
            token_ids = self.tokenizer.apply_chat_template(
                [message], add_generation_prompt=False, tokenize=True
            )
            token_ids_for_gen = token_ids

        elif isinstance(message['content'], list):
            image_wh = []
            text = ''
            for part in message['content']:
                assert isinstance(part, dict), f'part: {part}'
                if part['type'] == "image_url":
                    wh = (part['image_url']['w'], part['image_url']['h'])
                    image_wh.append(wh)
                elif part['type'] == "text":
                    text += part['text']

            token_ids_for_gen = self.tokenizer.apply_chat_template(
                [{'role': message['role'], 'content': text}], add_generation_prompt=False, tokenize=True
            )

            text = self._add_image_tokens_to_text(text, image_wh, image_wh=image_wh)
            new_message = {'role': message['role'], 'content': text}
            token_ids = self.tokenizer.apply_chat_template(
                [new_message], add_generation_prompt=False, tokenize=True
            )

        return token_ids, token_ids_for_gen

    async def sandbox_run_jupyter(self, cells: list, max_attempts: int=3, timeout: int=300):
        """Execute python code in sandbox."""
        if isinstance(self.sandbox_address, list):
            sandbox_address = random.choice(self.sandbox_address)
        elif isinstance(self.sandbox_address, str):
            sandbox_address = self.sandbox_address
        else:
            raise ValueError(f"Invalid sandbox_address: {self.sandbox_address}")
        
        try:
            resp = await run_jupyter_async(
                RunJupyterRequest(cells=cells, total_timeout=timeout),
                endpoint=f"http://{sandbox_address}",
                max_attempts=max_attempts,
                client_timeout=timeout+30,
            )
        except Exception as e:
            print(f"sandbox exception: {e}, {type(e)=}")
            return

        if resp.cells:
            return resp.cells[-1]
        else:
            print("[ToolChatCompletionScheduler] Warning: no cells in response")
            return

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=1,
            max_completion_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            logprobs=True,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample:
            kwargs.update({
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            })
        elif is_validate:
            kwargs.update({
                # "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            })

        # skip invalid tokens generated in rollout
        invalid_token_ids = getattr(self.config, "invalid_token_ids", [])
        if len(invalid_token_ids) > 0:
            sampling_params["logit_bias"] = {}
            for token in invalid_token_ids:
                sampling_params["logit_bias"][int(token)] = float("-inf")

        kwargs.update(sampling_params)
        print(f"[ToolChatCompletionScheduler] generate_sequences sampling params: {kwargs}")

        # 2. helper functions
        async def _process_tool_call(cells, resp):
            # === 1. 提取并验证工具调用 ===
            
            # 1.1 必须且只能找到一个 tool_call 块
            tool_calls = re.findall(r"<tool_call>\n(.*?)\n</tool_call>$", resp, re.DOTALL)
            if len(tool_calls) != 1 or resp.count("<tool_call>") != 1 or resp.count("</tool_call>") != 1:
                return

            # 1.2 tool_call 内容必须是合法的 JSON
            try:
                tool_call = json.loads(tool_calls[0])
            except json.JSONDecodeError:
                return

            # 1.3 tool_call 结构必须符合规范
            if not isinstance(tool_call, dict) or \
            'name' not in tool_call or \
            'arguments' not in tool_call or \
            len(tool_call.keys()) != 2:
                return

            # 1.4 目前只支持 'python' 工具
            if tool_call.get('name') != 'python':
                return

            if not isinstance(tool_call.get('arguments'), dict):
                return

            code = tool_call.get('arguments', {}).get('code')
            if not code:
                return

            # === 2. 执行代码并获取结果 ===
            cells.append(code)
            result = await self.sandbox_run_jupyter(cells)
            if not result:
                return

            # === 3. 处理执行结果，构建 content (核心优化部分) ===
            # 采用更清晰的 if/elif/else 结构，一次性构建好最终的 content
            
            content = []
            
            # 情况一：执行成功，有输出（stdout 或 display 数据）
            if result.stdout or result.display:
                texts = []
                images = []
                
                if result.stdout:
                    texts.append(result.stdout)
                    
                for item in result.display:
                    try:
                        if "image/png" in item:
                            # 图像处理逻辑
                            process = False
                            if hasattr(self, "_need_process"):
                                wh = Image.open(io.BytesIO(base64.b64decode(item['image/png']))).size
                                if self._need_process(wh[0], wh[1]):
                                    process = True
                            else:
                                process = True
                            if process:
                                image = base64_to_pil(item['image/png'])
                                image = self._process_image(image)
                                image_base64 = pil_to_base64(image)
                                wh = Image.open(io.BytesIO(base64.b64decode(image_base64))).size
                            else:
                                image_base64 = item['image/png']
                            
                            images.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}",
                                              "w": wh[0],
                                              "h": wh[1]}
                            })
                            texts.append(self.image_tag)
                        if 'text/plain' in item:
                            texts.append(item['text/plain'])
                    except Exception as e:
                        # 如果单个 display item 处理失败，跳过它，而不是中断整个流程
                        print(e)
                        continue
                
                # 将收集到的文本和图片组装成最终的 content 格式
                if texts:
                    texts.insert(0, '<tool_response>')
                    texts.append('</tool_response>')
                    content.append({"type": "text", "text": '\n'.join(texts).strip()})
                if images:
                    content.extend(images)

            # 情况二：执行出错
            elif result.error and isinstance(result.error, list) and result.error:
                try:
                    trace = '\n'.join(result.error[0].get('traceback', []))
                    # 移除 ANSI 颜色代码
                    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
                    clean_trace = ansi_escape.sub('', trace)
                    content = [{"type": "text", "text": clean_trace}]
                except (TypeError, AttributeError):
                    # 如果错误格式不规范，返回一个通用错误信息
                    content = [{"type": "text", "text": "<tool_response>\n[An unknown error occurred during execution.]\n</tool_response>"}]
            
            # 情况三：执行成功，但没有任何输出
            else:
                content = [{"type": "text", "text": "<tool_response>\n[Execution completed with no output.]\n</tool_response>"}]

            # === 4. 返回最终消息 ===
            if not content:
                return
            
            message = {"role": "user", "content": content}
            return message

        async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
            assert exception is None, f"exception: {exception}, {type(exception)=} {info['response_length']=}, {self.max_response_length=} {info['turn']=}"

            batch_conversations, batch_index, cells, turn, max_turns, response_length, metrics, current_kwargs = (
                info["batch_conversations"],
                info["batch_index"],
                info["cells"],
                info["turn"],
                info["max_turns"],
                info["response_length"],
                info["metrics"],
                info["kwargs"]
            )

            # process the response from completions
            assert len(completions.choices) == 1
            choice = completions.choices[0]
            role, content = choice.message.role, choice.message.content
            token_ids, logprobs = zip(*map(lambda x: (int(x.token[9:]), x.logprob), choice.logprobs.content))
            response_mask = [1] * len(token_ids)
            message = {
                "role": role,
                "content": content,
                "token_ids": list(token_ids),
                "logprobs": list(logprobs),
                "response_mask": response_mask,
            }
            message = self._update_message(message)

            batch_conversations[batch_index].append(message)
            response_length += len(message['token_ids_for_train'])

            if turn == max_turns or response_length > self.max_response_length:
                metrics['end_time'] = time.time()
                return

            # check if the response contains a tool call
            with simple_timer('tool_call', metrics):
                tool_message = await _process_tool_call(cells, content)
            if tool_message == None:
                metrics['end_time'] = time.time()
                return

            tool_message = self._update_message(tool_message)
            if response_length + len(tool_message['token_ids_for_train']) > self.max_response_length:
                metrics['end_time'] = time.time()
                return

            response_length += len(tool_message['token_ids_for_train'])
            batch_conversations[batch_index].append(tool_message)

            # resubmit chat completions with code block output
            extra_headers = {"x-request-id": completions.id}
            await self.submit_chat_completions(
                callback=callback,
                callback_additional_info={
                    "batch_conversations": batch_conversations,
                    "batch_index": batch_index,
                    "cells": cells,
                    "turn": turn + 1,
                    "max_turns": max_turns,
                    "response_length": response_length,
                    "metrics": metrics,
                    "kwargs": current_kwargs
                },
                model=self.model_name,
                messages=batch_conversations[batch_index] + [{"role": "assistant", "token_ids": self._chat_template_prefix_suffix['assistant']['prefix_ids']}],
                extra_headers=extra_headers,
                **current_kwargs,
            )


        # TODO: we may need to control max concurrent requests here, or it will harm prefix cache hit rate.
        tasks, batch_conversations = [], [None] * len(batch)
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        batch_metrics = [{}] * len(batch)

        # For aiohttp
        kwargs['skip_special_tokens'] = False

        for batch_index, conversation in enumerate(raw_prompts):
            if not isinstance(conversation, list):
                conversation = conversation.tolist()
            conversation = copy.deepcopy(conversation)
            conversation = [self._update_message(msg) for msg in conversation]
            batch_conversations[batch_index] = conversation
            batch_metrics[batch_index] = {'start_time': time.time()}

            if "max_turns" in batch.non_tensor_batch and batch.non_tensor_batch["max_turns"][batch_index] != -1:
                max_turns = batch.non_tensor_batch["max_turns"][batch_index]
            else:
                max_turns = self.max_turns
            if "single_response_length" in batch.non_tensor_batch and batch.non_tensor_batch["single_response_length"][batch_index] != -1:
                single_response_length = batch.non_tensor_batch["single_response_length"][batch_index]
            else:
                single_response_length = self.config.response_length

            new_kwargs = copy.deepcopy(kwargs)
            new_kwargs['max_completion_tokens'] = single_response_length
            new_kwargs['max_tokens'] = single_response_length

            tasks.append(
                asyncio.create_task(
                    self.submit_chat_completions(
                        callback=callback,
                        callback_additional_info={
                            "batch_conversations": batch_conversations,
                            "batch_index": batch_index,
                            "cells": [],
                            "turn": 1,
                            "max_turns": max_turns,
                            "response_length": 0,
                            "metrics": batch_metrics[batch_index],
                            "kwargs": new_kwargs
                        },
                        model=self.model_name,
                        messages=conversation + [{"role": "assistant", "token_ids": self._chat_template_prefix_suffix['assistant']['prefix_ids']}],
                        **new_kwargs,
                    )
                )
            )
        await asyncio.gather(*tasks)
        print(f"[{self.__class__.__name__}] generate_sequences done")

        post_start = time.time()
        output = self._postprocess(batch, batch_conversations)
        post_time = time.time() - post_start

        metrics_list = []
        for index, metrics in enumerate(batch_metrics):
            metrics['generate_sequences'] = metrics['end_time'] - metrics['start_time'] + post_time
            metrics['postprocess'] = post_time
            metrics.pop('start_time')
            metrics.pop('end_time')
            metrics_list.append(metrics)
        output.non_tensor_batch['metrics'] = np.array(metrics_list, dtype=object)

        return output

    def _postprocess(self, batch: DataProto, batch_conversations) -> DataProto:
        raw_prompts = batch.non_tensor_batch["raw_prompt"]

        batch_input_ids = []
        batch_resp_ids, batch_loss_mask, batch_logps = [], [], []
        multi_modal_inputs = []
        multi_modal_data = []
        for prompt, convs in zip(raw_prompts, batch_conversations):

            input_convs = convs[:len(prompt)]
            input_ids = []
            for message in input_convs:
                input_ids.extend(message['token_ids_for_train'])
            batch_input_ids.append(input_ids)

            resp_convs = convs[len(prompt):]
            resp_ids, loss_mask, logprobs = [], [], []
            whole_images = []

            for message in resp_convs:
                resp_ids.extend(message['token_ids_for_train'])
                loss_mask.extend(message['response_mask'])
                logprobs.extend(message['logprobs'])

            batch_resp_ids.append(resp_ids)
            batch_loss_mask.append(loss_mask)
            batch_logps.append(logprobs)

            for message in convs:
                if isinstance(message['content'], list):
                    for part in message['content']:
                        if 'image_url' in part:
                            image = part['image_url']['url'].split("base64,")[1]
                            if not self.data_config.get('use_image_bytes', False):
                                image = base64_to_pil(image)
                            whole_images.append(image)

            if self.data_config.get('use_image_bytes', False):
                multi_modal_data.append({"image": whole_images})
            else:
                image_inputs = self._build_multi_modal_input(whole_images)
                multi_modal_inputs.append(image_inputs)

        # prompts: [prompt] from input dataset
        # prompts = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False) for prompt in raw_prompts]
        # prompts = self.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=self.max_prompt_length, padding_side="left")
        
        input_ids = pad_2d_list_to_length(batch_input_ids, self.tokenizer.pad_token_id, max_length=self.max_prompt_length, padding_side="left")
        # input_attention_mask = get_response_mask(response_id=input_ids, eos_token=self.tokenizer.pad_token_id)
        prompts = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).to(torch.int64),
        }

        assert len(batch_conversations) == len(prompts["input_ids"])
        assert all(conversation[-1]['role'] == 'assistant' for conversation in batch_conversations)

        responses_ids = pad_2d_list_to_length(batch_resp_ids, self.tokenizer.pad_token_id, max_length=self.max_response_length)
        attention_mask = get_response_mask(response_id=responses_ids, eos_token=self.tokenizer.pad_token_id)
        responses = {"input_ids": responses_ids, "attention_mask": attention_mask}

        # clip response
        responses["input_ids"] = responses["input_ids"][:, :self.max_response_length]
        responses["attention_mask"] = responses["attention_mask"][:, :self.max_response_length]

        input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
        attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
        position_ids = self._get_position_ids(input_ids, attention_mask, multi_modal_inputs)

        batch_loss_mask = pad_2d_list_to_length(batch_loss_mask, 0, self.max_response_length)
        batch_loss_mask = batch_loss_mask[:, :self.max_response_length]
        batch_logps = pad_2d_list_to_length(batch_logps, 0.0, self.max_response_length)
        batch_logps = batch_logps[:, :self.max_response_length]

        batch = TensorDict(
            {
                "prompts": prompts["input_ids"],
                "responses": responses["input_ids"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "response_mask": batch_loss_mask,
                "rollout_log_probs": batch_logps,
            },
            batch_size=len(input_ids),
        )

        if self.data_config.get('use_image_bytes', False):
            multi_modal_inputs = [{}] * len(batch)
        else:
            multi_modal_data = [{}] * len(batch)

        non_tensor_batch = {
            "multi_modal_inputs": np.array(multi_modal_inputs, dtype=object),
            "multi_modal_data": np.array(multi_modal_data, dtype=object)
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    

    def _add_image_tokens_to_text(self, text, images=None, image_wh=None):
        """
        Add image tokens to the text based on the number of images.
        """
        _, text = self.preprocess_images_fn(images=images, text=text, image_wh=image_wh)
        return text
    
    def _build_multi_modal_input(self, images):
        """
        Build multi-modal input for InternVL.
        """
        image_inputs, _ = self.preprocess_images_fn(images=images)
        return image_inputs

    def _process_image(self, image):
        return image
    
    def _update_message(self, message):
        role = message['role']

        if role in ('user', 'system'):
            token_ids_for_train, token_ids_for_gen = self._message_to_token(message)
            message['token_ids'] = token_ids_for_gen
            message['token_ids_for_train'] = token_ids_for_train
            message['logprobs'] = [0.0] * len(token_ids_for_train)
            message['response_mask'] = [0] * len(token_ids_for_train)

        elif role == 'assistant':
            # get chat template
            prefix_ids = self._chat_template_prefix_suffix[role]['prefix_ids']
            if 'token_ids' in message:
                suffix_ids = self._chat_template_prefix_suffix[role]['suffix_ids_wo_eos'] if message['token_ids'][-1] == self.tokenizer.eos_token_id else self._chat_template_prefix_suffix[role]['suffix_ids']
                # apply chat template
                message['token_ids'] = prefix_ids + message['token_ids'] + suffix_ids
                message['token_ids_for_train'] = message['token_ids']
                message['logprobs'] = [0.0] * len(prefix_ids) + message['logprobs'] + [0.0] * len(suffix_ids)
                message['response_mask'] = [0] * len(prefix_ids) + message['response_mask'] + [0] * len(suffix_ids)
            else:
                token_ids_for_train, token_ids_for_gen = self._message_to_token(message)
                message['token_ids'] = token_ids_for_gen
                message['token_ids_for_train'] = token_ids_for_train
                message['logprobs'] = [0.0] * len(token_ids_for_train)
                message['response_mask'] = [0] * len(token_ids_for_train)

        else:
            raise ValueError(f"Unknown role: {role}")

        return message


class ToolChatCompletionScheduler_InternVL(ToolChatCompletionScheduler):

    image_tag = "<img></img>"

    def __init__(self, config, model_path, server_addresses, sandbox_address, data_config, max_turns=10, **kwargs):
        super().__init__(config, model_path, server_addresses, sandbox_address, data_config, max_turns, **kwargs)

        try:
            internvl_config_file = data_config['internvl_config_file']
        except KeyError:
            raise ValueError("`internvl_config_file` is required")
        assert os.path.exists(internvl_config_file), f"internvl_config_file {internvl_config_file} not found"
        internvl_config = json.load(open(internvl_config_file, 'r'))
        image_size = internvl_config.get('force_image_size', None) or internvl_config['vision_config']['image_size']
        patch_size = internvl_config['vision_config']['patch_size']
        downsample_ratio = internvl_config['downsample_ratio']
        self.preprocess_images_fn = partial(
            preprocess_images_internvl, 
            image_size=image_size, 
            min_dynamic_patch=internvl_config['min_dynamic_patch'], 
            max_dynamic_patch=internvl_config['max_dynamic_patch'], 
            use_thumbnail=internvl_config['use_thumbnail'], 
            num_image_token=int((image_size // patch_size)**2 * (downsample_ratio**2))
        )
        
    def _get_position_ids(self, input_ids, attention_mask, multi_modal_inputs):
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
        return position_ids
    
    def _process_image(self, image):
        image = adaptive_resize_image(image, max_aspect_ratio=200)
        return image
    
    def _need_process(self, w, h, max_aspect_ratio=200):
        longer, shorter = max(w, h), min(w, h)
        if longer / shorter > max_aspect_ratio:
            return True
        return False

class ToolChatCompletionScheduler_QwenVL(ToolChatCompletionScheduler):
    """
    A very naive implementation of ChatCompletionScheduler for demo purpose,
    only do single-turn chat completion.
    """

    image_tag = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, config, model_path, server_addresses, sandbox_address, data_config, max_turns=8, **kwargs):
        super().__init__(config, model_path, server_addresses, sandbox_address, data_config, max_turns, **kwargs)

        self.max_pixels = getattr(data_config, 'max_pixels', 1024 * 1024)
        self.processor = hf_processor(model_path, use_fast=True)
        self.preprocess_images_fn = partial(preprocess_images_qwenvl, processor=self.processor)
    
    def _process_image(self, image):
        image = adaptive_resize_image(
            image, 
            min_pixels=128*128,
            max_aspect_ratio=200,
            max_pixels=self.max_pixels,
            short_side=28)
        return image
    
    def _update_message(self, role, message):
        role = message['role']
        if role == 'assistant':
            extra_prefix = '<think>\n'
            extra_prefix_ids = self.tokenizer.encode(extra_prefix)

            message['content'] = extra_prefix + message['content']
            message['token_ids'] = extra_prefix_ids + message['token_ids']
            message['response_mask'] = [0] * len(extra_prefix_ids) + message['response_mask']
            message['logprobs'] = [0.0] * len(extra_prefix_ids) + message['logprobs']
        message = super()._update_message(message)
        return message

    def _get_position_ids(self, input_ids, attention_mask, multi_modal_inputs):
        position_ids = torch.stack([get_rope_index(
            self.processor,
            input_ids=input_ids[i],
            image_grid_thw=multi_modal_inputs[i]['image_grid_thw'],
            attention_mask=attention_mask[i],
        ) for i in range(len(input_ids))])
        return position_ids