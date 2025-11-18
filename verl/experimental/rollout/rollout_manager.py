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
import time
from functools import partial
import heapq
import importlib
import json
import logging
import os
import queue
import random
from concurrent.futures import Future
from abc import ABC, abstractmethod
import threading
from typing import Any, Optional
from PIL import Image

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoTokenizer, AutoProcessor
from uuid import uuid4

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.image_processors import image_convert
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_function
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.utils.torch_functional import pad_2d_list_to_length
from verl.workers.rollout.async_server import async_server_class, TokenOutput
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
def _process_samples_for_multimodal(multi_modal_data_batch, processor_path: str):
    processor = hf_processor(processor_path, trust_remote_code=True)
    results = []
    for multi_modal_data in multi_modal_data_batch:
        if multi_modal_data is None:
            image_list = []
            audio_list = []
        else:
            image_list = multi_modal_data.get("image", [])
            audio_list = multi_modal_data.get("audio", [])
        image_list = [image_convert(image) for image in image_list]
        output = processor.image_processor(images=image_list)
        multi_modal_inputs = {
            'pixel_values': output['pixel_values_flat'],
            'image_flags': output['image_flags']
        }

        if audio_list:
            audio_inputs = processor._preprocess_audio([''], audio_list)[1]
            audio_flags= torch.tensor([1] * len(audio_list), dtype=torch.long)
            audio_ret = dict(
                audio_values=audio_inputs['audio_values'],
                audio_len_after_cnn=audio_inputs['audio_len_after_cnn'],
                audio_token_num=audio_inputs['audio_token_num'],
                audio_flags=audio_flags,
            )
            multi_modal_inputs = multi_modal_inputs | audio_ret

        results.append(multi_modal_inputs)
    return results


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
        )
        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config

from dataclasses import dataclass, field
from enum import Enum

class SampleState(Enum):
    PENDING = "pending"
    FINISHED = "finished"
    TRUNCATED = "truncated"
    ABORTED = "aborted"

@dataclass
class Sample:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    rollout_log_probs: Optional[list[float]] = None

    multi_modal_data: Optional[dict[str, Any]] = None
    reward_score: Optional[float] = None
    num_turns: int = 0
    # metrics: AgentLoopMetrics = field(default_factory=AgentLoopMetrics)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    agent_name: str = None
    data_source: str = None
    reward_info: dict[str, Any] = None
    state: SampleState = SampleState.PENDING
    request_id: str = None
    aborted_response_length: int = 0 # for partial rollout

    index: int = -1
    rollout_n: int = -1

    user_turns: int = 0
    assistant_turns: int = 0

    staleness: int = -1

    uid: str = None

    @classmethod
    def from_dataset(cls, data: dict[str, Any], config: DictConfig, tokenizer: AutoTokenizer) -> "Sample":
        index = data['index'][0]
        # prompt_ids = data['raw_prompt_ids'].astype(int).squeeze().tolist()
        prompt_ids = data['input_ids'][0]
        prompt_ids = prompt_ids[prompt_ids != tokenizer.pad_token_id]
        prompt_ids = prompt_ids[:config.data.max_prompt_length]
        prompt_ids = prompt_ids.tolist()
        data_source = data['data_source'][0]
        reward_info = data['reward_model'][0]
        agent_name = data.get('agent_name', [None])[0] or config.actor_rollout_ref.rollout.agent.default_agent_loop

        if 'multi_modal_data' in data:
            multi_modal_data = data.get('multi_modal_data')[0]
        else:
            multi_modal_data = None

        return cls(
            index=index,
            rollout_n=0,
            prompt_ids=prompt_ids,
            agent_name=agent_name,
            data_source=data_source,
            reward_info=reward_info,
            uid=str(uuid4().hex),
            multi_modal_data=multi_modal_data,
        )

    @classmethod
    def to_dataproto(cls, samples: list["Sample"], config: DictConfig, tokenizer: AutoTokenizer, processor_path: str) -> DataProto:
        prompt_output = tokenizer.pad(
            {"input_ids": [sample.prompt_ids for sample in samples]},
            padding="max_length",
            max_length=config.data.max_prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
            padding_side="left",
        )
        response_output = tokenizer.pad(
            {"input_ids": [sample.response_ids for sample in samples]},
            padding="max_length",
            max_length=config.data.max_response_length,
            return_tensors="pt",
            return_attention_mask=True,
            padding_side="right",
        )
        response_mask = pad_2d_list_to_length(
            [sample.response_mask for sample in samples], 0, config.data.max_response_length
        )

        optional_outputs = {}
        if config.actor_rollout_ref.rollout.calculate_log_probs:
            optional_outputs["rollout_log_probs"] = pad_2d_list_to_length(
                [sample.rollout_log_probs for sample in samples], 0.0, config.data.max_response_length
            )

        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)


        batch = TensorDict(
            {
                "prompts": prompt_output["input_ids"],  # [bsz, prompt_length]
                "responses": response_output["input_ids"],  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(samples),
        )
        non_tensor_batch = {
            "__num_turns__": np.array([sample.assistant_turns for sample in samples]),
            "reward_model": np.array([sample.reward_info for sample in samples], dtype=object),
            "data_source": np.array([sample.data_source for sample in samples], dtype=object),
            "uid": np.array([sample.uid for sample in samples], dtype=object),
            "index": np.array([sample.index for sample in samples], dtype=object),
        }

        if any(sample.multi_modal_data is not None for sample in samples):
            # TODO: support internvl only for now
            if config.data.get('use_image_bytes', False):
                multi_modal_inputs_list = [{}] * len(samples)
                multi_modal_data_list = [sample.multi_modal_data for sample in samples]
            else:
                multi_modal_inputs_list = []
                multi_modal_data_list = [{}] * len(samples)

                assert processor_path is not None, "processor_path is required when multi_modal_data is not None"
                
                num_tasks = min(32, len(samples))  # Limit the number of concurrent tasks
                data_chunks = np.array_split([sample.multi_modal_data for sample in samples], num_tasks)
                tasks = [
                    # Note: processor can not be serialized, so we pass the processor_path instead
                    _process_samples_for_multimodal.remote(chunk.tolist(), processor_path)
                    for chunk in data_chunks if len(chunk) > 0
                ]
                for result_batch in ray.get(tasks):
                    multi_modal_inputs_list.extend(result_batch)

            non_tensor_batch["multi_modal_data"] = np.array(multi_modal_data_list, dtype=object)
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        scores = [sample.reward_score for sample in samples]
        if all(score is not None for score in scores):
            prompt_length = batch["prompts"].size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["token_level_scores"] = rm_scores

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [sample.extra_fields.get("reward_extra_info", {}) for sample in samples]
        all_keys = set()
        for info in reward_extra_infos:
            all_keys.update(info.keys())

        for key in all_keys:
            non_tensor_batch[key] = np.array([info.get(key) for info in reward_extra_infos], dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self, trainer_config: _DummyConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer, processor, **kwargs
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.init_class(trainer_config.config, tokenizer, processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process multi_modal data.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> Sample:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            Sample: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.reward_manager = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.loop = asyncio.get_event_loop()

    async def compute_score(
        self,
        data: DataProto,
    ) -> dict:
        """Compute reward score for agent loop output.

        NOTE: Since `reward_manager.__call__` is blocking function, we run it in thread pool to
        compute multiple samples in parallel.

        Args:
            data: reward function input

        Returns:
            dict: Reward score and reward extra info.
        """
        result = await self.loop.run_in_executor(
            None,
            self.reward_wrapper,
            data,
            True,  # return_dict
        )

        reward_score = result["reward_tensor"].sum(dim=-1).item()
        reward_extra_info = {k: v[0] for k, v in result.get("reward_extra_info", {}).items()}
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def reward_wrapper(self, data: DataProto, return_dict=False) -> torch.Tensor:
        """Assemble reward functions and reward model into one function and expose it to the event loop
        Args:
            return_dict: whether return as dict
            data: DataProto from compute reward score
        Returns:
            torch.Tensor: Reward score tensor.
        """
        return self.reward_manager(data, return_dict)

class GenerateState:
    def __init__(self):
        self._event = asyncio.Event()

    def reset(self):
        self._event.clear()

    def abort(self):
        self._event.set()

    def is_aborted(self):
        return self._event.is_set()

@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle]):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles)

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path)

        trace_config = config.trainer.get("rollout_trace", {})
        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

        self.generate_state = GenerateState()

    async def set_aborted(self):
        self.generate_state.abort()

    async def set_running(self):
        self.generate_state.reset()

    async def generate_sequences(self, samples: list[Sample], is_validate: bool = False) -> list[Sample]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
            max_tokens=config.response_length,
        )

        # override sampling params for validation
        if is_validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # skip invalid tokens generated in rollout
        invalid_token_ids = getattr(config, "invalid_token_ids", [])
        if len(invalid_token_ids) > 0:
            sampling_params["logit_bias"] = {}
            for token in invalid_token_ids:
                sampling_params["logit_bias"][int(token)] = float("-inf")

        tasks = []
        for i, sample in enumerate(samples):
            #TODO: pass `step` here
            trajectory_info = {"step": -1, "sample_index": sample.index, "rollout_n": sample.rollout_n, "validate": is_validate}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, trajectory_info, sample, self.generate_state)))
        outputs = await asyncio.gather(*tasks)

        return outputs

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        sample: Sample,
        generate_state: GenerateState,
    ) -> Sample:

        if sample.state == SampleState.FINISHED or sample.state == SampleState.TRUNCATED:
            return sample

        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            agent_name = sample.agent_name
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )
            sample = await agent_loop.run(sampling_params, sample, generate_state)

            if sample.state == SampleState.FINISHED:
                with simple_timer("reward", sample.metrics):
                    batch = TensorDict(
                        {
                            "prompts": torch.tensor(sample.prompt_ids).unsqueeze(0),  # [1, prompt_length]
                            "responses": torch.tensor(sample.response_ids).unsqueeze(0),  # [1, response_length]
                            # for compatibility with batch version of compute_score
                            "attention_mask": torch.ones(len(sample.prompt_ids) + len(sample.response_ids), dtype=torch.int32).unsqueeze(0),  # [1, prompt_length + response_length]
                        },
                        batch_size=1,
                    )
                    non_tensor_batch = {
                        "reward_model": np.array([sample.reward_info], dtype=object),
                        "data_source": np.array([sample.data_source], dtype=object),
                    }
                    reward_data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
                    try:
                        result = await asyncio.wait_for(
                            self.reward_manager_worker.compute_score.remote(reward_data),
                            timeout=self.config.actor_rollout_ref.rollout.reward_fn_timeout,
                        )
                        sample.reward_score = result["reward_score"]
                        sample.extra_fields["reward_extra_info"] = result["reward_extra_info"]
                    except asyncio.TimeoutError:
                        logger.warning("reward calculation timed out")
                        sample.reward_score = 0.0
                        sample.extra_fields["reward_extra_info"] = {}

            return sample


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class RolloutManager:
    """Rollout manager that manages a group of agent-loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, train_dataset, val_dataset):
        """Initialize rollout manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # LRU cache to map request_id to server
        self.ref_to_worker = LRUCache(maxsize=10000)

        self.epoch = 0
        self._create_dataloader(train_dataset, val_dataset)

        self.local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=True)
        self.processor = hf_processor(self.local_path, trust_remote_code=True)

        self.use_partial_rollout = self.config.actor_rollout_ref.rollout.partial_rollout.enable
        self.staleness_threshold = self.config.actor_rollout_ref.rollout.partial_rollout.staleness_threshold
        self.continuous_dispatch = self.config.actor_rollout_ref.rollout.continuous_dispatch

        if self.config.actor_rollout_ref.rollout.sample_filter_fn:
            self.sample_filter_fn = load_extern_function(self.config.actor_rollout_ref.rollout.sample_filter_fn)
        else:
            self.sample_filter_fn = None

    def _create_dataloader(self, train_dataset, val_dataset):
        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
        from torchdata.stateful_dataloader import StatefulDataLoader

        train_sampler = create_rl_sampler(self.config.data, train_dataset)
        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=1,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=default_collate_fn,
            sampler=train_sampler,
        )
        self.iter_train_dataset = iter(self.train_dataloader)
        self.data_buffer = []

        # For test
        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=default_collate_fn,
        )

    def _get_next_batch(self):
        if self.use_partial_rollout and len(self.data_buffer) > 0:
            return self.data_buffer.pop(0)

        try:
            data = next(self.iter_train_dataset)
        except StopIteration:
            print("reached end of train dataset, reset iterator")
            self.iter_train_dataset = iter(self.train_dataloader)
            data = next(self.iter_train_dataset)
            self.epoch += 1

        sample = Sample.from_dataset(data, self.config, self.tokenizer)
        samples = []
        for i in range(self.config.actor_rollout_ref.rollout.n):
            s = copy.deepcopy(sample)
            s.rollout_n = i
            s.request_id = uuid4().hex
            samples.append(s)

        return samples

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_tp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    print(f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting...")

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        for i in range(self.config.actor_rollout_ref.rollout.agent.num_workers):
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                ).remote(self.config, self.async_llm_servers)
            )

    def _staleness_filter(self, groups: list[list[Sample]]) -> list[list[Sample]]:
        return [group for group in groups if all(sample.staleness <= self.staleness_threshold for sample in group)]

    def generate_sequences(self) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        ray.get([worker.set_running.remote() for worker in self.agent_loop_workers])

        outputs = []
        pending_refs = []
        timing = {}

        # dispatch initial tasks
        for i in range(self.config.data.get("gen_batch_size", self.config.data.train_batch_size)):
            worker = self.agent_loop_workers[i % len(self.agent_loop_workers)]
            ref = worker.generate_sequences.remote(self._get_next_batch())
            pending_refs.append(ref)
            self.ref_to_worker[ref] = worker

        # dispatch tasks continuously
        while len(outputs) < self.config.data.train_batch_size:
            print(f"len(outputs): {len(outputs)}, len(pending_refs): {len(pending_refs)}")
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, fetch_local=True)

            while ready_refs:
                ref = ready_refs.pop(0)

                # The task is complete, we get the sample group and corresponding worker.
                group = ray.get(ref)
                worker = self.ref_to_worker.pop(ref, None)
                assert all(sample.state == SampleState.FINISHED for sample in group), "All samples must be finished"
                if not (self.sample_filter_fn and self.sample_filter_fn(group)):
                    outputs.append(group)

                # If the batch is full, stop processing and let the post-loop logic handle the rest.
                if len(outputs) >= self.config.data.train_batch_size:
                    break

                # If continuous dispatch is enabled, immediately schedule a new task
                if self.continuous_dispatch:
                    new_ref = worker.generate_sequences.remote(self._get_next_batch())
                    pending_refs.append(new_ref)
                    self.ref_to_worker[new_ref] = worker

        print(f"len(outputs): {len(outputs)}, len(pending_refs): {len(pending_refs)}")

        # abort all pending tasks
        with simple_timer("agent_loop/worker_abort", timing):
            ray.get([worker.set_aborted.remote() for worker in self.agent_loop_workers])

        time.sleep(1) # IMPORTANT: wait requests to be sent to vllm server
        with simple_timer("agent_loop/vllm_abort", timing):
            self.abort()
        
        logger.info("RolloutManager: waiting for pending tasks to complete...")
        with simple_timer("agent_loop/wait_for_pending", timing):
            # Collect results from leftover ready refs and clean up the cache    
            ready_groups = ray.get(ready_refs)
            try:
                aborted_groups = ray.get(pending_refs, timeout=10)
            except ray.exceptions.GetTimeoutError:
                logger.warning(f"RolloutManager: getting aborted groups timed out, aborting again...")
                self.abort()
                aborted_groups = ray.get(pending_refs)

            for ref in ready_refs + pending_refs:
                self.ref_to_worker.pop(ref, None)

            if self.use_partial_rollout:
                cache_groups = self._staleness_filter(ready_groups + aborted_groups)
                self.data_buffer.extend(cache_groups)

        logger.info("RolloutManager: pending tasks completed, waiting for sleep...")

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # flat samples
        samples = [sample for group in outputs for sample in group]

        # convert to dataproto
        with simple_timer("agent_loop/to_dataproto", timing):
            output = Sample.to_dataproto(samples, self.config, self.tokenizer, self.local_path)

        # calculate performance metrics
        metrics = [sample.metrics for sample in samples]  # List[Dict[str, str]]
        timing.update(self._performance_metrics(metrics, output))

        output.meta_info = {"timing": timing}
        return output

    def eval(self) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()

        ray.get([worker.set_running.remote() for worker in self.agent_loop_workers])

        pending_refs = []
        for i, data in enumerate(self.val_dataloader):
            sample = Sample.from_dataset(data, self.config, self.tokenizer)
            samples = []
            for i in range(self.config.actor_rollout_ref.rollout.val_kwargs.n):
                s = copy.deepcopy(sample)
                s.rollout_n = i
                s.request_id = uuid4().hex
                samples.append(s)

            worker = self.agent_loop_workers[i % len(self.agent_loop_workers)]
            ref = worker.generate_sequences.remote(samples, is_validate=True)
            pending_refs.append(ref)

        outputs = ray.get(pending_refs)
        ray.get([worker.set_aborted.remote() for worker in self.agent_loop_workers])
       
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

        # flat samples
        samples = [sample for group in outputs for sample in group]

        # convert to dataproto
        output = Sample.to_dataproto(samples, self.config, self.tokenizer, self.local_path)

        return output

    def _performance_metrics(self, metrics: list[dict[str, str]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for metric in metrics])
        t_tool_calls = np.array([metric.get("tool_calls", 0.) for metric in metrics])
        t_reward = np.array([metric.get("reward", 0.) for metric in metrics])
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()
        timing["agent_loop/reward/min"] = t_reward.min()
        timing["agent_loop/reward/max"] = t_reward.max()
        timing["agent_loop/reward/mean"] = t_reward.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/reward"] = t_reward[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])

    def abort(self):
        """Abort all rollout server instances."""
        ray.get([server.abort.remote() for server in self.async_llm_servers])
