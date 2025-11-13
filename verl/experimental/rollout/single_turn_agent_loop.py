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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.rollout.rollout_manager import AgentLoopBase, SampleState, register, Sample
from verl.experimental.rollout.utils import dedup_image_tokens
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params: dict[str, Any], sample: Sample, generate_state, **kwargs) -> Sample:
        metrics = {}
        request_id = sample.request_id
        prompt_ids = sample.prompt_ids.copy()
        multi_modal_data = sample.multi_modal_data

        # for partial rollout
        if sample.response_ids:
            prompt_ids = prompt_ids + sample.response_ids
            sampling_params = sampling_params.copy()
            sampling_params["max_tokens"] -= len(sample.response_ids)
            assert sampling_params["max_tokens"] > 0, f"max_tokens is {sampling_params['max_tokens']}, {len(sample.response_ids)=}"

        with simple_timer("generate_sequences", metrics):
            if multi_modal_data:
                # NOTE: for vllm only, since vllm will replicate the image placeholder token by multi_modal_data.
                prompt_ids = dedup_image_tokens(prompt_ids, self.processor)
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, multi_modal_data=multi_modal_data
            )

        sample.response_ids += output.token_ids
        sample.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += output.log_probs

        sample.extra_fields["finish_reason"] = output.finish_reason
        sample.metrics = metrics
        sample.assistant_turns = 1

        if output.finish_reason == "abort":
            sample.state = SampleState.ABORTED
        elif output.finish_reason == "stop" or output.finish_reason == "length":
            sample.state = SampleState.FINISHED
        else:
            raise ValueError(f"Unknown finish reason: {output.finish_reason}")

        return sample
