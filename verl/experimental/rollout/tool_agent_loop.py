# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import os
from enum import Enum
from typing import Any

from verl.experimental.rollout.rollout_manager import AgentLoopBase, register, Sample, SampleState, GenerateState
from verl.experimental.rollout.tool_parser import FunctionCall, ToolParser
from verl.experimental.rollout.utils import dedup_image_tokens
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.async_server import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    ABORTED = "aborted"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(self):
        self.multi_modal_data = None
        self.metrics = {}
        self.request_id = None
        self.tools_kwargs = {}

        # State variables
        self.prompt_ids: list[int] = []
        self.raw_prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.aborted_response_ids: list[int] = []

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        print(f"Initialized tools: {cls.tools}")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_prompt_length = config.data.max_prompt_length
        cls.max_response_length = config.data.max_response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        cls.terminate_on_tool_error = config.actor_rollout_ref.rollout.multi_turn.terminate_on_tool_error

    async def _init_tool_cache(self, request_id: str, response_ids: list[int]):
        for tool in self.tools.values():
            await tool.reset_cache(request_id)

        _, tool_calls = await self.tool_parser.extract_tool_calls(response_ids)
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)
                tool = self.tools[tool_name]
                await tool.append_cache(request_id, tool_args)
            except Exception as e:
                logger.warning(f"Error when appending tool cache: {e}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], sample: Sample, generate_state: GenerateState) -> Sample:
        agent_data = AgentData()

        # State machine loop
        state = AgentState.PENDING
        while state not in [AgentState.ABORTED, AgentState.TERMINATED]:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sample)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

            if generate_state.is_aborted() and state not in [AgentState.ABORTED, AgentState.TERMINATED]:
                state = AgentState.ABORTED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        sample.response_ids = response_ids[:self.max_response_length]
        sample.response_mask = agent_data.response_mask[:self.max_response_length]
        sample.rollout_log_probs = agent_data.response_logprobs[:self.max_response_length] if agent_data.response_logprobs else None
        sample.user_turns = agent_data.user_turns
        sample.assistant_turns = agent_data.assistant_turns
        sample.multi_modal_data = agent_data.multi_modal_data
        sample.metrics = agent_data.metrics
        sample.num_turns = agent_data.user_turns + agent_data.assistant_turns + 1
        sample.aborted_response_length = len(agent_data.aborted_response_ids)
        sample.extra_fields = {
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
        }
        if state == AgentState.ABORTED:
            sample.state = SampleState.ABORTED
        elif state == AgentState.TERMINATED:
            sample.state = SampleState.FINISHED
        
        return sample

    async def _handle_pending_state(self, agent_data: AgentData, sample: Sample) -> AgentState:
        """ init agent_data from sample """
        agent_data.request_id = sample.request_id
        agent_data.prompt_ids = sample.prompt_ids.copy()
        # agent_data.response_ids = sample.response_ids.copy()
        agent_data.response_mask = sample.response_mask.copy()
        agent_data.response_logprobs = sample.rollout_log_probs.copy() if sample.rollout_log_probs else []
        agent_data.multi_modal_data = sample.multi_modal_data
        agent_data.user_turns = sample.user_turns
        agent_data.assistant_turns = sample.assistant_turns

        # for partial rollout
        if sample.response_ids:
            agent_data.prompt_ids = sample.prompt_ids + sample.response_ids

            if sample.aborted_response_length > 0:
                finish_response_ids = sample.response_ids[:-sample.aborted_response_length]
                abort_response_ids = sample.response_ids[-sample.aborted_response_length:]
            else:
                finish_response_ids = sample.response_ids
                abort_response_ids = []
                
            agent_data.aborted_response_ids = abort_response_ids
            # For tool like jupyter, we need to initialize the tool cache for the next turn
            await self._init_tool_cache(sample.request_id, finish_response_ids)

        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            if agent_data.multi_modal_data:
                # NOTE: for vllm only, since vllm will replicate the image placeholder token by multi_modal_data.
                prompt_ids = dedup_image_tokens(agent_data.prompt_ids, self.processor)
            else:
                prompt_ids = agent_data.prompt_ids

            remaining_tokens = sampling_params["max_tokens"] - len(agent_data.aborted_response_ids)
            if remaining_tokens <= 0:
                output = TokenOutput(token_ids=[], log_probs=[], finish_reason="length")
            else:
                sampling_params = sampling_params.copy()
                sampling_params["max_tokens"] = remaining_tokens
                output = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    multi_modal_data=agent_data.multi_modal_data,
                )

        agent_data.response_ids = agent_data.aborted_response_ids + output.token_ids
        agent_data.aborted_response_ids = []
        agent_data.prompt_ids += output.token_ids
        agent_data.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Check termination conditions
        if len(agent_data.response_mask) >= self.max_response_length:
            agent_data.assistant_turns += 1
            return AgentState.TERMINATED

        if output.finish_reason == "abort":
            agent_data.aborted_response_ids = agent_data.response_ids.copy()
            return AgentState.ABORTED

        agent_data.assistant_turns += 1
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED

        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data.request_id))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        for tool_response, _, metrics in responses:
            if metrics and "error" in metrics:
                if self.terminate_on_tool_error:
                    return AgentState.TERMINATED

            # Create message from tool response
            message = {"role": "tool", "content": tool_response.text or ""}
            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                if agent_data.multi_modal_data is None or 'image' not in agent_data.multi_modal_data:
                    agent_data.multi_modal_data = {'image': []}
                elif not isinstance(agent_data.multi_modal_data['image'], list):
                    agent_data.multi_modal_data['image'] = [agent_data.multi_modal_data['image']]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_response.audio:
                #TODO(yzx): handle audio data
                raise NotImplementedError("Audio data is not currently supported.")


        # Process tool responses to token ids
        if self.processor is not None:
            if "InternVLProcessor" in self.processor.__class__.__name__:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        add_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                image_wh = [img['image_wh'] for img in new_images_this_turn]
                tool_response_with_image_token = self.processor.insert_media_placeholders_from_image_wh(raw_tool_response, image_wh)
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(tool_response_with_image_token, add_special_tokens=False),
                )
            else:
                raw_tool_response = await self.loop.run_in_executor(
                    None,
                    lambda: self.processor.apply_chat_template(
                        add_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                # Use only the new images from this turn for processing tool responses
                current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
                model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
                response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )

        response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.max_response_length:
            return AgentState.TERMINATED

        # Update prompt_ids, response_mask, multi_modal_data
        for img in new_images_this_turn:
            #TODO: remove this check in the future
            if isinstance(img, dict):
                agent_data.multi_modal_data['image'].append(img['image'])
            else:
                agent_data.multi_modal_data['image'].append(img)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], request_id: str,
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id = await tool.create(instance_id=request_id, create_kwargs=kwargs.get("create_kwargs", {}))
            assert instance_id == request_id
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
            if isinstance(tool_execution_response, str):
                tool_execution_response = ToolResponse(text=tool_execution_response)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
