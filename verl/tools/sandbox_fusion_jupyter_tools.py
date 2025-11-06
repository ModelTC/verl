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

import base64
import io
import logging
import os
import re
import threading
import asyncio
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4
from PIL import Image

from cachetools import LRUCache
import random
import ray
from sandbox_fusion import run_jupyter_async, RunJupyterRequest

from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.dataset.image_processors import base64_to_pil, pil_to_base64, adaptive_resize_image, pil_to_base64, pil_to_bytes

from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class SandboxFusionJupyterTool(BaseTool):
    """A tool for executing the code using sanbox fusion image.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "code_interpreter",
                "description": "A tool for execute code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "code needs to be execute and grad",
                        },
                    },
                    "required": ["code"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        
        
        self.default_timeout = config.get("default_timeout", 300)
        self.default_max_attempts = config.get("default_max_attempts", 3)
        
        self.sandbox_address = config.get("sandbox_address")
        if self.sandbox_address == "":
            raise ValueError("sandbox_address is not set")

        self.max_cells_cache = config.get("max_cells_cache", 512)
        self._cells_cache = LRUCache(maxsize=self.max_cells_cache)

        self.image_tag = config.get("image_tag", "<img></img>")

        log_msg = f"Init SandboxFusionTool with config: {config}"
        logger.info(log_msg)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def reset_cache(self, instance_id: str):
        if instance_id in self._cells_cache:
            del self._cells_cache[instance_id]

    async def append_cache(self, instance_id: str, parameters: dict[str, Any]):
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)

        if instance_id not in self._cells_cache:
            self._cells_cache[instance_id] = []
        self._cells_cache[instance_id].append(code)

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
        }
        if instance_id not in self._cells_cache:
            self._cells_cache[instance_id] = []

        return instance_id

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        max_attempts = parameters.get("max_attempts", self.default_max_attempts)
        if not isinstance(code, str):
            code = str(code)

        cells = self._cells_cache[instance_id]
        cells.append(code)

        result, error = await self.sandbox_run_jupyter(cells, max_attempts, timeout)

        if not result: #TODO(yzx): check result
            return ToolResponse(text="", image=[]), None, error

        # === 3. 处理执行结果，构建 content (核心优化部分) ===
        # 采用更清晰的 if/elif/else 结构，一次性构建好最终的 content
        
        texts = []
        images = []
        # 情况一：执行成功，有输出（stdout 或 display 数据）
        if result.stdout or result.display:
            if result.stdout:
                texts.append(result.stdout)
            for item in result.display:
                try:
                    if "image/png" in item:
                        # 图像处理逻辑
                        loop = asyncio.get_running_loop()
                        processed_image_data = await loop.run_in_executor(
                            None, self._process_display_image, item['image/png']
                        )
                        images.append(processed_image_data)
                        texts.append(self.image_tag)
                    if 'text/plain' in item:
                        texts.append(item['text/plain'])
                except Exception as e:
                    # 如果单个 display item 处理失败，跳过它，而不是中断整个流程
                    print(e)
                    continue
            
            # 将收集到的文本和图片组装成最终的 content 格式
            # if texts:
            #     texts.insert(0, '<tool_response>')
            #     texts.append('</tool_response>')

        # 情况二：执行出错
        elif result.error and isinstance(result.error, list) and result.error:
            try:
                trace = '\n'.join(result.error[0].get('traceback', []))
                # 移除 ANSI 颜色代码
                ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
                clean_trace = ansi_escape.sub('', trace)
                texts.append(clean_trace)
            except (TypeError, AttributeError):
                # 如果错误格式不规范，返回一个通用错误信息
                texts.append("[An unknown error occurred during execution.]")
        
        # 情况三：执行成功，但没有任何输出
        else:
            texts.append("[Execution completed with no output.]")

        # === 4. 返回最终消息 ===
        # sandbox has no score or metrics, use Nones
        return ToolResponse(text='\n'.join(texts).strip(), image=images), None, None

    def _process_display_image(self, png_base64: str) -> dict:
        image = base64_to_pil(png_base64)
        process = False
        if hasattr(self, "_need_process"):
            wh = image.size
            if self._need_process(wh[0], wh[1]):
                process = True
        else:
            process = True
        if process:
            image = self._process_image(image)

        return {
            'image': pil_to_bytes(image),
            'image_wh': (image.size[0], image.size[1])
        }

    async def sandbox_run_jupyter(self, cells: list, max_attempts: int, timeout: int):
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
                client_timeout=timeout,
            )
        except Exception as e:
            print(f"sandbox exception: {e}, {type(e)=}")
            print("** error code **: \n", "\n".join(cells))
            return None, {"error": str(e)}

        if resp.cells:
            return resp.cells[-1], None
        else:
            print("[SandboxFusionJupyterTool] Warning: no cells in response")
            return None, {"error": "no cells in response"}

    def _process_image(self, image):
        image = adaptive_resize_image(image, max_aspect_ratio=200)
        return image
    
    def _need_process(self, w, h, max_aspect_ratio=200):
        longer, shorter = max(w, h), min(w, h)
        if longer / shorter > max_aspect_ratio:
            return True
        return False

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
