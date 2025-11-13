import asyncio
from langchain_core.utils.function_calling import convert_to_openai_tool
import uuid
from verl.common import make_object_from_config
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema


class LangChainTool(BaseTool):
    def __init__(self, config, tool_schema):
        self.tool = make_object_from_config(config["tool_config"])
        super().__init__(config, tool_schema)
        self.timeout = config.get("timeout", 30)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(**convert_to_openai_tool(self.tool))

    async def execute(
        self, instance_id: str, parameters: dict, **kwargs
    ) -> tuple[str, float, dict]:
        metadata = {"thread_id": instance_id}
        try:
            tool_message = await asyncio.wait_for(
                self.tool.arun(
                    parameters,
                    metadata=metadata,
                    config={"metadata": metadata},
                    tool_call_id=str(uuid.uuid4()),
                ),
                self.timeout,
            )

        except Exception as e:
            return repr(e), 0, {}

        return tool_message.content, 0, {}
