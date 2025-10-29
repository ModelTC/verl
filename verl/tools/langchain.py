from langchain_core.utils.function_calling import convert_to_openai_tool
from verl.common import make_object_from_config
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema


class LangChainTool(BaseTool):
    def __init__(self, config, tool_schema):
        self.tool = make_object_from_config(config["tool_config"])
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(**convert_to_openai_tool(self.tool))

    async def execute(
        self, instance_id: str, parameters: dict, **kwargs
    ) -> tuple[str, float, dict]:
        # TODO: convert instance_id to run_manager parameters

        result = self.tool.run(parameters)
        return result, 0, {}
