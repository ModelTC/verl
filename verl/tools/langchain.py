import asyncio
import io
from langchain_core.utils.function_calling import convert_to_openai_tool
from PIL import Image
import urllib.request
import uuid
from verl.common import make_object_from_config
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.dataset.image_processors import adaptive_resize_image


def url_to_image(url: str):
    headers = {'User-Agent': 'Mozilla/5.0'}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers)
    ) as s:
        # ensure the image can be safely taken by the model
        image = Image.open(s)
        image = adaptive_resize_image(image, max_aspect_ratio=200)
        assert image.size[0] > 0, image.size[1] > 0
        with io.BytesIO() as f:
            image.save(f, "PNG")
            f.seek(0)
            return {
                "image": f.getvalue(),
                "image_wh": (image.size[0], image.size[1]),
            }


class LangChainTool(BaseTool):
    def __init__(self, config, tool_schema):
        self.tool = make_object_from_config(config["tool_config"])
        super().__init__(config, tool_schema)
        self.image_placeholder = config.get("image_placeholder", "<img></img>")
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
            error_text = repr(e)
            return error_text, None, error_text

        if isinstance(tool_message.content, str):
            return tool_message.content, None, None
        else:
            assert isinstance(tool_message.content, list)
            return ToolResponse(
                text="\n".join(
                    (
                        self.image_placeholder
                        if i["type"] == "image_url"
                        else i["text"]
                    )
                    for i in tool_message.content
                ),
                image=[
                    url_to_image(i["image_url"]["url"])
                    for i in tool_message.content
                    if i["type"] == "image_url"
                ]), None, None
