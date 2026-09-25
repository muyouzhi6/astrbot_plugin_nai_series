"""Optional tools registered only when enabled in plugin configuration."""

from typing import Any

from astrbot.core.agent.tool import FunctionTool
from pydantic.dataclasses import dataclass


@dataclass
class NaiImageTool(FunctionTool):
    plugin: Any = None
    family: str = "nai4"

    async def call(self, context, **kwargs):
        return await self.plugin.generate_as_tool(context.context.event, self.family, kwargs)


def make_tools(plugin):
    return [
        NaiImageTool(
            name=f"{family}_generate_image",
            family=family,
            plugin=plugin,
            description=f"当用户明确要求用 {family} 生图时直接调用. 每次提交一张后台图片, 完成后自动发送, 用户可继续聊天. 调用前后不要输出提示、进度或完成说明. 保留用户原意, 不编造画师串或预设. 不确定预设时留空. 同一请求只提交一次, 无需轮询. nai4 指 4.5, nai5 指 5, 不得互换.",
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "用户原始画面描述, 保留数字和关系"},
                    "preset": {"type": "string", "description": "可选预设精确名称, 不知道则留空"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )
        for family in ("nai4", "nai5")
    ]
