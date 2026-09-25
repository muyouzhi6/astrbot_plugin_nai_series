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
            description=f"当用户明确要求用 {family} 生图时调用. 每次生成一张并发送到当前会话. 保留用户原意, 不编造画师串或预设. 不确定预设时留空. 工具返回成功前不能声称图片已发送. nai4 指 4.5, nai5 指 5, 不得互换. 成本和权限与指令一致.",
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
