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
    max_count = max(1, min(8, int(plugin.config.get("batch_max_count", 4))))
    return [
        NaiImageTool(
            name=f"{family}_generate_image",
            family=family,
            plugin=plugin,
            description=(
                f"用户明确要求用 NAI 画图时调用. 当前配置模型为 {family}, 不需要用户提供模型名. "
                "NAI 的正向 prompt 应是英文 Danbooru 风格 tags 和必要的简短英文短语, 用英文逗号分隔, 不直接把整段中文照搬进 prompt. "
                "先忠实理解用户描述, 再依次写出主体及数量、指定的外观/衣着、动作与对象关系、指定的场景和细节; "
                "例如 '两只猫在窗边玩球' -> '2 cats, playing with a ball, by the window'. "
                "数量、位置、否定与约束必须保留; 非标准标签可用简短英文短语表达, 不要硬造 tag. "
                "用户已给出的英文 tags 原样保留. 不擅自补外观、镜头、质量词、画师串或负面词, "
                "画师串和负面词由插件按配置面板固定的模型预设处理, 不要传入预设名或尝试切换预设. "
                f"count 按用户要求填写 1-{max_count}, 默认 1, 超过上限不可悄悄少画; "
                "nai4 是 4.5, nai5 是 5, 不得互换. 图片在后台发送, 调用前不输出提示, "
                "一次请求只调用一次, 不轮询."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "忠实转换后的英文 NAI 正向 tags/短语, 英文逗号分隔. "
                            "保留主体数量、动作及对象关系、衣着、位置、否定和用户明确细节; "
                            "不要添写画师串、负面词或未要求的设定. "
                            "例: 1girl, red dress, holding a book, sitting by the window"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": "用户要求的图片张数, 默认为 1",
                        "minimum": 1,
                        "maximum": max_count,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )
        for family in (plugin._llm_family(),)
    ]
