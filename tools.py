"""Optional tools registered only when enabled in plugin configuration."""

from typing import Any

from astrbot.core.agent.tool import FunctionTool
from pydantic.dataclasses import dataclass


@dataclass
class NaiImageTool(FunctionTool):
    plugin: Any = None
    family: str = "nai4"
    editing: bool = False

    async def call(self, context, **kwargs):
        if self.editing:
            kwargs["_img2img"] = True
        return await self.plugin.generate_as_tool(context.context.event, self.family, kwargs)


def make_tools(plugin):
    max_count = max(1, min(8, int(plugin.config.get("batch_max_count", 4))))
    result = [
        NaiImageTool(
            name=f"{family}_generate_image",
            family=family,
            plugin=plugin,
            description=(
                f"用户明确要求用 NAI 画图时调用. 当前配置模型为 {family}, 不需要用户提供模型名. "
                "NAI 的正向 prompt 应是英文 Danbooru 风格 tags 和必要的简短英文短语, 用英文逗号分隔. "
                "先忠实理解用户描述, 再依次写出主体及数量、指定的外观/衣着、动作与对象关系、指定的场景和细节; "
                "例如 '两只猫在窗边玩球' -> '2 cats, playing with a ball, by the window'. "
                "数量、位置、否定与约束必须保留; 非标准标签可用简短英文短语表达, 不要硬造 tag. "
                "用户已给出的英文 tags 原样保留. 画师名由插件配置的预设自动添加, 不要在 prompt 中写入画师名 (如 wlop, artgerm 等人名). "
                "负面词由预设处理, 不传入. 但可以添加镜头角度、光照、通用画风描述 (如 anime style, illustration, soft lighting, cinematic lighting) 和质量词 (masterpiece, best quality, very aesthetic, absurdres). "
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
                            "保留主体数量、动作及对象关系、衣着、位置、否定和用户明确细节. "
                            "可添加镜头角度 (cowboy shot, from below)、光照 (soft lighting)、通用画风 (anime style, illustration) 和质量词 (masterpiece, best quality, very aesthetic, absurdres). "
                            "不要写画师名 (wlop, artgerm 等人名由预设自动添加). "
                            "例: 1girl, red dress, holding a book, sitting by the window, soft lighting, anime style, masterpiece, best quality, very aesthetic"
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
    original = result[0]
    import copy

    parameters = copy.deepcopy(original.parameters)
    parameters["properties"].update(
        {
            "strength": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "重绘强度, 越大改动越大; 默认 0.6",
            },
            "noise": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "额外噪声, 默认 0.1",
            },
            "image_index": {
                "type": "integer",
                "minimum": 1,
                "maximum": 16,
                "description": "多图时按用户指定选择; 同消息在前, 引用在后; 从 1 开始, 不得猜测",
            },
        }
    )
    result.append(
        NaiImageTool(
            name=f"{original.family}_edit_image",
            family=original.family,
            plugin=plugin,
            editing=True,
            description=(
                "用户明确要求使用 NAI 修改附图或引用图片时调用. 只读取本轮附图、引用消息或合并转发中的图片, 不使用头像, 不接受 URL/路径. "
                "prompt 描述修改后的完整画面, 保留用户要求的主体和关系. 可添加镜头、光照、通用画风和质量词, 但不写画师名. 无图就提示补图, 不得改调文生图. "
                "图生图不是人物参考或氛围转移. " + original.description
            ),
            parameters=parameters,
        )
    )
    original.description = (
        "有本轮附图或引用图时自动图生图, 无图时文生图; 多张图按用户指定的 image_index 选择, 不得猜测. "
        "参考图读取失败不会改为文生图. 有参考图时 prompt 描述修改后的完整画面. "
        + original.description
    )
    original.parameters = copy.deepcopy(parameters)
    return result
