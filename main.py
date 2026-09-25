"""Nai series commands: capture configuration once, await actual delivery."""

import asyncio
import hashlib
import math
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.star.filter.command import GreedyStr

from .commands import generation_args, resolve_size
from .gallery import render_gallery, save_preview
from .models import (
    GenerationRequest,
    SessionState,
    find_preset,
    model_family,
    normalize_model,
    parse_presets,
)
from .providers import GenerationError, ProviderRouter
from .storage import Store
from .tools import make_tools
from .translator import PromptTranslator


class NaiSeriesPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.store = Store(StarTools.get_data_dir("astrbot_plugin_nai_series"))
        self.router = ProviderRouter(config)
        self.translator = PromptTranslator(config)
        self.semaphore = asyncio.Semaphore(max(1, int(config.get("max_concurrency", 2))))
        self.pending = 0
        self.user_tasks = {}
        self.cooldowns = {}
        self._tool_names = []
        changed = False
        for preset in config.get("presets", []):
            if isinstance(preset, dict) and not preset.get("id"):
                preset["id"] = uuid.uuid4().hex
                changed = True
        if changed and callable(getattr(config, "save_config", None)):
            config.save_config()
        self._load_presets()
        if config.get("enable_llm_tool", False):
            tools = make_tools(self)
            self.context.add_llm_tools(*tools)
            self._tool_names = [tool.name for tool in tools]

    async def terminate(self):
        for name in self._tool_names:
            self.context.provider_manager.llm_tools.remove_tool(name)
        tasks = list(self.user_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _load_presets(self):
        return parse_presets(self.config.get("presets", []))

    def _key(self, event):
        return (
            "session:"
            + hashlib.sha256(
                f"{event.unified_msg_origin}:{event.get_sender_id()}".encode()
            ).hexdigest()
        )

    def _state(self, event):
        raw = self.store.get(self._key(event), {})
        return SessionState(
            model=raw.get("model", self.config.get("default_model", "nai4")),
            presets=raw.get("presets", {}),
            last_request=raw.get("last_request"),
        )

    def _persist(self, event, state):
        self.store.put(self._key(event), asdict(state))

    def _model(self, value):
        if value in ("nai4", "nai5"):
            model = normalize_model(self.config.get(f"model_{value}", value))
            if model_family(model) != value:
                raise ValueError(f"{value} 配置的模型名不属于该模型族")
            return model
        return normalize_model(value)

    def _preset(self, event, model, explicit=None):
        family = model_family(model)
        state = self._state(event)
        name = (
            explicit
            if explicit is not None
            else state.presets.get(family, self.config.get(f"default_preset_{family}", ""))
        )
        if name in ("", "none", "无"):
            return None
        preset = find_preset(self._load_presets(), str(name), model)
        if preset is None:
            raise ValueError(f"{family} 预设 {name} 已不存在, 请重新选择或 /预设 {family} none")
        return preset

    def _request(self, model, translated, preset, flags=None):
        flags = flags or {}
        family = model_family(model)
        width, height = resolve_size(
            str(flags.get("size", self.config.get(f"size_{family}", "竖图")))
        )
        artist = preset.artist_prompt if preset else ""
        negative = (
            preset.negative_prompt
            if preset and preset.negative_prompt is not None
            else str(self.config.get(f"negative_prompt_{family}", ""))
        )
        return GenerationRequest(
            model=model,
            prompt=", ".join(p for p in (artist.strip(), translated.strip()) if p),
            negative_prompt=negative,
            width=width,
            height=height,
            steps=int(flags.get("steps", self.config.get(f"steps_{family}", 28))),
            scale=float(flags.get("scale", self.config.get(f"scale_{family}", 5))),
            sampler=str(self.config.get(f"sampler_{family}", "k_euler_ancestral")),
            noise_schedule=str(self.config.get(f"noise_schedule_{family}", "karras")),
            cfg_rescale=float(self.config.get(f"cfg_rescale_{family}", 0)),
            quality=False,
        )

    def _check_access(self, event):
        if self.config.get("admin_only", False) and not event.is_admin():
            raise ValueError("当前仅允许管理员生图")
        allowed = [str(x) for x in self.config.get("allowed_users", [])]
        if allowed and event.get_sender_id() not in allowed and not event.is_admin():
            raise ValueError("当前用户未获得生图权限")

    async def _execute(self, event, model, prompt, *, flags=None, snapshot=None, save_as=None):
        self._check_access(event)
        key = self._key(event)
        flags = flags or {}
        if key in self.user_tasks:
            raise ValueError("你已有正在处理的生图请求")
        if self.pending >= max(1, int(self.config.get("queue_limit", 10))):
            raise ValueError("生图队列已满, 本次未提交")
        remaining = float(self.config.get("user_cooldown", 10)) - (
            time.monotonic() - self.cooldowns.get(key, 0)
        )
        if remaining > 0:
            raise ValueError(f"生图冷却剩余 {math.ceil(remaining)} 秒")
        if not snapshot and (not prompt.strip() or len(prompt) > 12000):
            raise ValueError("请提供 1-12000 字符的画面描述")
        # Snapshot the model, preset and parameters before waiting for the queue.
        preset = None if snapshot else self._preset(event, model, flags.get("preset"))
        draft = (
            GenerationRequest(**snapshot)
            if snapshot
            else self._request(model, prompt, preset, flags)
        )
        self.pending += 1
        self.user_tasks[key] = asyncio.current_task()
        try:
            if self.semaphore.locked():
                await event.send(event.plain_result(f"已排队, 当前处理中 {self.pending} 个请求"))
            async with self.semaphore:
                translated = (
                    prompt
                    if snapshot or flags.get("raw")
                    else await self.translator.translate(prompt)
                )
                request = (
                    draft
                    if snapshot
                    else replace(
                        draft,
                        prompt=", ".join(
                            p
                            for p in (
                                (preset.artist_prompt if preset else "").strip(),
                                translated.strip(),
                            )
                            if p
                        ),
                    )
                )
                if translated != prompt and self.config.get("show_translated_prompt", True):
                    await event.send(event.plain_result(f"翻译: {translated}"))
                await event.send(
                    event.plain_result(
                        f"生成中: {request.model} | {request.size} | {request.steps} 步 | 预设 {preset.name if preset else '无或重绘快照'}"
                    )
                )
                self.cooldowns[key] = time.monotonic()
                images, _route = await self.router.generate(request)
                if len(images) != 1:
                    raise GenerationError("接口返回图片数量与单张请求不一致")
                state = self._state(event)
                state.last_request = asdict(request)
                self._persist(event, state)
                if save_as:
                    await asyncio.to_thread(
                        save_preview, images[0].data, self.store.preview_path(save_as)
                    )
                await event.send(event.chain_result([Image.fromBytes(images[0].data)]))
                logger.info(
                    "[NaiSeries] delivered model=%s size=%s preset=%s",
                    request.model,
                    request.size,
                    preset.name if preset else "snapshot",
                )
                return f"图片已发送, 模型 {request.model}, 尺寸 {request.size}."
        finally:
            self.pending -= 1
            self.user_tasks.pop(key, None)

    async def _run(self, event, family, text):
        try:
            prompt, flags = generation_args(text)
            model = self._model(flags.pop("model", family))
            if family in ("nai4", "nai5") and model_family(model) != family:
                raise ValueError(f"请使用 /{model_family(model)} 调用另一模型族")
            await self._execute(event, model, prompt, flags=flags)
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(f"未完成: {exc}")
        except Exception:
            logger.exception("[NaiSeries] generation or delivery failed")
            yield event.plain_result(
                "生成或图片发送失败, 请检查插件日志; 为避免重复计费没有自动重试"
            )

    @filter.command("nai4")
    async def cmd_nai4(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, "nai4", str(prompt or "")):
            yield item

    @filter.command("nai5")
    async def cmd_nai5(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, "nai5", str(prompt or "")):
            yield item

    @filter.command("nai生图")
    async def cmd_generate(self, event: AstrMessageEvent, prompt: GreedyStr = GreedyStr):
        async for item in self._run(event, self._state(event).model, str(prompt or "")):
            yield item

    def _choose(self, event, family, name):
        model = self._model(family)
        if name not in ("none", "无"):
            preset = find_preset(self._load_presets(), name, model)
            if not preset:
                raise ValueError("没有此预设, 请使用 /nai预设 查看精确名称")
            name = preset.id
        else:
            name = "none"
        state = self._state(event)
        state.presets[model_family(model)] = name
        state.model = family
        self._persist(event, state)
        return f"已切换 {model_family(model)} 预设: {name}. 仅影响你在当前会话的设置."

    @filter.command("预设")
    async def cmd_preset(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            family, name = str(query).strip().split(maxsplit=1)
            yield event.plain_result(self._choose(event, family, name))
        except ValueError as exc:
            yield event.plain_result(f"{exc}. 用法: /预设 nai4|nai5 预设名")

    @filter.command("预设nai4")
    async def cmd_preset4(self, event: AstrMessageEvent, name: GreedyStr = GreedyStr):
        try:
            yield event.plain_result(self._choose(event, "nai4", str(name)))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("预设nai5")
    async def cmd_preset5(self, event: AstrMessageEvent, name: GreedyStr = GreedyStr):
        try:
            yield event.plain_result(self._choose(event, "nai5", str(name)))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai预设")
    async def cmd_gallery(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            args = str(query or "").split()
            presets = self._load_presets()
            if args and args[0] in ("nai4", "nai5"):
                family = args.pop(0)
                presets = [p for p in presets if p.family == family]
            pages = max(1, math.ceil(len(presets) / 6))
            page = int(args[0]) if args else 1
            if not 1 <= page <= pages:
                raise ValueError(f"页码必须在 1-{pages}")
            if not presets:
                yield event.plain_result("没有配置预设, 请在插件设置的预设列表添加")
                return
            selected = presets[(page - 1) * 6 : page * 6]
            data = await asyncio.to_thread(
                render_gallery,
                selected,
                self.store,
                page,
                pages,
                self.config.get("gallery_font", ""),
            )
            yield event.chain_result([Image.fromBytes(data)])
            yield event.plain_result("\n".join(f"{p.family}: {p.name}" for p in selected))
        except ValueError as exc:
            yield event.plain_result(str(exc))

    def _query_preset(self, query):
        parts = query.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0] in ("nai4", "nai5"):
            preset = find_preset(self._load_presets(), parts[1], parts[0])
        else:
            preset = find_preset(self._load_presets(), query)
        if not preset:
            raise ValueError("未找到预设, 请用 /nai预设 查看")
        return preset

    async def _attachment(self, event):
        for component in event.get_messages():
            if isinstance(component, Image):
                return await component.convert_to_file_path()
            if isinstance(component, Reply):
                for quoted in component.chain or []:
                    if isinstance(quoted, Image):
                        return await quoted.convert_to_file_path()
                from astrbot.core.utils.quoted_message import extract_quoted_message_images

                references = await extract_quoted_message_images(event, component)
                if references:
                    return (
                        await Image.fromURL(references[0]).convert_to_file_path()
                        if references[0].startswith("http")
                        else references[0]
                    )
        return None

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("nai预设展示")
    async def cmd_bind_preview(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            preset = self._query_preset(str(query))
            path = await self._attachment(event)
            if not path:
                yield event.plain_result(
                    "请附图或引用图, 再发送 /nai预设展示 [nai4|nai5] 预设名. 自动生成样图使用 /nai预设试画"
                )
                return
            data = await asyncio.to_thread(Path(path).read_bytes)
            await asyncio.to_thread(save_preview, data, self.store.preview_path(preset))
            yield event.plain_result(
                f"已保存 {preset.family}/{preset.name} 展示图. /nai预设 查看图册."
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("nai预设试画")
    async def cmd_sample(self, event: AstrMessageEvent, query: GreedyStr = GreedyStr):
        try:
            preset = self._query_preset(str(query))
            await self._execute(
                event,
                self._model(preset.family),
                preset.preview_prompt,
                flags={"preset": preset.id, "size": preset.preview_size},
                save_as=preset,
            )
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai模型")
    async def cmd_model(self, event: AstrMessageEvent, value: GreedyStr = GreedyStr):
        value = str(value or "").strip()
        state = self._state(event)
        if value:
            state.model = value
            self._persist(event, state)
        yield event.plain_result(
            f"/nai生图 当前模型: {self._model(state.model)}. /nai4 和 /nai5 始终使用各自模型配置."
        )

    @filter.command("nai状态")
    async def cmd_status(self, event: AstrMessageEvent):
        try:
            rows = []
            for family in ("nai4", "nai5"):
                model = self._model(family)
                preset = self._preset(event, model)
                request = self._request(model, "状态检查", preset)
                rows.append(
                    f"{family}: {model}\n预设: {preset.name if preset else '无'} | {request.size} | {request.steps} 步\n负面词: {len(request.negative_prompt)} 字符"
                )
            yield event.plain_result(
                "\n\n".join(rows)
                + f"\n处理中: {self.pending}\nLLM 工具: {'开' if self._tool_names else '关'}"
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai重绘")
    async def cmd_reroll(self, event: AstrMessageEvent):
        snapshot = self._state(event).last_request
        if not snapshot:
            yield event.plain_result("当前会话没有历史生图请求")
            return
        try:
            await self._execute(event, snapshot["model"], "", snapshot=snapshot)
        except (ValueError, GenerationError) as exc:
            yield event.plain_result(str(exc))

    @filter.command("nai参数")
    async def cmd_parameters(self, event: AstrMessageEvent):
        snapshot = self._state(event).last_request
        if not snapshot:
            yield event.plain_result("当前会话没有历史生图参数")
            return
        text = f"上次实际请求\n模型: {snapshot['model']}\n尺寸: {snapshot['width']}x{snapshot['height']} | 步数: {snapshot['steps']} | CFG: {snapshot['scale']}\n正向: {snapshot['prompt']}\n负面: {snapshot['negative_prompt']}"
        for start in range(0, len(text), 1800):
            yield event.plain_result(text[start : start + 1800])

    @filter.command("nai取消")
    async def cmd_cancel(self, event: AstrMessageEvent):
        task = self.user_tasks.get(self._key(event))
        if task and not task.done():
            task.cancel()
            yield event.plain_result("已取消本地等待; 若上游已接单, 可能仍生成并计费")
        else:
            yield event.plain_result("当前没有你的待处理任务")

    @filter.command("nai帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "Nai系列生图\n/nai4 描述 或 /nai5 描述\n选项放描述后: --preset 名称 --raw --size 832x1216 --steps 23 --scale 5 --model 完整模型名\n/预设 nai4|nai5 名称 (none 关闭预设)\n/nai预设 [nai4|nai5] [页码]: 查看图册, 不产生生图费用\n/nai预设展示 [nai4|nai5] 名称 + 附图/引用图: 设置展示图(管理员)\n/nai预设试画 [nai4|nai5] 名称: 生成并保存样图(付费, 管理员)\n/nai模型 完整名 + /nai生图 描述\n/nai状态 /nai重绘 /nai取消\n设置按当前用户和会话分别保存, 两模型预设互不影响."
        )

    async def generate_as_tool(self, event, family, kwargs):
        if not self.config.get("enable_llm_tool", False):
            return "未执行, LLM 生图工具未开启"
        try:
            return await self._execute(
                event,
                self._model(family),
                str(kwargs.get("prompt", "")),
                flags={"preset": kwargs["preset"]} if kwargs.get("preset") else {},
            )
        except (ValueError, GenerationError) as exc:
            return f"未完成: {exc}"
        except Exception:
            logger.exception("[NaiSeries] tool delivery failed")
            return "生成或发送失败, 没有自动重试"
