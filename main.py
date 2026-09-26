"""Nai series commands and silent, durable background image tools."""

import asyncio
import hashlib
import math
import re
import uuid
from dataclasses import asdict, replace
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.message.message_event_result import ResultContentType
from astrbot.core.platform.astrbot_message import MessageMember
from astrbot.core.star.filter.command import GreedyStr

from .background import ACTIVE, LABELS, BackgroundQueue
from .commands import generation_args, resolve_size
from .gallery import render_gallery, save_preview
from .models import (
    GenerationRequest,
    Preset,
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


class BatchCommandWakeFilter(filter.CustomFilter):
    def filter(self, event, cfg):
        return bool(event.is_at_or_wake_command)


class NaiSeriesPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.store = Store(StarTools.get_data_dir("astrbot_plugin_nai_series"))
        self.router = ProviderRouter(config)
        self.translator = PromptTranslator(config)
        self.queue = BackgroundQueue(self.store, config, self._process_job, self._completed)
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
        await self.queue.close()
        for name in self._tool_names:
            self.context.provider_manager.llm_tools.remove_func(name)

    async def initialize(self):
        manager = getattr(self.context, "platform_manager", None)
        if manager and getattr(manager, "platform_insts", []):
            self.queue.start()

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        self.queue.start()

    @property
    def pending(self):
        return sum(job["state"] in ACTIVE for job in self.store.jobs())

    @filter.on_llm_request()
    async def silence_tool_status(self, event: AstrMessageEvent, req):
        if event.get_extra("_nai_completion", False):
            if req.func_tool:
                for name in ("nai4_generate_image", "nai5_generate_image"):
                    req.func_tool.remove_tool(name)
            return
        recent = self.store.jobs(self._key(event), limit=4, origin="tool")
        facts = [
            f"{j['request']['model']}: {j['state']}, 原描述 {j.get('original_prompt', '')[:500]}, "
            f"实际提示词 {j['request']['prompt'][:700]}"
            for j in recent
            if j.get("state") in {"delivered", "generated", "sending", "send_unknown", "failed"}
        ]
        if facts:
            req.system_prompt = (req.system_prompt or "") + (
                "\n你最近为当前用户处理的生图记录(以任务状态为准, 不要声称看到了图像细节):\n"
                + "\n".join(facts)
            )
        if not self.config.get("enable_llm_tool", False):
            return
        req.system_prompt = (
            (req.system_prompt or "")
            + "\n调用 nai4_generate_image 或 nai5_generate_image 时直接调用工具, 不输出开场白或进度; 图片完成后会在原会话自然接话."
        )
        if event.get_extra("_nai_send_guard"):
            return
        original_send = event.send

        async def guarded_send(message):
            if getattr(message, "type", None) == "tool_call":
                content = str(getattr(message, "chain", ""))
                if any(name in content for name in ("nai4_generate_image", "nai5_generate_image")):
                    return
            if event.get_extra("_nai_silent"):
                message.chain = [part for part in message.chain if not isinstance(part, Plain)]
                if not message.chain:
                    return
            return await original_send(message)

        # Guard only this event, leaving other tools and all platform APIs untouched.
        event.send = guarded_send
        event.set_extra("_nai_original_send", original_send)
        event.set_extra("_nai_send_guard", True)
        original_stream = getattr(event, "send_streaming", None)
        if original_stream:

            async def guarded_stream(generator, use_fallback=False):
                # Do not leak streamed preambles before the model chooses its tool.
                buffered = [chain async for chain in generator]

                async def replay():
                    for chain in buffered:
                        if not event.get_extra("_nai_silent"):
                            yield chain

                return await original_stream(replay(), use_fallback)

            event.send_streaming = guarded_stream

    @filter.on_using_llm_tool()
    async def before_tool(self, event: AstrMessageEvent, tool, tool_args):
        if not event.get_extra("_nai_send_guard"):
            return
        pending = event.get_extra("_nai_preamble", [])
        event.set_extra("_nai_preamble", [])
        if tool.name in ("nai4_generate_image", "nai5_generate_image"):
            event.set_extra("_nai_silent", True)
        elif pending:
            await event.get_extra("_nai_original_send")(MessageChain(chain=pending))

    @filter.on_agent_done()
    async def agent_done(self, event: AstrMessageEvent, run_context, resp):
        event.set_extra("_nai_agent_done", True)

    @filter.on_decorating_result()
    async def silence_tool_reply(self, event: AstrMessageEvent):
        if not event.get_extra("_nai_send_guard") and not event.get_extra("_nai_silent"):
            return
        result = event.get_result()
        if not result:
            return
        if event.get_extra("_nai_silent"):
            result.chain = [part for part in result.chain if not isinstance(part, Plain)]
            if not result.chain:
                event.clear_result()
        elif result.result_content_type == ResultContentType.LLM_RESULT:
            pending = event.get_extra("_nai_preamble", [])
            if not event.get_extra("_nai_agent_done"):
                event.set_extra("_nai_preamble", pending + result.chain)
                event.clear_result()
            elif pending:
                result.chain = pending + result.chain
                event.set_extra("_nai_preamble", [])

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

    def _llm_preset(self, family):
        name = str(self.config.get(f"llm_preset_{family}", "")).strip()
        if not name:
            name = str(self.config.get(f"default_preset_{family}", "")).strip()
        return name or "none"

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

    def _submit(
        self,
        event,
        model,
        prompt,
        *,
        flags=None,
        snapshot=None,
        save_as=None,
        silent=False,
        count=1,
        completion=None,
        origin="command",
    ):
        self._check_access(event)
        flags = flags or {}
        if not snapshot and (not prompt.strip() or len(prompt) > 12000):
            raise ValueError("请提供 1-12000 字符的画面描述")
        preset = None if snapshot else self._preset(event, model, flags.get("preset"))
        draft = (
            GenerationRequest(**snapshot)
            if snapshot
            else self._request(model, prompt, preset, flags)
        )
        count = self._count(count)
        batch_id = uuid.uuid4().hex[:12] if count > 1 else ""
        payloads = [
            {
                "owner": self._key(event),
                "umo": event.unified_msg_origin,
                "request": asdict(draft),
                "original_prompt": prompt,
                "artist": preset.artist_prompt if preset else "",
                "raw": bool(snapshot or flags.get("raw")),
                "silent": silent,
                "preview": asdict(save_as) if save_as else None,
                "batch_id": batch_id,
                "batch_count": count,
                "completion": completion if origin == "tool" else None,
                "origin": origin,
            }
            for _ in range(count)
        ]
        ids = self.queue.submit_many(payloads)
        return ids[0] if count == 1 else ids

    def _count(self, value):
        try:
            count = int(value)
        except (TypeError, ValueError):
            raise ValueError("生图数量必须是整数") from None
        limit = max(1, min(8, int(self.config.get("batch_max_count", 4))))
        if not 1 <= count <= limit:
            raise ValueError(f"每次可生成 1-{limit} 张")
        return count

    async def _completion_target(self, event):
        manager = getattr(self.context, "conversation_manager", None)
        message = getattr(event, "message_obj", None)
        if manager is None or message is None:
            return None
        try:
            cid = await manager.get_curr_conversation_id(event.unified_msg_origin)
        except Exception:
            return None
        if not cid:
            return None
        return {
            "conversation_id": str(cid),
            "platform_id": event.get_platform_id(),
            "message_type": event.get_message_type().value,
            "self_id": str(message.self_id),
            "session_id": str(message.session_id),
            "sender_id": str(event.get_sender_id()),
            "sender_name": str(event.get_sender_name() or ""),
            "group_id": str(message.group_id or ""),
            "source_message_id": str(message.message_id or ""),
        }

    async def _completed(self, job):
        if job.get("origin") != "tool":
            return
        target = job.get("completion")
        if not target or job["state"] in ACTIVE:
            return
        batch_id = job.get("batch_id")
        jobs = (
            [j for j in self.store.jobs(job["owner"]) if j.get("batch_id") == batch_id]
            if batch_id
            else [job]
        )
        if len(jobs) != job.get("batch_count", 1) or any(j["state"] in ACTIVE for j in jobs):
            return
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return
        try:
            cid = await manager.get_curr_conversation_id(job["umo"])
            if str(cid or "") != target["conversation_id"]:
                return
            conversation = await manager.get_conversation(job["umo"], cid)
            adapter = self.context.get_platform_inst(target["platform_id"])
            if conversation is None or adapter is None:
                return
            facts = "\n".join(
                f"{j['id']}: 模型 {j['request']['model']}, 状态 {j['state']}, "
                f"用户描述 {j.get('original_prompt', '')[:500]}, "
                f"实际提示词 {j['request']['prompt'][:700]}"
                for j in reversed(jobs)
            )
            chain = [At(qq=target["self_id"])] if target["group_id"] else []
            if target.get("source_message_id"):
                chain.append(
                    Reply(
                        id=target["source_message_id"],
                        sender_id=target["sender_id"],
                        sender_nickname=target["sender_name"],
                        message_str="",
                    )
                )
            abm = await StarTools.create_message(
                type=target["message_type"],
                self_id=target["self_id"],
                session_id=target["session_id"],
                sender=MessageMember(user_id=target["sender_id"], nickname=target["sender_name"]),
                message=chain,
                message_str="",
                group_id=target["group_id"],
            )
            event = adapter.create_event(abm)
            event.is_wake = True
            event.is_at_or_wake_command = True
            event.set_extra("_nai_completion", True)
            event.set_extra("_nai_completion_batch", batch_id or job["id"])
            req = event.request_llm(
                prompt=(
                    "后台生图已结束. 根据以下真实任务记录, 用原有人格自然地对原用户说一句话. "
                    "图已经单独发送; 不再调用生图工具, 不要输出任务编号或机械的状态播报, "
                    "不要臆测提示词以外的画面细节.\n" + facts
                ),
                conversation=conversation,
            )
            if not self.store.claim("completion:" + (batch_id or job["id"])):
                return
            event.set_extra("provider_request", req)
            adapter.commit_event(event)
        except Exception:
            logger.exception("[NaiSeries] completion callback failed for task=%s", job["id"])

    async def _send(self, umo, components):
        matched = await asyncio.wait_for(
            self.context.send_message(umo, MessageChain(chain=components)), timeout=120
        )
        if matched is False:
            raise RuntimeError("platform_unavailable")

    async def _process_job(self, job, queue):
        try:
            if job["state"] != "generated":
                queue.transition(job, "translating")
                prompt = job["original_prompt"]
                translated = prompt if job["raw"] else await self.translator.translate(prompt)
                draft = GenerationRequest(**job["request"])
                request = (
                    draft
                    if job["raw"]
                    else replace(
                        draft,
                        prompt=", ".join(
                            p for p in (job["artist"].strip(), translated.strip()) if p
                        ),
                    )
                )
                if (
                    not job["silent"]
                    and translated != prompt
                    and self.config.get("show_translated_prompt", False)
                ):
                    await self._send(job["umo"], [Plain(f"翻译: {translated}")])
                queue.transition(job, "running", request=asdict(request))
                images, _route = await self.router.generate(request)
                if len(images) != 1:
                    raise GenerationError("接口返回图片数量与单张请求不一致")
                path = self.store.image_path(job["id"])
                temp = path.with_suffix(".tmp")
                temp.write_bytes(images[0].data)
                temp.chmod(0o600)
                temp.replace(path)
                state = self.store.get(job["owner"], {})
                state["last_request"] = asdict(request)
                self.store.put(job["owner"], state)
                queue.transition(job, "generated")
                if job["preview"]:
                    await asyncio.to_thread(
                        save_preview,
                        images[0].data,
                        self.store.preview_path(Preset(**job["preview"])),
                    )
            image = await asyncio.to_thread(self.store.image_path(job["id"]).read_bytes)
            queue.transition(job, "sending")
            await self._send(job["umo"], [Image.fromBytes(image)])
            queue.transition(job, "delivered")
            logger.info(
                "[NaiSeries] delivered task=%s model=%s", job["id"], job["request"]["model"]
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "[NaiSeries] task=%s stage=%s error=%s", job["id"], job["state"], type(exc).__name__
            )
            if not job["silent"]:
                try:
                    await self._send(
                        job["umo"], [Plain(f"任务 {job['id']} 未完成, 用 /nai任务 查看状态.")]
                    )
                except Exception:
                    pass
            raise

    async def _run(self, event, family, text, count_override=None):
        event.should_call_llm(True)
        try:
            prompt, flags = generation_args(text)
            flagged_count = flags.pop("count", None)
            if count_override is not None and flagged_count is not None:
                raise ValueError("批量指令已指定数量, 不要再写 --count")
            count = self._count(
                count_override
                if count_override is not None
                else flagged_count
                if flagged_count is not None
                else 1
            )
            model = self._model(flags.pop("model", family))
            if family in ("nai4", "nai5") and model_family(model) != family:
                raise ValueError(f"请使用 /{model_family(model)} 调用另一模型族")
            self._submit(
                event,
                model,
                prompt,
                flags=flags,
                count=count,
            )
            yield event.plain_result(f"已提交 {count} 张, 图片完成后自动发送.")
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

    @filter.regex(r"^\d+nai[45](?:\s|$)")
    @filter.custom_filter(BatchCommandWakeFilter)
    async def cmd_batch(self, event: AstrMessageEvent):
        match = re.fullmatch(r"(\d+)nai([45])(?:\s+([\s\S]*))?", event.get_message_str().strip())
        if match is None:
            return
        async for item in self._run(event, "nai" + match[2], match[3] or "", match[1]):
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
        event.should_call_llm(True)
        try:
            preset = self._query_preset(str(query))
            task_id = self._submit(
                event,
                self._model(preset.family),
                preset.preview_prompt,
                flags={"preset": preset.id, "size": preset.preview_size},
                save_as=preset,
            )
            yield event.plain_result(f"已提交样图任务 {task_id}.")
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
        event.should_call_llm(True)
        snapshot = self._state(event).last_request
        if not snapshot:
            yield event.plain_result("当前会话没有历史生图请求")
            return
        try:
            task_id = self._submit(event, snapshot["model"], "", snapshot=snapshot)
            yield event.plain_result(f"已提交重绘任务 {task_id}.")
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
    async def cmd_cancel(self, event: AstrMessageEvent, task_id: str = ""):
        count = await self.queue.cancel(self._key(event), task_id.strip())
        if count:
            yield event.plain_result(f"已取消 {count} 个本地任务; 已提交的任务仍可能计费.")
        else:
            yield event.plain_result("当前没有你的待处理任务")

    @filter.command("nai任务")
    async def cmd_tasks(self, event: AstrMessageEvent):
        jobs = self.store.jobs(self._key(event), limit=8)
        rejected = self.store.get("rejected:" + self._key(event), "")
        yield event.plain_result(
            "\n".join(
                f"{j['id']} | {j['request']['model']} | {LABELS.get(j['state'], j['state'])}"
                for j in jobs
            )
            + (f"\n上次未提交: {rejected}" if rejected else "")
            or "当前没有生图任务"
        )

    @filter.command("nai取图")
    async def cmd_retrieve(self, event: AstrMessageEvent, task_id: str):
        job = self.store.job(task_id.strip(), self._key(event))
        if not job or not self.store.image_path(job["id"]).is_file():
            yield event.plain_result("未找到可取回的图片, 请用 /nai任务 查看任务编号.")
            return
        if job["state"] in ACTIVE:
            yield event.plain_result("任务正在处理, 请稍后取图.")
            return
        data = await asyncio.to_thread(self.store.image_path(job["id"]).read_bytes)
        yield event.chain_result([Image.fromBytes(data)])

    @filter.command("nai帮助")
    async def cmd_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "Nai系列生图\n/nai4 描述 或 /nai5 描述\n批量: /3nai4 描述 或 /3nai5 描述\n选项放描述后: --count 3 --preset 名称 --raw --size 832x1216 --steps 23 --scale 5 --model 完整模型名\n/预设 nai4|nai5 名称 (none 关闭预设)\n/nai预设 [nai4|nai5] [页码]: 查看图册, 不产生生图费用\n/nai预设展示 [nai4|nai5] 名称 + 附图/引用图: 设置展示图(管理员)\n/nai预设试画 [nai4|nai5] 名称: 生成并保存样图(付费, 管理员)\n/nai模型 完整名 + /nai生图 描述\n/nai状态 /nai重绘 /nai任务\n/nai取消 [任务编号]: 取消自己当前会话的任务\n/nai取图 任务编号: 重新发送已保存图片, 不重新生图\n设置按当前用户和会话分别保存, 两模型预设互不影响."
        )

    async def generate_as_tool(self, event, family, kwargs):
        if event.get_extra("_nai_completion", False):
            return None
        event.set_extra("_nai_silent", True)
        event.clear_result()
        if not self.config.get("enable_llm_tool", False):
            return None
        try:
            count = self._count(kwargs.get("count", 1))
            prompt = str(kwargs.get("prompt", ""))
            preset = self._llm_preset(family)
            signature = hashlib.sha256(repr((family, prompt, count, preset)).encode()).hexdigest()
            submitted = event.get_extra("_nai_submitted", {})
            if signature in submitted:
                return None
            task_id = self._submit(
                event,
                self._model(family),
                prompt,
                flags={"preset": preset},
                silent=True,
                count=count,
                completion=await self._completion_target(event),
                origin="tool",
            )
            submitted[signature] = task_id
            event.set_extra("_nai_submitted", submitted)
            self.store.put("rejected:" + self._key(event), "")
        except (ValueError, GenerationError) as exc:
            event.set_extra("_nai_rejected", str(exc))
            self.store.put("rejected:" + self._key(event), str(exc))
            logger.info("[NaiSeries] tool rejected: %s", type(exc).__name__)
        except Exception:
            logger.warning("[NaiSeries] tool submission failed")
        # AstrBot treats None as a direct-result tool and ends the agent loop.
        return None
