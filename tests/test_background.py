"""Lifecycle, admission, snapshots and real AstrBot silent tool execution."""

import asyncio
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import test_behavior as behavior
from astrbot_plugin_nai_series.background import BackgroundQueue
from astrbot_plugin_nai_series.models import GenerationRequest
from astrbot_plugin_nai_series.providers import GeneratedImage
from test_behavior import Event, png


@unittest.skipIf(
    behavior.NaiSeriesPlugin is None, "Run with AstrBot environment for plugin integration"
)
class BackgroundTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = behavior.PluginTests.asyncSetUp
    asyncTearDown = behavior.PluginTests.asyncTearDown
    finish = behavior.PluginTests.finish
    execute = behavior.PluginTests.execute

    async def test_batch_atomic_limits_and_concurrent_delivery(self):
        self.plugin.config.update(max_user_tasks=4, queue_limit=4, batch_max_count=4)
        gate = asyncio.Event()

        async def generate(request):
            await gate.wait()
            return [GeneratedImage(png())], "test"

        self.plugin.router.generate.side_effect = generate
        event = Event()
        ids = self.plugin._submit(event, "nai-diffusion-5-full", "cat", count=4)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(self.plugin.store.jobs()), 4)
        with self.assertRaisesRegex(ValueError, "队列已满"):
            self.plugin._submit(event, "nai-diffusion-5-full", "dog")
        await asyncio.sleep(0.05)
        self.assertEqual(self.plugin.router.generate.await_count, 2)
        gate.set()
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 4)
        self.assertEqual(self.plugin.context.send_message.await_count, 4)
        self.assertTrue(all(j["state"] == "delivered" for j in self.plugin.store.jobs()))

    async def test_batch_completion_once_and_conversation_change_gate(self):
        self.plugin.config.update(max_user_tasks=4, enable_llm_tool=True)
        self.plugin.queue.completed = self.plugin._completed
        conversation = object()
        manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="conversation-1"),
            get_conversation=AsyncMock(return_value=conversation),
        )
        adapter = SimpleNamespace(create_event=MagicMock(), commit_event=MagicMock())
        synthetic = Event()
        synthetic.request_llm = MagicMock(return_value=SimpleNamespace())
        adapter.create_event.return_value = synthetic
        self.plugin.context.conversation_manager = manager
        self.plugin.context.get_platform_inst = lambda platform_id: adapter
        target = dict(
            conversation_id="conversation-1",
            platform_id="QQ",
            message_type="GroupMessage",
            self_id="42",
            session_id="123",
            sender_id="1",
            sender_name="tester",
            group_id="123",
            source_message_id="456",
        )
        event = Event()
        ids = self.plugin._submit(
            event, "nai-diffusion-5-full", "猫", count=3, completion=target, origin="tool"
        )
        await self.finish()
        adapter.commit_event.assert_called_once()
        self.assertEqual(adapter.create_event.call_args.args[0].message[1].id, "456")
        self.assertEqual(synthetic.request_llm.call_args.kwargs["conversation"], conversation)
        self.assertIn("用户描述 猫", synthetic.request_llm.call_args.kwargs["prompt"])
        self.assertEqual(
            self.plugin.store.get(
                "completion:" + self.plugin.store.job(ids[0], self.plugin._key(event))["batch_id"]
            ),
            True,
        )
        req = SimpleNamespace(system_prompt="", func_tool=None)
        await self.plugin.silence_tool_status(event, req)
        self.assertIn("实际提示词", req.system_prompt)

        manager.get_curr_conversation_id.return_value = "conversation-2"
        self.plugin._submit(event, "nai-diffusion-5-full", "dog", completion=target, origin="tool")
        await self.finish()
        adapter.commit_event.assert_called_once()

    async def test_only_tool_jobs_appear_in_llm_context_even_after_commands(self):
        self.plugin.config.update(enable_llm_tool=True, max_user_tasks=8)
        event = Event()
        self.plugin._submit(event, "nai-diffusion-5-full", "bot画的猫", origin="tool")
        await self.finish()
        for number in range(5):
            self.plugin._submit(event, "nai-diffusion-5-full", f"用户指令{number}")
        await self.finish()
        req = SimpleNamespace(system_prompt="", func_tool=None)
        await self.plugin.silence_tool_status(event, req)
        self.assertIn("bot画的猫", req.system_prompt)
        self.assertNotIn("用户指令", req.system_prompt)

    async def test_command_task_never_dispatches_completion_even_with_target(self):
        self.plugin.queue.completed = self.plugin._completed
        event = Event()
        target = {"conversation_id": "unchanged"}
        self.plugin.context.conversation_manager = SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="unchanged")
        )
        self.plugin._submit(event, "nai-diffusion-5-full", "指令画的猫", completion=target)
        await self.finish()
        self.plugin.context.conversation_manager.get_curr_conversation_id.assert_not_awaited()

    async def test_invalid_batch_rejects_without_partial_submission(self):
        event = Event()
        with self.assertRaisesRegex(ValueError, "每次可生成"):
            self.plugin._submit(event, "nai-diffusion-5-full", "cat", count=9)
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_tool_returns_immediately_and_sends_only_image(self):
        self.plugin.config.update(enable_llm_tool=True, show_translated_prompt=True)
        self.plugin.translator.translate = AsyncMock(return_value="one orange cat")
        gate = asyncio.Event()

        async def generate(request):
            await gate.wait()
            return [GeneratedImage(png())], "test"

        self.plugin.router.generate.side_effect = generate
        event = Event()
        result = await asyncio.wait_for(
            self.plugin.generate_as_tool(event, "nai5", {"prompt": "一只橘猫"}), 0.2
        )
        self.assertIsNone(result)
        await asyncio.sleep(0)
        self.assertEqual(self.plugin.pending, 1)
        self.assertFalse(event.messages)
        self.plugin.context.send_message.assert_not_awaited()
        gate.set()
        await self.finish()
        from astrbot.api.message_components import Image

        calls = self.plugin.context.send_message.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[1].chain[0], Image)
        self.assertEqual(len(calls[0].args[1].chain), 1)

    async def test_real_executor_returns_none_without_sending_text(self):
        from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
        from astrbot_plugin_nai_series.tools import make_tools

        event = Event()
        self.plugin.config["enable_llm_tool"] = True
        context = SimpleNamespace(context=SimpleNamespace(event=event), tool_call_timeout=2)
        outputs = [
            value
            async for value in FunctionToolExecutor.execute(
                make_tools(self.plugin)[0], context, prompt="cat"
            )
        ]
        self.assertEqual(outputs, [None])
        self.assertFalse(event.messages)
        await self.finish()

    async def test_same_turn_duplicate_is_not_charged_twice(self):
        self.plugin.config["enable_llm_tool"] = True
        event = Event()
        for _ in range(2):
            await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat"})
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 1)

    async def test_fixed_tool_presets_ignore_session_and_model_arguments(self):
        self.plugin.config.update(enable_llm_tool=True, llm_preset_nai4="A", llm_preset_nai5="B")
        event = Event()
        self.plugin._choose(event, "nai4", "none")
        self.plugin._choose(event, "nai5", "none")
        for family, expected in (("nai4", "style4"), ("nai5", "style5")):
            self.plugin.config["llm_model"] = family
            await self.plugin.generate_as_tool(
                event, family, {"prompt": "cat", "preset": "nonexistent preset"}
            )
            await self.finish()
            request = self.plugin.router.generate.call_args.args[0]
            self.assertEqual(request.prompt, f"{expected}, cat")
            self.assertEqual(request.negative_prompt, "bad" + family[-1])
        command = await self.execute(event, "nai5", "cat")
        self.assertEqual(command["request"]["prompt"], "cat")

    async def test_blank_fixed_preset_uses_model_default_not_session(self):
        self.plugin.config.update(enable_llm_tool=True, llm_preset_nai5=" ")
        event = Event()
        self.plugin._choose(event, "nai5", "none")
        await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat"})
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].prompt, "style5, cat")

    async def test_fixed_none_disables_preset_even_with_session_choice(self):
        self.plugin.config.update(enable_llm_tool=True, llm_preset_nai5="none")
        event = Event()
        self.plugin._choose(event, "nai5", "B")
        await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat", "preset": "B"})
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].prompt, "cat")

    async def test_invalid_fixed_preset_rejects_without_generation(self):
        self.plugin.config.update(enable_llm_tool=True)
        for name in ("missing", "A"):
            self.plugin.config["llm_preset_nai5"] = name
            event = Event()
            await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat"})
            self.assertIn("不存在", event.get_extra("_nai_rejected"))
            self.assertIn("可用预设: B", event.get_extra("_nai_rejected"))
        self.assertEqual(self.plugin.store.jobs(), [])
        self.plugin.router.generate.assert_not_awaited()

    async def test_ignored_preset_argument_does_not_bypass_deduplication(self):
        self.plugin.config.update(enable_llm_tool=True, llm_preset_nai5="B")
        event = Event()
        for name in ("B", "wrong", ""):
            await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat", "preset": name})
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 1)

    async def test_rejection_is_visible_once_without_paid_request(self):
        self.plugin.config.update(enable_llm_tool=True, llm_preset_nai5="missing")
        event = Event()
        for _ in range(2):
            await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat"})
        self.plugin.context.send_message.assert_awaited_once()
        chain = self.plugin.context.send_message.call_args.args[1].chain
        self.assertIn("聊天生图预设 missing 不存在", chain[0].text)
        self.plugin.router.generate.assert_not_awaited()

    async def test_selected_model_overrides_stale_tool_and_command_is_unchanged(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config.update(
            enable_llm_tool=True, llm_model="nai4", model_nai4="nai-diffusion-4-5-curated"
        )
        self.assertEqual([t.name for t in make_tools(self.plugin)], ["nai4_generate_image"])
        event = Event()
        await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat", "model": "nai5"})
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.model, "nai-diffusion-4-5-curated")
        self.assertEqual(request.prompt, "style4, cat")
        self.assertEqual(request.negative_prompt, "bad4")
        job = await self.execute(event, self.plugin._model("nai5"), "cat")
        self.assertEqual(job["request"]["model"], "nai-diffusion-5-full")

    def test_model_switch_exposes_only_selected_tool(self):
        from astrbot_plugin_nai_series.tools import make_tools

        for family in ("nai4", "nai5"):
            self.plugin.config["llm_model"] = family
            tools = make_tools(self.plugin)
            self.assertEqual([t.name for t in tools], [family + "_generate_image"])
            self.assertNotIn("model", tools[0].parameters["properties"])
            self.assertNotIn("preset", tools[0].parameters["properties"])
        self.plugin.config["llm_model"] = "invalid"
        with self.assertRaisesRegex(ValueError, "聊天生图模型"):
            make_tools(self.plugin)

    async def test_llm_batch_count_and_same_turn_deduplication(self):
        self.plugin.config.update(enable_llm_tool=True, max_user_tasks=4)
        event = Event()
        kwargs = {"prompt": "三张橘猫", "count": 3}
        self.assertIsNone(await self.plugin.generate_as_tool(event, "nai5", kwargs))
        self.assertIsNone(await self.plugin.generate_as_tool(event, "nai5", kwargs))
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 3)
        self.assertEqual(self.plugin.context.send_message.await_count, 3)
        self.assertEqual(len(self.plugin.store.jobs(self.plugin._key(event))), 3)

    def test_tool_description_teaches_english_tags_and_matches_batch_limit(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["batch_max_count"] = 4
        for tool in make_tools(self.plugin):
            self.assertNotIn("preset", tool.parameters["properties"])
            self.assertIn("英文 Danbooru", tool.description)
            self.assertIn("数量、位置、否定", tool.description)
            self.assertIn("画师串和负面词由插件", tool.description)
            self.assertIn(
                "英文 NAI 正向 tags", tool.parameters["properties"]["prompt"]["description"]
            )
            self.assertEqual(tool.parameters["properties"]["count"]["maximum"], 4)
        self.plugin.config["batch_max_count"] = 6
        self.assertEqual(make_tools(self.plugin)[0].parameters["properties"]["count"]["maximum"], 6)

    async def test_snapshot_survives_preset_and_model_edits_in_queue(self):
        self.plugin.translator.translate = AsyncMock(return_value="translated cat")
        event = Event()
        self.plugin._submit(event, self.plugin._model("nai5"), "cat")
        self.plugin.config["presets"][1]["artist_prompt"] = "WRONG"
        self.plugin.config["presets"][1]["negative_prompt"] = "WRONG"
        self.plugin.config["model_nai5"] = "nai-diffusion-5-curated"
        await self.finish()
        req = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(
            (req.model, req.prompt, req.negative_prompt),
            ("nai-diffusion-5-full", "style5, translated cat", "bad5"),
        )

    async def test_global_capacity_counts_waiters_and_cancel_before_start(self):
        self.plugin.config.update(max_user_tasks=10, queue_limit=3)
        gate = asyncio.Event()
        self.plugin.router.generate.side_effect = lambda request: None

        async def generate(request):
            await gate.wait()
            return [GeneratedImage(png())], "test"

        self.plugin.router.generate.side_effect = generate
        event = Event()
        ids = [self.plugin._submit(event, "nai-diffusion-5-full", f"cat {n}") for n in range(3)]
        with self.assertRaisesRegex(ValueError, "队列已满"):
            self.plugin._submit(Event("2"), "nai-diffusion-5-full", "cat")
        await self.plugin.queue.cancel(self.plugin._key(event), ids[2])
        await asyncio.sleep(0)
        self.assertEqual(self.plugin.router.generate.await_count, 2)
        gate.set()
        await self.finish()
        self.assertEqual(self.plugin.pending, 0)
        self.assertEqual(
            self.plugin.store.job(ids[2], self.plugin._key(event))["state"], "cancelled"
        )

    async def test_restart_only_recovers_unsubmitted_or_generated(self):
        owner = self.plugin._key(Event())
        req = asdict(GenerationRequest("nai-diffusion-5-full", "style, cat", "bad5"))
        for state in ["queued", "translating", "running", "generated", "sending"]:
            self.plugin.store.save_job(
                {
                    "id": state,
                    "state": state,
                    "owner": owner,
                    "umo": Event.unified_msg_origin,
                    "request": req,
                    "created": time.time(),
                    "original_prompt": "cat",
                    "artist": "style",
                    "silent": True,
                    "raw": True,
                    "preview": None,
                }
            )
        self.plugin.store.image_path("generated").write_bytes(png())
        self.plugin.queue.start()
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 2)
        self.assertEqual(self.plugin.context.send_message.await_count, 3)
        self.assertEqual(self.plugin.store.job("running", owner)["state"], "interrupted")
        self.assertEqual(self.plugin.store.job("sending", owner)["state"], "send_unknown")

    async def test_two_owners_cannot_recover_same_jobs(self):
        self.plugin.queue.start()
        other = BackgroundQueue(self.plugin.store, {}, self.plugin._process_job)
        with self.assertRaisesRegex(ValueError, "已有"):
            other.start()
        await self.plugin.queue.close()
        other.start()
        await other.close()

    async def test_shutdown_does_not_repeat_paid_generation(self):
        gate = asyncio.Event()

        async def generate(request):
            await gate.wait()
            return [GeneratedImage(png())], "test"

        self.plugin.router.generate.side_effect = generate
        event = Event()
        task_id = self.plugin._submit(event, "nai-diffusion-5-full", "cat")
        await asyncio.sleep(0)
        await self.plugin.queue.close()
        self.assertEqual(
            self.plugin.store.job(task_id, self.plugin._key(event))["state"], "interrupted"
        )
        self.plugin.queue = BackgroundQueue(self.plugin.store, {}, self.plugin._process_job)
        self.plugin.queue.start()
        await self.finish()
        self.assertEqual(self.plugin.router.generate.await_count, 1)

    async def test_silent_failure_and_owner_scoped_retrieval(self):
        self.plugin.config["enable_llm_tool"] = True
        self.plugin.context.send_message.side_effect = TimeoutError("unknown")
        event = Event()
        await self.plugin.generate_as_tool(event, "nai5", {"prompt": "cat"})
        await self.finish()
        job = self.plugin.store.jobs()[0]
        self.assertEqual(job["state"], "send_unknown")
        self.assertEqual(self.plugin.context.send_message.await_count, 1)
        denied = [r async for r in self.plugin.cmd_retrieve(Event("other"), job["id"])]
        self.assertIn("未找到", denied[0])
        images = [r async for r in self.plugin.cmd_retrieve(event, job["id"])]
        self.assertIsInstance(images[0], list)
        self.assertEqual(self.plugin.router.generate.await_count, 1)

    async def test_tool_status_guard_leaves_other_tools_untouched(self):
        from astrbot.api.event import MessageChain

        self.plugin.config["enable_llm_tool"] = True
        event = Event()
        await self.plugin.silence_tool_status(event, SimpleNamespace(system_prompt=""))
        await event.send(MessageChain(type="tool_call").message("调用工具 nai5_generate_image"))
        await event.send(MessageChain(type="tool_call").message("调用工具 search"))
        self.assertEqual(len(event.messages), 1)

    async def test_none_platform_match_is_not_success(self):
        self.plugin.context.send_message.return_value = False
        job = await self.execute(Event(), "nai-diffusion-5-full", "cat", silent=True)
        self.assertEqual(job["state"], "send_unknown")

    async def test_terminate_uses_real_tool_manager_api_and_releases_owner(self):
        from astrbot.core.provider.func_tool_manager import FunctionToolManager
        from astrbot_plugin_nai_series.tools import make_tools

        manager = FunctionToolManager()
        self.plugin.context.provider_manager = SimpleNamespace(llm_tools=manager)
        self.plugin._tool_names = [tool.name for tool in make_tools(self.plugin)]
        self.plugin.queue.start()
        await self.plugin.terminate()
        self.assertIsNone(self.plugin.queue.lock)

    async def test_llm_preamble_suppressed_but_normal_final_preserved(self):
        from astrbot.core.message.message_event_result import MessageEventResult, ResultContentType

        self.plugin.config["enable_llm_tool"] = True
        event = Event()
        await self.plugin.silence_tool_status(event, SimpleNamespace(system_prompt=""))
        event.result = (
            MessageEventResult()
            .message("正在生成")
            .set_result_content_type(ResultContentType.LLM_RESULT)
        )
        await self.plugin.silence_tool_reply(event)
        self.assertIsNone(event.result)
        await self.plugin.before_tool(event, SimpleNamespace(name="nai5_generate_image"), {})
        await event.send(MessageEventResult().message("已生成"))
        self.assertFalse(event.messages)
        other = Event()
        await self.plugin.silence_tool_status(other, SimpleNamespace(system_prompt=""))
        await self.plugin.agent_done(other, None, None)
        other.result = (
            MessageEventResult()
            .message("正常聊天")
            .set_result_content_type(ResultContentType.LLM_RESULT)
        )
        await self.plugin.silence_tool_reply(other)
        self.assertEqual(other.result.chain[0].text, "正常聊天")

    async def test_streamed_preamble_is_not_sent_before_tool_selection(self):
        from astrbot.api.event import MessageChain

        self.plugin.config["enable_llm_tool"] = True
        event = Event()

        async def send_stream(generator, use_fallback=False):
            async for chain in generator:
                await event.send(chain)

        event.send_streaming = send_stream
        await self.plugin.silence_tool_status(event, SimpleNamespace(system_prompt=""))

        async def stream():
            yield MessageChain().message("正在画")
            await self.plugin.before_tool(event, SimpleNamespace(name="nai5_generate_image"), {})

        await event.send_streaming(stream(), True)
        self.assertFalse(event.messages)
