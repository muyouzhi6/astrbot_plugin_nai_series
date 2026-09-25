"""Lifecycle, admission, snapshots and real AstrBot silent tool execution."""

import asyncio
import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
                make_tools(self.plugin)[1], context, prompt="cat"
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
