import asyncio
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_behavior as behavior
from astrbot_plugin_nai_series.models import GenerationRequest
from astrbot_plugin_nai_series.providers import OfficialNovelAIProvider, OpenAIImagesProvider
from astrbot_plugin_nai_series.references import collect_images, resolve_image
from test_behavior import Event, png
from test_core import CoreTests


def image_segment(data=None):
    return {
        "type": "image",
        "data": {"file": "base64://" + base64.b64encode(data or png()).decode()},
    }


def message_event(chain, raw=None):
    event = Event()
    event.message_obj = SimpleNamespace(message=chain, raw_message=raw)
    event.get_messages = lambda: chain
    event.bot = SimpleNamespace(call_action=AsyncMock(return_value={}))
    return event


class ReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_component_recovers_current_message(self):
        event = message_event([{"type": "image", "file": "/gone/reference.png"}])
        event.message_obj.message_id = 123
        event.bot.call_action.side_effect = [{}, {"message": [image_segment()]}]
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_awaited_with("get_msg", message_id=123)

    async def test_raw_reply_keeps_embedded_component(self):
        event = message_event(
            [{"type": "reply", "id": "123", "chain": [image_segment()]}],
            {"message": [{"type": "reply", "data": {"id": 123}}]},
        )
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_not_awaited()

    async def test_private_network_image_is_rejected(self):
        event = message_event([{"type": "image", "url": "http://127.0.0.1/secret.png"}])
        with self.assertRaisesRegex(ValueError, "读取失败"):
            await resolve_image(event)

    async def test_current_and_reply_image(self):
        for chain in (
            [image_segment()],
            [{"type": "reply", "id": "123", "chain": [image_segment()]}],
        ):
            event = message_event(chain)
            data = await resolve_image(event)
            from PIL import Image

            with Image.open(io.BytesIO(data)) as image:
                self.assertEqual(image.size, (64, 64))
            event.bot.call_action.assert_not_awaited()

    async def test_reply_fetched_by_id(self):
        event = message_event([{"type": "reply", "data": {"id": 456}}])
        event.bot.call_action.return_value = {"data": {"message": [image_segment()]}}
        self.assertTrue(await resolve_image(event))
        event.bot.call_action.assert_awaited_once_with("get_msg", message_id=456)

    async def test_raw_payload_recovers_cleaned_temp_file(self):
        event = message_event(
            [{"type": "image", "file": "/gone/test.png"}],
            {"message": [image_segment()]},
        )
        self.assertTrue(await resolve_image(event))
        self.assertEqual(len(await collect_images(event)), 1)

    async def test_media_id_and_forward(self):
        event = message_event([{"type": "forward", "data": {"id": "forward-1"}}])
        event.bot.call_action.side_effect = [
            {"messages": [{"content": [{"type": "image", "data": {"file": "opaque.png"}}]}]},
            {"url": image_segment()["data"]["file"]},
        ]
        self.assertTrue(await resolve_image(event))
        self.assertEqual(event.bot.call_action.call_args.args, ("get_image",))

    async def test_no_avatar_or_url_fallback(self):
        for chain in (
            [],
            [{"type": "at", "data": {"qq": "123"}}],
            [{"type": "text", "data": {"text": "https://example.org/a.png"}}],
        ):
            with self.assertRaisesRegex(ValueError, "附图"):
                await resolve_image(message_event(chain))

    async def test_multiple_requires_explicit_selection(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (128, 64), "blue").save(buffer, format="PNG")
        event = message_event([image_segment(), image_segment(buffer.getvalue())])
        with self.assertRaisesRegex(ValueError, "2 张图"):
            await resolve_image(event)
        self.assertTrue(await resolve_image(event, 2))
        for index in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                await resolve_image(event, index)

    async def test_duplicate_current_reply_and_cq_string(self):
        ref = image_segment()["data"]["file"]
        event = message_event([], {"message": f"[CQ:image,file={ref}]"})
        self.assertTrue(await resolve_image(event))
        event = message_event([image_segment(), {"type": "reply", "chain": [image_segment()]}])
        self.assertEqual(len(await collect_images(event)), 1)

    async def test_invalid_reference_does_not_generate(self):
        with self.assertRaisesRegex(ValueError, "读取失败"):
            await resolve_image(message_event([image_segment(b"not an image")]))

    async def test_real_astrbot_components(self):
        if behavior.NaiSeriesPlugin is None:
            self.skipTest("AstrBot unavailable")
        from astrbot.api.message_components import Image, Reply

        for chain in ([Image.fromBytes(png())], [Reply(id="123", chain=[Image.fromBytes(png())])]):
            self.assertTrue(await resolve_image(message_event(chain)))


class PayloadTests(CoreTests):
    def test_http_errors_keep_only_safe_correlation_fields(self):
        from astrbot_plugin_nai_series.providers import response_error

        response = SimpleNamespace(status=502, headers={})
        body = b'{"error":{"request_id":"abc123","message":"NAI upstream HTTP 500; not retried","secret":"do-not-log"}}'
        error = response_error(response, body)
        self.assertEqual(error.request_id, "abc123")
        self.assertIn("上游 HTTP 500", str(error))
        self.assertNotIn("do-not-log", str(error))
        body = b'{"error":{"request_id":"bad\\nheader","message":"a credential"}}'
        error = response_error(response, body)
        self.assertEqual(error.request_id, "")
        self.assertNotIn("credential", str(error))

    async def test_img2img_both_protocols(self):
        encoded = base64.b64encode(png()).decode()
        for cls in (OpenAIImagesProvider, OfficialNovelAIProvider):
            provider = cls("test", f"http://127.0.0.1:{self.port}", "test")
            request = GenerationRequest(
                "nai-diffusion-5-full",
                "cat",
                "",
                action="img2img",
                image=encoded,
                strength=0.7,
                noise=0.2,
            )
            self.assertTrue(await provider.generate(request))
            body = self.requests[-1][1]
            self.assertEqual(body["action"], "img2img")
            parameters = body.get("parameters", body)
            self.assertEqual(parameters["image"], encoded)
            self.assertEqual(parameters["strength"], 0.7)
            self.assertEqual(parameters["noise"], 0.2)

    def test_validation_and_sampler_whitespace(self):
        for steps in (29, 50, 0, True, 28.1):
            with self.assertRaisesRegex(ValueError, "1-28"):
                GenerationRequest("nai5", "cat", "", steps=steps)
        for kwargs in (
            {"action": "img2img"},
            {"strength": float("nan")},
            {"noise": 2},
            {"reference_id": "../file"},
        ):
            with self.assertRaises(ValueError):
                GenerationRequest("nai5", "cat", "", **kwargs)
        self.assertEqual(
            GenerationRequest("nai5", "cat", "", sampler=" k_dpmpp_2s_ancestral ").sampler,
            "k_dpmpp_2s_ancestral",
        )


@unittest.skipIf(behavior.NaiSeriesPlugin is None, "AstrBot unavailable")
class PluginImg2ImgTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = behavior.PluginTests.asyncSetUp
    asyncTearDown = behavior.PluginTests.asyncTearDown
    finish = behavior.PluginTests.finish

    async def test_parallel_tool_calls_enqueue_once(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        original = self.plugin._reference_flags
        gate = asyncio.Event()

        async def slow(*args):
            await gate.wait()
            return await original(*args)

        with patch.object(self.plugin, "_reference_flags", side_effect=slow):
            first = asyncio.create_task(tool.call(context, prompt="cat"))
            await asyncio.sleep(0)
            await tool.call(context, prompt="cat")
            gate.set()
            await first
        await self.finish()
        self.plugin.router.generate.assert_awaited_once()
        self.assertEqual(len(self.plugin.store.jobs()), 1)

    async def test_cancelled_tool_releases_submission_claim(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        with patch.object(self.plugin, "_reference_flags", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await tool.call(context, prompt="cat")
        self.assertEqual(event.get_extra("_nai_submitted"), {})
        self.plugin.router.generate.assert_not_awaited()

    async def test_command_persisted_reference_reroll_and_plain_legacy(self):
        event = message_event([image_segment()])
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --img2img --strength 0.8")]
        self.assertIn("已提交", str(result))
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.action, "img2img")
        self.assertTrue(request.image)
        self.assertEqual(request.strength, 0.8)
        snapshot = self.plugin._state(event).last_request
        self.assertEqual(snapshot["image"], "")
        self.assertTrue(self.plugin.store.reference_path(snapshot["reference_id"]).is_file())
        event.message_obj.message = []
        event.get_messages = lambda: []
        [v async for v in self.plugin.cmd_reroll(event)]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0], request)
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")

    async def test_missing_image_bad_steps_do_not_translate_or_charge(self):
        event = message_event([])
        for prompt in ("cat --img2img", "cat --img2img --steps 29", "cat --strength 0.4"):
            result = [v async for v in self.plugin.cmd_nai5(event, prompt)]
            self.assertIn("未完成", str(result))
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_commands_auto_edit_attached_image(self):
        for handler in (self.plugin.cmd_nai4, self.plugin.cmd_nai5, self.plugin.cmd_generate):
            event = message_event([image_segment()])
            result = [v async for v in handler(event, "cat --strength 0.45")]
            self.assertIn("已提交", str(result))
            await self.finish()
            request = self.plugin.router.generate.call_args.args[0]
            self.assertEqual(request.action, "img2img")
            self.assertEqual(request.strength, 0.45)
            self.assertTrue(request.image)

    async def test_command_auto_edit_quoted_image(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {"message": [image_segment()]}
        result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "img2img")
        event.bot.call_action.assert_awaited_once_with("get_msg", message_id=123)

    async def test_unknown_reference_does_not_fall_back_to_text(self):
        for chain in (
            [{"type": "reply", "data": {"id": 123}}],
            [{"type": "forward", "data": {"id": "forward-1"}}],
            [image_segment(b"invalid image")],
        ):
            event = message_event(chain)
            result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
            self.assertIn("未完成", str(result))
        self.plugin.router.generate.assert_not_awaited()
        self.assertEqual(self.plugin.store.jobs(), [])

    async def test_text_only_reply_uses_text_generation(self):
        event = message_event([{"type": "reply", "data": {"id": 123}}])
        event.bot.call_action.return_value = {
            "message": [{"type": "text", "data": {"text": "hello"}}]
        }
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")

    async def test_text_generation_does_not_lookup_current_message(self):
        event = message_event([])
        event.message_obj.message_id = 123
        [v async for v in self.plugin.cmd_nai5(event, "cat")]
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "generate")
        event.bot.call_action.assert_not_awaited()

    async def test_legacy_tool_auto_edits_and_deduplicates_edit_tool(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        generate, edit = make_tools(self.plugin)
        self.assertIn("image_index", generate.parameters["properties"])
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        await asyncio.gather(
            generate.call(context, prompt="cat"),
            edit.call(context, prompt="cat", strength=0.6, noise=0.1),
        )
        await self.finish()
        self.plugin.router.generate.assert_awaited_once()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "img2img")

    async def test_auto_mode_requires_selection_for_multiple_images(self):
        event = message_event([image_segment(), image_segment(b"second candidate")])
        result = [v async for v in self.plugin.cmd_nai5(event, "cat")]
        self.assertIn("2 张图", str(result))
        self.plugin.router.generate.assert_not_awaited()
        result = [v async for v in self.plugin.cmd_nai5(event, "cat --image 1")]
        self.assertIn("已提交", str(result))
        await self.finish()
        self.assertEqual(self.plugin.router.generate.call_args.args[0].action, "img2img")

    async def test_edit_tool_preserves_preset_and_deduplicates(self):
        from astrbot_plugin_nai_series.tools import make_tools

        self.plugin.config["enable_llm_tool"] = True
        event = message_event([image_segment()])
        tool = make_tools(self.plugin)[1]
        self.assertNotIn("url", tool.parameters["properties"])
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        for _ in range(2):
            await tool.call(context, prompt="cat", strength=0.5)
        await self.finish()
        request = self.plugin.router.generate.call_args.args[0]
        self.assertEqual(request.action, "img2img")
        self.assertEqual(request.prompt, "style5, cat")
        self.assertEqual(request.negative_prompt, "bad5")
        self.plugin.router.generate.assert_awaited_once()

    async def test_stale_reference_fails_before_provider(self):
        event = message_event([])
        self.plugin._submit(event, "nai5", "cat", flags={"img2img": True, "reference_id": "0" * 64})
        await self.finish()
        self.plugin.router.generate.assert_not_awaited()

    def test_references_are_owner_scoped_and_kept_for_reroll(self):
        store = self.plugin.store
        a = store.save_reference("a", png())
        b = store.save_reference("b", png())
        self.assertNotEqual(a, b)
        store.put("a", {"last_request": {"reference_id": a}})
        store.prune_jobs(float("inf"), set())
        self.assertTrue(store.reference_path(a).exists())
        self.assertFalse(store.reference_path(b).exists())
