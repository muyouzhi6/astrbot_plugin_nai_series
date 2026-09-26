"""Plugin behavior tests use real AstrBot when available."""

import asyncio
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_nai_series.background import BackgroundQueue
from astrbot_plugin_nai_series.commands import generation_args
from astrbot_plugin_nai_series.gallery import render_gallery, save_preview
from astrbot_plugin_nai_series.models import GenerationRequest, find_preset, parse_presets
from astrbot_plugin_nai_series.providers import GeneratedImage, GenerationError, ProviderRouter
from astrbot_plugin_nai_series.storage import Store
from PIL import Image

try:
    from astrbot.core.star.filter.command import CommandFilter, GreedyStr
    from astrbot_plugin_nai_series.main import NaiSeriesPlugin
    from astrbot_plugin_nai_series.tools import make_tools
    from astrbot_plugin_nai_series.translator import DEFAULT_TRANSLATOR_PROMPT, PromptTranslator
except ImportError:
    NaiSeriesPlugin = None


@unittest.skipIf(NaiSeriesPlugin is None, "AstrBot is not installed")
class TranslatorConfigTests(unittest.TestCase):
    def test_schema_default_matches_runtime_default(self):
        schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertTrue(DEFAULT_TRANSLATOR_PROMPT.strip())
        self.assertEqual(schema["translator_system_prompt"]["default"], DEFAULT_TRANSLATOR_PROMPT)

    def test_missing_or_blank_prompt_uses_default(self):
        for config in ({}, {"translator_system_prompt": ""}, {"translator_system_prompt": "  "}):
            with self.subTest(config=config):
                self.assertEqual(PromptTranslator(config).system_prompt, DEFAULT_TRANSLATOR_PROMPT)

    def test_custom_prompt_overrides_default(self):
        translator = PromptTranslator({"translator_system_prompt": "Custom prompt"})
        self.assertEqual(translator.system_prompt, "Custom prompt")

    def test_prefix_is_preserved_with_default_and_custom_prompt(self):
        for prompt in ("", "Custom prompt"):
            with self.subTest(prompt=prompt):
                translator = PromptTranslator(
                    {"translator_system_prompt": prompt, "translator_custom_prefix": "Prefix"}
                )
                self.assertEqual(
                    translator.system_prompt, f"Prefix\n\n{prompt or DEFAULT_TRANSLATOR_PROMPT}"
                )


def png():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), "orange").save(stream, format="PNG")
    return stream.getvalue()


class DomainTests(unittest.TestCase):
    def test_ambiguous_names_require_family(self):
        presets = parse_presets(
            [
                {"name": "日系", "model": "nai4", "artist_prompt": "A"},
                {"name": "日系", "model": "nai5", "artist_prompt": "B"},
            ]
        )
        with self.assertRaises(ValueError):
            find_preset(presets, "日系")
        self.assertEqual(find_preset(presets, "日系", "nai4").artist_prompt, "A")
        self.assertIsNone(find_preset(presets, "日", "nai4"))

    def test_flags_preserve_weights(self):
        text, flags = generation_args(
            "1.8::artist:foo::, a cat --preset 黑白 风格 --raw --size 640x640"
        )
        self.assertEqual(text, "1.8::artist:foo::, a cat")
        self.assertEqual(flags, {"preset": "黑白 风格", "raw": True, "size": "640x640"})

    def test_batch_count_flag(self):
        self.assertEqual(generation_args("猫 --count 3"), ("猫", {"count": "3"}))

    def test_invalid_generation_rejected(self):
        for kwargs in [{"steps": 0}, {"width": 833}, {"scale": 40}]:
            with self.assertRaises(ValueError):
                GenerationRequest("nai-diffusion-5-full", "cat", "bad", **kwargs)

    def test_empty_negative_is_explicit(self):
        preset = parse_presets(
            [{"name": "空负面", "model": "nai5", "artist_prompt": "", "negative_prompt": ""}]
        )[0]
        self.assertEqual(preset.negative_prompt, "")

    def test_storage_and_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            store.put("k", {"model": "nai5"})
            self.assertEqual(Store(Path(directory)).get("k"), {"model": "nai5"})
            presets = parse_presets(
                [
                    {"name": "画师一", "model": "nai4", "artist_prompt": "style"},
                    {"name": "画师二", "model": "nai5", "artist_prompt": "style"},
                ]
            )
            self.assertNotEqual(store.preview_path(presets[0]), store.preview_path(presets[1]))
            save_preview(png(), store.preview_path(presets[0]))
            result = render_gallery(presets, store, 1, 1)
            with Image.open(io.BytesIO(result)) as image:
                self.assertEqual(image.width, 900)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    def test_detect_official_by_url_without_provider_selector(self):
        router = ProviderRouter(
            {"primary_base_url": "https://image.novelai.net/v1", "primary_api_key": "test"}
        )
        from astrbot_plugin_nai_series.providers import OfficialNovelAIProvider

        self.assertIsInstance(router.providers[0], OfficialNovelAIProvider)
        self.assertEqual(router.providers[0].base_url, "https://image.novelai.net")

    async def test_no_retry_after_timeout_or_invalid_result(self):
        router = ProviderRouter({})
        first, second = AsyncMock(), AsyncMock()
        first.generate.side_effect = GenerationError("timeout")
        router.providers = [first, second]
        with self.assertRaises(GenerationError):
            await router.generate(GenerationRequest("nai-diffusion-5-full", "cat", ""))
        second.generate.assert_not_awaited()

    async def test_explicit_unavailable_route_can_switch_without_model_change(self):
        router = ProviderRouter({})
        first, second = AsyncMock(), AsyncMock()
        first.generate.side_effect = GenerationError("not found", 404)
        second.generate.return_value = [GeneratedImage(png())]
        router.providers = [first, second]
        request = GenerationRequest("nai-diffusion-5-full", "style, cat", "bad5")
        await router.generate(request)
        second.generate.assert_awaited_once_with(request)


class Event:
    unified_msg_origin = "QQ:GroupMessage:123"

    def __init__(self, user="1"):
        self.user = user
        self.messages = []
        self.extra = {}
        self.result = None
        self.call_llm = False

    def should_call_llm(self, value):
        self.call_llm = value

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def get_result(self):
        return self.result

    def clear_result(self):
        self.result = None

    def get_sender_id(self):
        return self.user

    def is_admin(self):
        return True

    def plain_result(self, text):
        return text

    def chain_result(self, chain):
        return chain

    async def send(self, value):
        self.messages.append(value)


@unittest.skipIf(NaiSeriesPlugin is None, "Run with AstrBot environment for plugin integration")
class PluginTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.plugin = NaiSeriesPlugin.__new__(NaiSeriesPlugin)
        self.plugin.config = {
            "user_cooldown": 0,
            "translator_enabled": False,
            "presets": [
                {
                    "name": "A",
                    "model": "nai4",
                    "artist_prompt": "style4",
                    "negative_prompt": "bad4",
                },
                {
                    "name": "B",
                    "model": "nai5",
                    "artist_prompt": "style5",
                    "negative_prompt": "bad5",
                },
            ],
            "default_preset_nai4": "A",
            "default_preset_nai5": "B",
            "steps_nai4": 23,
            "steps_nai5": 28,
        }
        self.plugin.store = Store(Path(self.directory.name))
        self.plugin.queue = BackgroundQueue(
            self.plugin.store, self.plugin.config, self.plugin._process_job
        )
        self.plugin.context = SimpleNamespace(send_message=AsyncMock(return_value=True))
        self.plugin.translator = PromptTranslator(self.plugin.config)
        self.plugin.router = AsyncMock()
        self.plugin.router.generate.return_value = ([GeneratedImage(png())], "test")

    async def asyncTearDown(self):
        await self.plugin.queue.close()
        self.directory.cleanup()

    async def finish(self):
        tasks = list(self.plugin.queue.tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 5)

    async def execute(self, event, model, prompt, **kwargs):
        task_id = self.plugin._submit(event, model, prompt, **kwargs)
        await self.finish()
        return self.plugin.store.job(task_id, self.plugin._key(event))

    async def test_nai4_and_nai5_commands_do_not_cross_profiles(self):
        event = Event()
        for family, handler in [("nai4", self.plugin.cmd_nai4), ("nai5", self.plugin.cmd_nai5)]:
            results = [item async for item in handler(event, "cat")]
            self.assertIn("已提交", results[0])
            await self.finish()
            request = self.plugin.router.generate.call_args.args[0]
            self.assertEqual(request.model, self.plugin._model(family))
            self.assertEqual(request.prompt, f"style{family[-1]}, cat")
            self.assertEqual(request.negative_prompt, f"bad{family[-1]}")
        self.assertEqual(self.plugin.context.send_message.await_count, 2)

    async def test_numbered_commands_generate_batch_without_bot_context(self):
        self.plugin.config.update(max_user_tasks=8, enable_llm_tool=True)
        event = Event()
        for family, count in (("nai4", 2), ("nai5", 3)):
            event.get_message_str = lambda family=family, count=count: f"{count}{family} 猫 --raw"
            results = [item async for item in self.plugin.cmd_batch(event)]
            self.assertIn(f"已提交 {count} 张", results[0])
            self.assertTrue(event.call_llm)
        await self.finish()
        jobs = self.plugin.store.jobs(self.plugin._key(event))
        self.assertEqual(len(jobs), 5)
        self.assertTrue(all(j["origin"] == "command" and j["completion"] is None for j in jobs))
        self.assertEqual(self.plugin.router.generate.await_count, 5)
        req = SimpleNamespace(system_prompt="original", func_tool=None)
        await self.plugin.silence_tool_status(event, req)
        self.assertNotIn("猫", req.system_prompt)

    async def test_numbered_command_rejects_duplicate_count_and_limit(self):
        self.plugin.config.update(max_user_tasks=8, batch_max_count=4)
        event = Event()
        for text, reason in (("5nai5 猫", "1-4"), ("3nai4 猫 --count 2", "不要再写")):
            event.get_message_str = lambda text=text: text
            results = [item async for item in self.plugin.cmd_batch(event)]
            self.assertIn(reason, results[0])
        self.assertEqual(self.plugin.store.jobs(), [])

    def test_numbered_command_requires_real_wake_and_full_token(self):
        from astrbot.core.star.filter.regex import RegexFilter
        from astrbot_plugin_nai_series.main import BatchCommandWakeFilter

        gate = BatchCommandWakeFilter()
        regex = RegexFilter(r"^\d+nai[45](?:\s|$)")
        event = SimpleNamespace(is_at_or_wake_command=False, get_message_str=lambda: "3nai5 猫")
        self.assertFalse(gate.filter(event, None))
        event.is_at_or_wake_command = True
        self.assertTrue(gate.filter(event, None))
        self.assertTrue(regex.filter(event, None))
        for text in ("请3nai5 猫", "3nai51 猫", "3nai5extra 猫"):
            event.get_message_str = lambda text=text: text
            self.assertFalse(regex.filter(event, None))

    def test_command_filter_greedy_argument_excludes_command_name(self):
        params = CommandFilter("nai5").validate_and_convert_params(
            ["a", "cat", "--raw"], {"prompt": GreedyStr}
        )
        self.assertEqual(params["prompt"], "a cat --raw")

    def test_user_isolation_and_restart_persistence(self):
        event, other = Event(), Event("2")
        self.plugin._choose(event, "nai5", "none")
        self.assertIsNone(self.plugin._preset(event, "nai5"))
        self.assertEqual(self.plugin._preset(other, "nai5").name, "B")
        self.plugin.store = Store(Path(self.directory.name))
        self.assertIsNone(self.plugin._preset(event, "nai5"))

    def test_editing_selected_preset_uses_latest_text(self):
        event = Event()
        self.plugin._choose(event, "nai5", "B")
        self.plugin.config["presets"][1]["artist_prompt"] = "new style"
        request = self.plugin._request(
            self.plugin._model("nai5"), "cat", self.plugin._preset(event, "nai5")
        )
        self.assertEqual(request.prompt, "new style, cat")

    async def test_failed_delivery_never_reports_success(self):
        event = Event()

        self.plugin.context.send_message.side_effect = RuntimeError("send failed")
        job = await self.execute(event, self.plugin._model("nai5"), "cat")
        self.assertEqual(job["state"], "send_unknown")
        self.assertEqual(self.plugin.pending, 0)
        self.assertFalse(self.plugin.queue.tasks)

    async def test_reroll_preserves_actual_prompt_after_config_change(self):
        event = Event()
        await self.execute(event, self.plugin._model("nai5"), "cat")
        snapshot = self.plugin._state(event).last_request
        self.plugin.config["presets"][1]["artist_prompt"] = "changed"
        await self.execute(event, snapshot["model"], "", snapshot=snapshot)
        self.assertEqual(asdict(self.plugin.router.generate.call_args.args[0]), snapshot)

    def test_curated_keeps_family_preset_without_overriding_model(self):
        self.plugin.config["model_nai4"] = "nai-diffusion-4-5-curated"
        model = self.plugin._model("nai4")
        request = self.plugin._request(model, "cat", self.plugin._preset(Event(), model))
        self.assertEqual(request.model, "nai-diffusion-4-5-curated")
        self.assertEqual(request.prompt, "style4, cat")

    async def test_disabled_tool_does_not_generate(self):
        text = await self.plugin.generate_as_tool(Event(), "nai5", {"prompt": "cat"})
        self.assertIsNone(text)
        self.plugin.router.generate.assert_not_awaited()

    async def test_enabled_tool_dispatches_correct_model(self):
        from types import SimpleNamespace

        self.plugin.config["enable_llm_tool"] = True
        tools = make_tools(self.plugin)
        self.assertEqual([t.name for t in tools], ["nai4_generate_image", "nai5_generate_image"])
        context = SimpleNamespace(context=SimpleNamespace(event=Event()))
        result = await tools[1].call(context, prompt="cat", preset="B")
        self.assertIsNone(result)
        await self.finish()
        self.assertEqual(
            self.plugin.router.generate.call_args.args[0].model, "nai-diffusion-5-full"
        )

    def test_tool_registration_switch(self):
        for enabled in (False, True):
            config = {"enable_llm_tool": enabled}
            context = MagicMock()
            with patch(
                "astrbot_plugin_nai_series.main.StarTools.get_data_dir",
                return_value=Path(self.directory.name),
            ):
                plugin = NaiSeriesPlugin(context, config)
            self.assertEqual(len(plugin._tool_names), 2 if enabled else 0)
            self.assertEqual(context.add_llm_tools.call_count, 1 if enabled else 0)

    async def test_blank_prompt_and_bad_options_never_charge(self):
        results = [v async for v in self.plugin.cmd_nai5(Event(), "")]
        self.assertTrue(results)
        self.plugin.router.generate.assert_not_awaited()
        results = [v async for v in self.plugin.cmd_nai4(Event(), "cat --model nai5")]
        self.assertTrue(results)
        self.plugin.router.generate.assert_not_awaited()

    async def test_concurrency_and_cancellation_release_slot(self):
        gate = asyncio.Event()

        async def pending(request):
            await gate.wait()
            return [GeneratedImage(png())], "test"

        self.plugin.router.generate.side_effect = pending
        event = Event()
        self.plugin._submit(event, "nai-diffusion-5-full", "cat")
        self.plugin._submit(event, "nai-diffusion-4-5-full", "another cat")
        await asyncio.sleep(0)
        self.assertEqual(self.plugin.router.generate.await_count, 2)
        with self.assertRaisesRegex(ValueError, "上限"):
            self.plugin._submit(event, "nai-diffusion-5-full", "third cat")
        await self.plugin.queue.cancel(self.plugin._key(event))
        self.assertEqual(self.plugin.pending, 0)
        self.assertFalse(self.plugin.queue.tasks)

    async def test_translation_failure_does_not_charge(self):
        self.plugin.translator.translate = AsyncMock(side_effect=ValueError("translation blocked"))
        job = await self.execute(Event(), "nai-diffusion-5-full", "一只猫")
        self.assertEqual(job["state"], "failed")
        self.plugin.router.generate.assert_not_awaited()

    async def test_gallery_never_calls_image_provider(self):
        result = [v async for v in self.plugin.cmd_gallery(Event(), "nai5")]
        self.assertTrue(result)
        self.plugin.router.generate.assert_not_awaited()

    async def test_bind_direct_preview(self):
        from astrbot.api.message_components import Image as Component

        target = Path(self.directory.name) / "reference.png"
        target.write_bytes(png())
        event = Event()
        event.get_messages = lambda: [Component.fromFileSystem(str(target))]
        result = [v async for v in self.plugin.cmd_bind_preview(event, "nai5 B")]
        self.assertIn("已保存", result[0])
        preset = self.plugin._query_preset("nai5 B")
        self.assertTrue(self.plugin.store.preview_path(preset).exists())
        self.plugin.router.generate.assert_not_awaited()
