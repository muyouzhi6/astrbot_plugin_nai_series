"""Exercise both relay transports with actual HTTP requests and no paid API calls."""

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientConnectionError, ClientTimeout, web
from astrbot_plugin_nai_series.models import GenerationRequest
from astrbot_plugin_nai_series.providers import (
    GenerationError,
    OfficialNovelAIProvider,
    OpenAIImagesProvider,
    ProviderRouter,
    download_image,
    openai_response_objects,
)
from test_behavior import NaiSeriesPlugin
from test_core import PNG

HAS_ASTRBOT = NaiSeriesPlugin is not None

B64 = base64.b64encode(PNG).decode()
URI = f"data:image/png;base64,{B64}"
REQUEST = GenerationRequest("nai-diffusion-5-full", "cat", "")


def chat(content):
    return {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]}


def sse(*objects):
    return (
        ": heartbeat\n\n"
        + "".join("data: " + json.dumps(item) + "\n\n" for item in objects)
        + "data: [DONE]\n\n"
    )


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.body = json.dumps({"data": [{"b64_json": B64}]}).encode()
        self.content_type = "application/json"
        self.status = 200
        self.download_status = 200
        self.downloads = []

        async def generate(request):
            self.requests.append((request.path, await request.json(), dict(request.headers)))
            if self.content_type == "text/event-stream":
                response = web.StreamResponse(
                    status=self.status, headers={"Content-Type": self.content_type}
                )
                await response.prepare(request)
                for offset in range(0, len(self.body), 17):
                    await response.write(self.body[offset : offset + 17])
                    await asyncio.sleep(0)
                await response.write_eof()
                return response
            return web.Response(body=self.body, status=self.status, content_type=self.content_type)

        async def image(request):
            self.downloads.append(request.path)
            return web.Response(body=PNG, status=self.download_status, content_type="image/png")

        app = web.Application()
        app.router.add_post("/v1/images/generations", generate)
        app.router.add_post("/v1/chat/completions", generate)
        app.router.add_get("/image.png", image)
        app.router.add_get("/image(with-parentheses).png", image)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        self.image_url = self.base_url + "/image.png"
        self.provider = OpenAIImagesProvider("mock", self.base_url + "/v1", "fake-credential")
        # Production still rejects private downloads. The test enables only this mock GET.
        self.public_ip = patch(
            "astrbot_plugin_nai_series.providers.ipaddress",
            SimpleNamespace(ip_address=lambda _: SimpleNamespace(is_global=True)),
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def set_json(self, value):
        self.content_type = "application/json"
        self.body = json.dumps(value).encode()

    async def assert_image(self):
        with self.public_ip:
            images = await self.provider.generate(REQUEST)
        self.assertEqual([item.data for item in images], [PNG])

    async def test_default_missing_field_and_all_three_routes(self):
        for stream in (None, False, True):
            config = {}
            for name in ("primary", "secondary", "third"):
                config[f"{name}_base_url"] = self.base_url + "/v1"
                config[f"{name}_api_key"] = "fake-credential"
            if stream is not None:
                config["use_stream"] = stream
            router = ProviderRouter(config)
            for provider in router.providers:
                await provider.generate(REQUEST)
                _, payload, headers = self.requests[-1]
                self.assertEqual(payload["stream"], stream is True)
                self.assertNotIn("event-stream", headers.get("Accept", ""))
        official = ProviderRouter(
            {
                "primary_base_url": "https://image.novelai.net/v1",
                "primary_api_key": "fake-credential",
                "use_stream": True,
            }
        )
        self.assertIsInstance(official.providers[0], OfficialNovelAIProvider)

    async def test_nonstream_formats_with_both_switch_values(self):
        for stream in (False, True):
            self.provider.use_stream = stream
            formats = [
                {"data": [{"b64_json": B64}]},
                {"data": [{"url": self.image_url}]},
                chat(f"图片：![cat]({self.image_url})"),
                chat(URI),
                chat(f"![cat]({URI})"),
                chat(
                    [
                        {"type": "text", "text": "图片："},
                        {"type": "image_url", "image_url": {"url": self.image_url}},
                    ]
                ),
                chat(
                    [
                        {"type": "text", "text": "图片："},
                        {"type": "text", "text": f"![]({self.image_url})"},
                    ]
                ),
                chat([{"type": "image_url", "image_url": {"url": URI}}]),
                chat(["图片：", {"type": "text", "text": URI}]),
                chat(f'![cat](<{self.image_url}> "生成图")'),
            ]
            for response in formats:
                with self.subTest(stream=stream, format=formats.index(response)):
                    self.set_json(response)
                    await self.assert_image()
                    self.assertEqual(self.requests[-1][1]["stream"], stream)
        self.assertTrue(self.downloads)

    async def test_img2img_obeys_switch_and_preserves_reference(self):
        for stream in (False, True):
            self.provider.use_stream = stream
            await self.provider.generate(
                GenerationRequest(
                    "nai-diffusion-5-full",
                    "cat",
                    "",
                    action="img2img",
                    image=B64,
                    strength=0.7,
                    noise=0.2,
                )
            )
            payload = self.requests[-1][1]
            self.assertEqual(payload["stream"], stream)
            self.assertEqual(payload["action"], "img2img")
            self.assertEqual(payload["image"], B64)
            self.assertEqual((payload["strength"], payload["noise"]), (0.7, 0.2))

    async def test_sse_fallback_images_and_split_chat_content(self):
        markdown = f"![]({self.image_url})"
        formats = [
            sse({"data": [{"b64_json": B64}]}),
            sse({"data": [{"url": self.image_url}]}),
            sse(chat(URI)),
            sse(
                *[
                    {"choices": [{"index": 0, "delta": {"content": text}}]}
                    for text in (markdown[:8], markdown[8:])
                ]
            ),
            sse(
                *[
                    {"choices": [{"index": 0, "delta": {"content": text}}]}
                    for text in (URI[:25], URI[25:80], URI[80:])
                ]
            ),
            sse({"image": B64, "final": False}, {"image": B64, "final": True}),
        ]
        for stream in (False, True):
            self.provider.use_stream = stream
            for content_type in ("text/event-stream", "application/json"):
                self.content_type = content_type
                for response in formats:
                    with self.subTest(stream=stream, content_type=content_type):
                        self.body = response.encode()
                        await self.assert_image()

    async def test_standard_image_generation_completed_event_uses_top_level_b64_json(self):
        self.content_type = "text/event-stream"
        self.body = sse({"type": "image_generation.completed", "b64_json": B64}).encode()
        await self.assert_image()

    async def test_markdown_image_url_with_parentheses_is_downloaded(self):
        image_url = self.base_url + "/image(with-parentheses).png"
        self.set_json(chat(f"![cat]({image_url})"))
        with self.public_ip:
            images = await self.provider.generate(REQUEST)
        self.assertEqual([item.data for item in images], [PNG])

    async def test_multiple_images_and_content_parts(self):
        self.set_json(
            chat(
                [
                    {"type": "text", "text": f"![]({self.image_url})"},
                    {"type": "image_url", "image_url": {"url": URI}},
                ]
            )
        )
        with self.public_ip:
            images = await self.provider.generate(REQUEST)
        self.assertEqual(len(images), 2)

    async def test_http_and_errors_are_distinct_and_do_not_echo_body(self):
        for stream in (False, True):
            self.provider.use_stream = stream
            self.status = 429
            self.set_json({"error": {"message": "do-not-log-response-body-fake-credential"}})
            with self.assertRaisesRegex(GenerationError, "HTTP 429") as context:
                await self.provider.generate(REQUEST)
            self.assertEqual(context.exception.status, 429)
            self.assertNotIn("fake-credential", str(context.exception))
            self.assertNotIn("do-not-log", str(context.exception))
            self.status = 200
            cases = [
                ({"data": []}, "没有图片"),
                (chat("已生成，但没有图片链接"), "没有图片"),
                ({"unrecognized": "fake-credential"}, "格式无法识别"),
                ([], "格式无法识别"),
                ({"data": {}}, "格式无法识别"),
                ({"data": ["bad"]}, "格式无法识别"),
                ({"data": [{"b64_json": "bad-base64"}]}, "Base64 格式无法识别"),
                ({"data": [{"b64_json": 123}]}, "格式无法识别"),
                ({"data": [{"b64_json": base64.b64encode(b"not-image").decode()}]}, "有效图片"),
                (chat({"invalid": True}), "content 应为文本或数组"),
                (chat([123]), "content 数组条目无效"),
                (
                    chat([{"type": "image_url", "image_url": {"url": "data:image/png;base64,?"}}]),
                    "data URI",
                ),
                ({"choices": {}}, "choices 应为数组"),
                ({"choices": [123]}, "choices 数组条目无效"),
                ({"choices": [{"message": "bad"}]}, "message 应为对象"),
            ]
            for response, message in cases:
                self.set_json(response)
                with self.assertRaisesRegex(GenerationError, message) as context:
                    await self.provider.generate(REQUEST)
                self.assertNotIn("fake-credential", str(context.exception))
            for body, message in [
                ("<html>fake-credential</html>", "格式无法识别"),
                ("data: not-json\n\n", "格式无法识别"),
                (sse({"error": {"message": "fake-credential"}}), "上游错误"),
            ]:
                self.body = body.encode()
                with self.assertRaisesRegex(GenerationError, message) as context:
                    await self.provider.generate(REQUEST)
                self.assertNotIn("fake-credential", str(context.exception))

    async def test_download_failure_keeps_http_separate_from_generation_http(self):
        self.set_json({"data": [{"url": self.image_url}]})
        self.download_status = 403
        with (
            self.public_ip,
            self.assertRaisesRegex(GenerationError, "URL 下载失败.*HTTP 403") as context,
        ):
            await self.provider.generate(REQUEST)
        # Download rejection must never trigger a new paid generation via another route.
        self.assertIsNone(context.exception.status)
        self.assertEqual(len(self.requests), 1)

    async def test_download_network_errors_and_invalid_address(self):
        self.set_json({"data": [{"url": self.image_url}]})
        for error, expected in (
            (asyncio.TimeoutError(), "下载超时"),
            (ClientConnectionError(), "网络连接错误"),
        ):
            with patch("aiohttp.ClientSession.get", side_effect=error):
                with self.public_ip, self.assertRaisesRegex(GenerationError, expected):
                    await self.provider.generate(REQUEST)
        with self.assertRaisesRegex(GenerationError, "地址无效"):
            await download_image("file:///fake-credential", ClientTimeout(total=1))
        with self.assertRaisesRegex(GenerationError, "内网"):
            await download_image(self.image_url, ClientTimeout(total=1))

    async def test_generation_timeout_network_and_size_errors(self):
        for error, message in (
            (asyncio.TimeoutError(), "图片请求超时"),
            (ClientConnectionError(), "图片网络请求失败"),
        ):
            with patch("aiohttp.ClientSession.post", side_effect=error):
                with self.assertRaisesRegex(GenerationError, message):
                    await self.provider.generate(REQUEST)
        with patch(
            "astrbot_plugin_nai_series.providers.read_limited",
            AsyncMock(side_effect=GenerationError("图片接口响应超出大小限制")),
        ):
            with self.assertRaisesRegex(GenerationError, "超出大小限制"):
                await self.provider.generate(REQUEST)

    async def test_router_does_not_retry_parse_or_download_errors(self):
        for error in (GenerationError("返回里没有图片"), GenerationError("URL 下载失败: HTTP 403")):
            first, second = AsyncMock(), AsyncMock()
            first.generate.side_effect = error
            router = ProviderRouter({})
            router.providers = [first, second]
            with self.assertRaises(GenerationError):
                await router.generate(REQUEST)
            second.generate.assert_not_awaited()

    def test_multiline_sse_and_malformed_encoding(self):
        body = b'event: message\ndata: {"data":\ndata: []}\n\ndata: [DONE]\n\n'
        self.assertEqual(openai_response_objects(body), [{"data": []}])
        for body in (b"\xff", b"data: []\n\n"):
            with self.assertRaisesRegex(GenerationError, "格式无法识别"):
                openai_response_objects(body)

    @unittest.skipUnless(HAS_ASTRBOT, "Requires AstrBot")
    async def test_translator_stream_switch_and_content_array(self):
        from astrbot_plugin_nai_series.translator import PromptTranslator

        self.set_json(chat([{"type": "text", "text": "POSITIVE: cat"}, "\nUNDESIRED: bad"]))
        for stream in (None, False, True):
            config = {
                "translator_base_url": self.base_url + "/v1",
                "translator_api_key": "fake-credential",
            }
            if stream is not None:
                config["use_stream"] = stream
            translator = PromptTranslator(config)
            result = await translator.translate("一只猫")
            self.assertIn("POSITIVE", result)
            self.assertEqual(self.requests[-1][1]["stream"], stream is True)
        self.body = sse(
            {"choices": [{"index": 0, "delta": {"content": "POSITIVE: cat"}}]},
            {"choices": [{"index": 0, "delta": {"content": "\nUNDESIRED: bad"}}]},
        ).encode()
        self.content_type = "text/event-stream"
        translator = PromptTranslator(
            {
                "translator_base_url": self.base_url + "/v1",
                "translator_api_key": "fake-credential",
                "use_stream": False,
            }
        )
        for stream in (False, True):
            translator.use_stream = stream
            self.assertEqual("POSITIVE: cat, UNDESIRED: bad", await translator.translate("一只猫"))

    @unittest.skipUnless(HAS_ASTRBOT, "Requires AstrBot")
    async def test_translator_errors_are_safe_and_distinct(self):
        from astrbot_plugin_nai_series.translator import PromptTranslator

        translator = PromptTranslator(
            {
                "translator_base_url": self.base_url,
                "translator_api_key": "fake-credential",
            }
        )
        self.status = 503
        self.set_json({"error": {"message": "do-not-log-fake-credential"}})
        with patch("astrbot_plugin_nai_series.translator.logger.warning") as log:
            with self.assertRaisesRegex(ValueError, "翻译接口 HTTP 503") as context:
                await translator.translate("猫")
            self.assertNotIn("fake-credential", str(log.call_args))
            self.assertNotIn("do-not-log", str(log.call_args))
        self.assertNotIn("fake-credential", str(context.exception))
        self.status = 200
        cases = [
            ({"choices": []}, "没有文本"),
            ({"choices": [{}]}, "没有文本"),
            ({"choices": [123]}, "格式无法识别"),
            ({"choices": [{"message": "bad"}]}, "格式无法识别"),
            (
                {"choices": [{"message": {"content": "cat"}, "finish_reason": "length"}]},
                "未完整返回",
            ),
            ({"error": {"message": "fake-credential"}}, "上游错误"),
        ]
        for response, message in cases:
            self.set_json(response)
            with self.assertRaisesRegex(ValueError, message) as context:
                await translator.translate("猫")
            self.assertNotIn("fake-credential", str(context.exception))
            self.assertIn("未发起付费生图", str(context.exception))
        for error, message in (
            (asyncio.TimeoutError(), "翻译请求超时"),
            (ClientConnectionError(), "翻译网络请求失败"),
        ):
            with patch("aiohttp.ClientSession.post", side_effect=error):
                with self.assertRaisesRegex(ValueError, message):
                    await translator.translate("猫")
        self.body = b"<html>do-not-log-fake-credential</html>"
        with self.assertRaisesRegex(ValueError, "格式无法识别") as context:
            await translator.translate("猫")
        self.assertNotIn("fake-credential", str(context.exception))


@unittest.skipUnless(HAS_ASTRBOT, "Requires AstrBot")
class ErrorDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_reason_reaches_notifications_and_task_status(self):
        from test_behavior import Event, PluginTests

        await PluginTests.asyncSetUp(self)
        try:
            self.plugin.router.generate.side_effect = GenerationError("图片接口返回里没有图片")
            event = Event()
            self.plugin._submit(event, "nai-diffusion-5-full", "cat")
            await PluginTests.finish(self)
            job = self.plugin.store.jobs()[0]
            self.assertEqual(job["generation_error"], "图片接口返回里没有图片")
            text = " ".join(
                item.text
                for call in self.plugin.context.send_message.call_args_list
                for item in call.args[1].chain
                if hasattr(item, "text")
            )
            self.assertIn("返回里没有图片", text)
            status = [item async for item in self.plugin.cmd_tasks(event)]
            self.assertIn("返回里没有图片", str(status))
            self.plugin.translator.translate = AsyncMock(
                side_effect=ValueError("翻译接口 HTTP 503")
            )
            self.plugin._submit(event, "nai-diffusion-5-full", "猫")
            await PluginTests.finish(self)
            self.assertIn("翻译接口 HTTP 503", self.plugin.store.jobs()[0]["generation_error"])
            self.assertEqual(self.plugin.router.generate.await_count, 1)
        finally:
            await PluginTests.asyncTearDown(self)
