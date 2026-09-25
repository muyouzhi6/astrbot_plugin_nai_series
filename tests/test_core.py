import base64
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

from aiohttp import web
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_nai_series.models import (
    GenerationRequest,
    find_preset,
    model_family,
    normalize_model,
    parse_presets,
)
from astrbot_plugin_nai_series.providers import OfficialNovelAIProvider, OpenAIImagesProvider

_buffer = io.BytesIO()
Image.new("RGB", (64, 64), "orange").save(_buffer, format="PNG")
PNG = _buffer.getvalue()


class CoreTests(unittest.IsolatedAsyncioTestCase):
    def test_official_json_and_final_sse(self):
        provider = OfficialNovelAIProvider("native", "https://image.novelai.net", "test")
        encoded = base64.b64encode(PNG).decode()
        self.assertEqual(
            provider._parse_json_image(json.dumps({"image": encoded}).encode())[0].data, PNG
        )
        stream = (
            "data: "
            + json.dumps({"image": encoded, "final": False})
            + "\n\ndata: "
            + json.dumps({"image": encoded, "final": True})
            + "\n\ndata: [DONE]\n"
        )
        self.assertEqual(len(provider._parse_sse_images(stream.encode())), 1)

    async def asyncSetUp(self):
        self.requests = []

        async def images(request):
            self.requests.append((request.path, await request.json()))
            return web.json_response({"data": [{"b64_json": base64.b64encode(PNG).decode()}]})

        async def official(request):
            self.requests.append((request.path, await request.json()))
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("image_0.png", PNG)
            return web.Response(body=buffer.getvalue(), status=201, content_type="application/zip")

        app = web.Application()
        app.router.add_post("/v1/images/generations", images)
        app.router.add_post("/ai/generate-image", official)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def test_aliases_and_model_specific_presets(self):
        self.assertEqual(model_family("nai4"), "nai4")
        self.assertEqual(model_family("nai-diffusion-4-5-curated"), "nai4")
        self.assertEqual(model_family("nai-diffusion-5-full"), "nai5")
        self.assertEqual(normalize_model("nai5"), "nai-diffusion-5-full")
        self.assertEqual(normalize_model("nai4"), "nai-diffusion-4-5-full")
        presets = parse_presets(
            json.dumps(
                [
                    {
                        "name": "4风格",
                        "model": "nai4",
                        "artist_prompt": "style4",
                        "negative_prompt": "bad4",
                    },
                    {
                        "name": "5风格",
                        "model": "nai5",
                        "artist_prompt": "style5",
                        "negative_prompt": "bad5",
                    },
                ],
                ensure_ascii=False,
            )
        )
        self.assertEqual(find_preset(presets, "4风格", "nai4").negative_prompt, "bad4")
        self.assertIsNone(find_preset(presets, "5风格", "nai4"))

    async def test_openai_images_payload_does_not_send_seed(self):
        provider = OpenAIImagesProvider("test", f"http://127.0.0.1:{self.port}/v1", "secret")
        request = GenerationRequest("nai-diffusion-5-full", "style5, apple", "bad5", steps=28)
        images = await provider.generate(request)
        self.assertEqual(images[0].data, PNG)
        path, payload = self.requests[0]
        self.assertEqual(path, "/v1/images/generations")
        self.assertEqual(payload["model"], "nai-diffusion-5-full")
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["negative_prompt"], "bad5")

    async def test_official_zip_payload(self):
        provider = OfficialNovelAIProvider("official", f"http://127.0.0.1:{self.port}", "secret")
        images = await provider.generate(
            GenerationRequest("nai-diffusion-4-5-full", "apple", "bad")
        )
        self.assertEqual(images[0].data, PNG)
        self.assertEqual(self.requests[0][0], "/ai/generate-image")
        self.assertEqual(self.requests[0][1]["parameters"]["negative_prompt"], "bad")
        self.assertEqual(self.requests[0][1]["model"], "nai-diffusion-4-5-full")


if __name__ == "__main__":
    unittest.main()
