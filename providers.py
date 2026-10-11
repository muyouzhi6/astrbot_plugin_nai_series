"""Automatic routing for OpenAI Images-compatible and official NovelAI APIs."""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import re
import secrets
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
from PIL import Image as PILImage

from .models import GenerationRequest


class GenerationError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, request_id: str = ""):
        super().__init__(message)
        self.message = message
        self.status = status
        self.request_id = request_id


def response_error(response, body=b""):
    # Only persist a validated correlation ID, never an untrusted error body.
    request_id = response.headers.get("X-Request-Id", "") or response.headers.get(
        "X-Oneapi-Request-Id", ""
    )
    upstream = ""
    try:
        error = json.loads(body).get("error", {})
        if isinstance(error, dict):
            request_id = request_id or str(error.get("request_id", ""))
            match = re.fullmatch(
                r"NAI upstream HTTP ([1-5][0-9]{2}); not retried", str(error.get("message", ""))
            )
            if match:
                upstream = f", 上游 HTTP {match[1]}"
    except (ValueError, AttributeError):
        pass
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", request_id):
        request_id = ""
    message = f"图片接口 HTTP {response.status}" + upstream
    if request_id:
        message += f", 请求编号 {request_id}"
    return GenerationError(message + "; 未自动重试", response.status, request_id)


def image_parameters(request):
    if request.action != "img2img":
        return {}
    if not request.image:
        raise GenerationError("参考图未就绪, 未提交生成")
    return {"image": request.image, "strength": request.strength, "noise": request.noise}


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime: str = "image/png"


async def read_limited(response, limit=32 * 1024 * 1024):
    data = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise GenerationError("图片接口响应超出大小限制")
    return bytes(data)


class PublicResolver(aiohttp.resolver.DefaultResolver):
    async def resolve(self, host, port=0, family=0):
        records = await super().resolve(host, port, family)
        if any(not ipaddress.ip_address(record["host"]).is_global for record in records):
            raise GenerationError("图片下载地址不得指向内网")
        return records


async def download_image(url, timeout):
    parsed = urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise GenerationError("图片下载地址无效")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise GenerationError("图片下载地址不得指向内网")
    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(resolver=PublicResolver()), timeout=timeout
        ) as session:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    raise GenerationError(f"图片 URL 下载失败: HTTP {response.status}; 未重新生成")
                return await read_limited(response, 24 * 1024 * 1024)
    except asyncio.TimeoutError as exc:
        raise GenerationError("图片 URL 下载超时; 未重新生成") from exc
    except aiohttp.ClientError as exc:
        raise GenerationError("图片 URL 下载失败: 网络连接错误; 未重新生成") from exc


def _validate_image(data: bytes) -> None:
    if len(data) > 24 * 1024 * 1024:
        raise GenerationError("图片超出大小限制")
    try:
        with PILImage.open(io.BytesIO(data)) as image:
            if image.format not in ("PNG", "JPEG", "WEBP") or image.width * image.height > 24000000:
                raise ValueError("Invalid image dimensions or format")
            image.load()
    except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise GenerationError("服务商返回的内容不是有效图片") from exc


def openai_response_objects(body: bytes, content_type: str = "") -> list[dict]:
    """Decode JSON or SSE independently of the requested transport mode."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GenerationError("接口返回格式无法识别") from exc
    is_sse = "event-stream" in content_type.lower() or text.lstrip().startswith(
        ("data:", "event:", ":")
    )
    if not is_sse:
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise GenerationError("接口返回格式无法识别: 不是有效 JSON 或 SSE") from exc
        if not isinstance(data, dict):
            raise GenerationError("接口返回格式无法识别: JSON 应为对象")
        return [data]

    events = []
    fields = []

    def flush():
        if not fields:
            return
        payload = "\n".join(fields).strip()
        fields.clear()
        if not payload or payload == "[DONE]":
            return
        try:
            data = json.loads(payload)
        except ValueError as exc:
            raise GenerationError("接口 SSE 返回格式无法识别: 事件不是有效 JSON") from exc
        if not isinstance(data, dict):
            raise GenerationError("接口 SSE 返回格式无法识别: 事件应为对象")
        events.append(data)

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif line.startswith("data:"):
            fields.append(line[5:].lstrip(" "))
    flush()
    if not events:
        raise GenerationError("接口 SSE 返回里没有内容")
    return events


def chat_content_parts(content: Any) -> tuple[str, list[str]]:
    """Return text and explicit image URLs from chat multimodal content."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise GenerationError("接口返回格式无法识别: content 应为文本或数组")
    text, urls = [], []
    for part in content:
        if isinstance(part, str):
            text.append(part)
        elif isinstance(part, dict):
            if isinstance(part.get("text"), str):
                text.append(part["text"])
            image_url = part.get("image_url", part.get("url"))
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if isinstance(image_url, str) and image_url:
                urls.append(image_url)
        else:
            raise GenerationError("接口返回格式无法识别: content 数组条目无效")
    return "".join(text), urls


class OpenAIImagesProvider:
    def __init__(
        self, name: str, base_url: str, api_key: str, timeout: int = 300, use_stream: bool = False
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.use_stream = use_stream

    async def _parse_image_value(
        self, value: str, timeout: aiohttp.ClientTimeout
    ) -> GeneratedImage:
        if not isinstance(value, str) or not value.strip():
            raise GenerationError("图片返回格式无法识别: 图片字段应为非空文本")
        value = value.strip()
        if value.startswith("data:"):
            match = re.fullmatch(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=]+)", value)
            if not match:
                raise GenerationError("图片 data URI 格式无法识别")
            value = match[1]
        if value.startswith(("http://", "https://")):
            try:
                raw = await download_image(value, timeout)
            except GenerationError as exc:
                raise GenerationError(f"图片 URL 下载失败: {exc.message}") from exc
            except (ValueError, TypeError) as exc:
                raise GenerationError("图片 URL 下载失败: 地址格式无效") from exc
        else:
            try:
                raw = base64.b64decode(value, validate=True)
            except (ValueError, TypeError) as exc:
                raise GenerationError("图片 Base64 格式无法识别") from exc
        await asyncio.to_thread(_validate_image, raw)
        return GeneratedImage(raw)

    async def _parse_response(
        self, body: bytes, content_type: str, timeout: aiohttp.ClientTimeout
    ) -> list[GeneratedImage]:
        values = []
        texts: dict[int, list[str]] = {}
        recognized = False
        for data in openai_response_objects(body, content_type):
            if data.get("error") or data.get("event_type") == "error":
                raise GenerationError("图片接口返回上游错误; 未自动重新生成")
            if "data" in data:
                recognized = True
                if not isinstance(data["data"], list):
                    raise GenerationError("图片返回格式无法识别: data 应为数组")
                for item in data["data"]:
                    if not isinstance(item, dict):
                        raise GenerationError("图片返回格式无法识别: data 数组条目无效")
                    value = item.get("b64_json") or item.get("url")
                    if value:
                        values.append(value)
            # Some relays expose NovelAI final-image events directly.
            if "image" in data:
                recognized = True
                if data.get("final") is not False and data.get("event_type") != "intermediate":
                    values.append(data["image"])
            if "choices" in data:
                recognized = True
                if not isinstance(data["choices"], list):
                    raise GenerationError("图片返回格式无法识别: choices 应为数组")
                for choice in data["choices"]:
                    if not isinstance(choice, dict):
                        raise GenerationError("图片返回格式无法识别: choices 数组条目无效")
                    message = choice.get("message", choice.get("delta", {}))
                    if not isinstance(message, dict):
                        raise GenerationError("图片返回格式无法识别: message 应为对象")
                    text, urls = chat_content_parts(message.get("content"))
                    index = choice.get("index", 0)
                    if type(index) is not int:
                        raise GenerationError("图片返回格式无法识别: choice 编号无效")
                    texts.setdefault(index, []).append(text)
                    values.extend(urls)
        for parts in texts.values():
            text = "".join(parts)
            # Collect in document order; data URIs inside markdown are decoded once.
            pattern = (
                r"!\[[^\]]*\]\(\s*<?((?:https?://|data:image/)[^\s<>]+?)>?"
                r'(?:\s+"[^"\n]*")?\s*\)'
                r"|(data:image/[\w.+-]+;base64,[A-Za-z0-9+/=]+)"
            )
            for match in re.finditer(pattern, text):
                values.append(match[1] or match[2])
        if not recognized:
            raise GenerationError("图片接口返回格式无法识别")
        if not values:
            raise GenerationError("图片接口返回里没有图片")
        images = []
        for value in values:
            images.append(await self._parse_image_value(value, timeout))
        return images

    async def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        url = f"{base}/v1/images/generations"
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "size": request.size,
            "n": 1,
            "response_format": "b64_json",
            "steps": request.steps,
            "scale": request.scale,
            "sampler": request.sampler,
            "noise_schedule": request.noise_schedule,
            "cfg_rescale": request.cfg_rescale,
        }
        payload["stream"] = self.use_stream
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if request.action == "img2img":
            payload.update(action=request.action, **image_parameters(request))
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    body = await read_limited(response)
                    if response.status != 200:
                        raise response_error(response, body)
                    return await self._parse_response(
                        body, response.headers.get("content-type", "").lower(), timeout
                    )
            except asyncio.TimeoutError as exc:
                raise GenerationError(f"{self.name} 图片请求超时") from exc
            except aiohttp.ClientError as exc:
                raise GenerationError("图片网络请求失败, 未自动重试以避免重复计费") from exc
            except (ValueError, TypeError) as exc:
                raise GenerationError("图片接口返回了无效数据") from exc


class OfficialNovelAIProvider:
    def __init__(self, name: str, base_url: str, api_key: str, timeout: int = 300):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    async def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        url = f"{self.base_url}/ai/generate-image"
        payload = {
            "input": request.prompt,
            "model": request.model,
            "action": request.action,
            "parameters": {
                "width": request.width,
                "height": request.height,
                "scale": request.scale,
                "sampler": request.sampler,
                "steps": request.steps,
                "n_samples": 1,
                "noise_schedule": request.noise_schedule,
                "ucPreset": 3,
                "qualityToggle": request.quality,
                "negative_prompt": request.negative_prompt,
                "seed": secrets.randbelow(2**32),
                "params_version": 3,
                "cfg_rescale": request.cfg_rescale,
                "v4_prompt": {
                    "caption": {"base_caption": request.prompt, "char_captions": []},
                    "use_coords": False,
                    "use_order": True,
                },
                "v4_negative_prompt": {
                    "caption": {"base_caption": request.negative_prompt, "char_captions": []},
                    "legacy_uc": False,
                },
            },
        }
        payload["parameters"].update(image_parameters(request))
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    body = await read_limited(response)
                    if response.status not in (200, 201):
                        raise response_error(response, body)
                    content_type = response.headers.get("content-type", "").lower()
                    if "json" in content_type or body.lstrip().startswith(b"{"):
                        return self._parse_json_image(body)
                    if "event-stream" in content_type or body.startswith(b"data:"):
                        return self._parse_sse_images(body)
                    try:
                        archive = zipfile.ZipFile(io.BytesIO(body))
                    except zipfile.BadZipFile as exc:
                        raise GenerationError(
                            f"{self.name} 返回内容不是 JSON, SSE 或图片 ZIP"
                        ) from exc
                    images: list[GeneratedImage] = []
                    for item in archive.infolist():
                        if item.is_dir() or not item.filename.lower().endswith(
                            (".png", ".jpg", ".jpeg")
                        ):
                            continue
                        if item.file_size > 24 * 1024 * 1024 or len(images) >= 8:
                            raise GenerationError("图片 ZIP 超出大小限制")
                        raw = archive.read(item)
                        _validate_image(raw)
                        mime = (
                            "image/jpeg"
                            if item.filename.lower().endswith((".jpg", ".jpeg"))
                            else "image/png"
                        )
                        images.append(GeneratedImage(raw, mime))
                    if not images:
                        raise GenerationError(f"{self.name} ZIP 中没有图片")
                    return images
            except asyncio.TimeoutError as exc:
                raise GenerationError(f"{self.name} 图片请求超时") from exc
            except aiohttp.ClientError as exc:
                raise GenerationError("图片网络请求失败, 未自动重试以避免重复计费") from exc
            except (ValueError, TypeError, zipfile.BadZipFile) as exc:
                raise GenerationError("图片接口返回了无效数据") from exc

    @staticmethod
    def _decode_image(value: str) -> GeneratedImage:
        raw = base64.b64decode(value, validate=True)
        _validate_image(raw)
        return GeneratedImage(raw)

    def _parse_json_image(self, body: bytes) -> list[GeneratedImage]:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GenerationError(f"{self.name} JSON 响应无法解析") from exc
        encoded = data.get("image") if isinstance(data, dict) else None
        if isinstance(encoded, str) and encoded:
            return [self._decode_image(encoded)]
        if isinstance(data, dict) and isinstance(data.get("images"), list):
            images = []
            for item in data["images"]:
                encoded = item.get("image") if isinstance(item, dict) else item
                if not isinstance(encoded, str) or not encoded:
                    raise GenerationError("图片 JSON 中存在无效条目")
                images.append(self._decode_image(encoded))
            if images:
                return images
        raise GenerationError(f"{self.name} JSON 响应没有图片")

    def _parse_sse_images(self, body: bytes) -> list[GeneratedImage]:
        images: list[GeneratedImage] = []
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload in {b"[DONE]", b""}:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            encoded = data.get("image") if isinstance(data, dict) else None
            if not isinstance(data, dict):
                raise GenerationError("图片流事件格式错误")
            if data.get("error") or data.get("event_type") == "error":
                raise GenerationError("图片流返回上游错误")
            if (
                isinstance(encoded, str)
                and encoded
                and (data.get("final") is True or data.get("event_type") == "final")
            ):
                images = [self._decode_image(encoded)]
        if not images:
            raise GenerationError(f"{self.name} SSE 响应没有图片")
        return images


class ProviderRouter:
    """Try configured endpoints in order; provider choice never enters user commands."""

    def __init__(self, config: dict[str, Any]):
        timeout = int(config.get("request_timeout", 300))
        self.providers = []
        for name, key in (("primary", "primary"), ("secondary", "secondary"), ("third", "third")):
            url = str(config.get(f"{key}_base_url", "")).strip()
            api_key = str(config.get(f"{key}_api_key", "")).strip()
            if url and api_key:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username:
                    raise ValueError("图片连接地址必须是有效 HTTP(S) URL")
                if parsed.hostname == "novelai.net" or parsed.hostname.endswith(".novelai.net"):
                    self.providers.append(
                        OfficialNovelAIProvider(
                            name, f"{parsed.scheme}://{parsed.netloc}", api_key, timeout
                        )
                    )
                else:
                    self.providers.append(
                        OpenAIImagesProvider(
                            name, url, api_key, timeout, bool(config.get("use_stream", False))
                        )
                    )

    async def generate(self, request: GenerationRequest) -> tuple[list[GeneratedImage], str]:
        if not self.providers:
            raise GenerationError("没有配置可用的 NAI 图片连接")
        errors: list[str] = []
        for provider in self.providers:
            try:
                return await provider.generate(request), provider.name
            except GenerationError as exc:
                errors.append(exc.message)
                # Only explicit pre-generation rejection can select another route.
                if exc.status not in (401, 403, 404):
                    raise
        raise GenerationError("所有 NAI 图片连接均失败: " + " | ".join(errors))
