"""Automatic routing for OpenAI Images-compatible and official NovelAI APIs."""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import secrets
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
from PIL import Image as PILImage

from .models import GenerationRequest


class GenerationError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.message = message
        self.status = status


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
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=PublicResolver()), timeout=timeout
    ) as session:
        async with session.get(url, allow_redirects=False) as response:
            if response.status != 200:
                raise GenerationError("图片已经生成但下载失败, 没有自动重新生成")
            return await read_limited(response, 24 * 1024 * 1024)


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


class OpenAIImagesProvider:
    def __init__(self, name: str, base_url: str, api_key: str, timeout: int = 300):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    body = await read_limited(response)
                    if response.status != 200:
                        raise GenerationError(
                            f"图片接口 HTTP {response.status}, 请检查模型权限/配额和连接配置",
                            response.status,
                        )
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise GenerationError(f"{self.name} 返回了无效 JSON") from exc
                    images: list[GeneratedImage] = []
                    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                        raise GenerationError("图片接口响应格式无效")
                    for item in data.get("data", []):
                        encoded = item.get("b64_json") if isinstance(item, dict) else None
                        if encoded:
                            raw = base64.b64decode(encoded, validate=True)
                            await asyncio.to_thread(_validate_image, raw)
                            images.append(GeneratedImage(raw))
                            continue
                        image_url = item.get("url") if isinstance(item, dict) else None
                        if image_url:
                            raw = await download_image(image_url, timeout)
                            await asyncio.to_thread(_validate_image, raw)
                            images.append(GeneratedImage(raw))
                    if not images:
                        raise GenerationError(f"{self.name} 没有返回图片")
                    return images
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
            "action": "generate",
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
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    body = await read_limited(response)
                    if response.status not in (200, 201):
                        raise GenerationError(
                            f"图片接口 HTTP {response.status}, 请检查模型权限/配额和连接配置",
                            response.status,
                        )
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
            return [self._decode_image(item) for item in data["images"] if isinstance(item, str)]
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
            if data.get("error"):
                raise GenerationError("图片流返回上游错误")
            if isinstance(encoded, str) and encoded and data.get("final") is True:
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
                    self.providers.append(OpenAIImagesProvider(name, url, api_key, timeout))

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
