"""Faithful Chinese-to-NAI prompt translation with user-overridable instructions."""

from __future__ import annotations

import asyncio
import json
import re

import aiohttp
from astrbot.api import logger

DEFAULT_TRANSLATOR_PROMPT = """You translate a user's Chinese image description into a NovelAI prompt.
Preserve the user's meaning exactly. Keep every subject, count, action, relationship, position, clothing item, object, measurement, size comparison, negation, and limitation.
Do not invent hair color, hairstyle, eye color, body type, facial expression, gaze, camera, location, weather, lighting, season, year, props, or clothing that the user did not mention.
Do not turn a broad object into a specific one: do not turn ice cream into an ice-cream cone, or a hairstyle into a ponytail. Keep numbers and units literally, for example 1-meter-tall ice cream.
Translate rather than invent content. Do not reclassify ages, consent, or identities. Do not add claims that override the provider's safety rules.
Use concise English tags or short English phrases. Do not add quality tags, artist tags, years, negative prompts, or weights unless the user explicitly supplied them.
Return one comma-separated line only, with no explanation or markdown."""


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


class PromptTranslator:
    def __init__(self, config: dict):
        self.enabled = bool(config.get("translator_enabled", True))
        self.base_url = str(config.get("translator_base_url", "")).rstrip("/")
        self.api_key = str(config.get("translator_api_key", ""))
        self.model = str(config.get("translator_model", "gpt-4o-mini"))
        self.system_prompt = (
            str(config.get("translator_system_prompt", "")).strip() or DEFAULT_TRANSLATOR_PROMPT
        )
        prefix = str(config.get("translator_custom_prefix", "")).strip()
        self.system_prompt = f"{prefix}\n\n{self.system_prompt}" if prefix else self.system_prompt
        self.timeout = int(config.get("translator_timeout", 45))

    async def translate(self, text: str) -> str:
        if not self.enabled or not has_chinese(text):
            return text
        if not self.base_url or not self.api_key:
            raise ValueError("翻译接口未配置, 请配置翻译模型或使用 --raw")
        url = (
            f"{self.base_url}/chat/completions"
            if self.base_url.endswith("/v1")
            else f"{self.base_url}/v1/chat/completions"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": 1500,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise RuntimeError(f"translator HTTP {response.status}")
                    data = json.loads(body)
                    if not isinstance(data, dict) or not data.get("choices"):
                        raise RuntimeError("translator returned no choices")
                    choice = data["choices"][0]
                    if choice.get("finish_reason") not in (None, "stop"):
                        raise RuntimeError("translator did not finish normally")
                    message = choice.get("message", {})
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise RuntimeError(
                            f"translator returned no text: {data.get('choices', [{}])[0].get('finish_reason')}"
                        )
                    return content.strip().replace("\n", ", ")
        except (
            asyncio.TimeoutError,
            aiohttp.ClientError,
            json.JSONDecodeError,
            RuntimeError,
            TypeError,
            AttributeError,
            IndexError,
        ) as exc:
            logger.warning("[NaiSeries] 翻译失败 (%s)", type(exc).__name__)
            raise ValueError(
                "翻译未返回有效文本, 未发起付费生图. 请检查翻译模型或使用 --raw"
            ) from exc
