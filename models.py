"""Domain models and normalization helpers for the Nai series plugin."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MODEL_ALIASES = {
    "nai4": "nai-diffusion-4-5-full",
    "nai45": "nai-diffusion-4-5-full",
    "4": "nai-diffusion-4-5-full",
    "4.5": "nai-diffusion-4-5-full",
    "nai5": "nai-diffusion-5-full",
    "5": "nai-diffusion-5-full",
    "curated": "nai-diffusion-4-5-curated",
    "nai4-curated": "nai-diffusion-4-5-curated",
    "nai5-curated": "nai-diffusion-5-curated",
}


def normalize_model(value: str | None, default: str = "nai-diffusion-4-5-full") -> str:
    """Resolve friendly aliases while preserving explicit provider model IDs."""
    raw = (value or default).strip().lower()
    return MODEL_ALIASES.get(raw, raw or default)


def model_family(model: str) -> str:
    normalized = normalize_model(model)
    if normalized.startswith("nai-diffusion-5-"):
        return "nai5"
    if normalized.startswith("nai-diffusion-4-"):
        return "nai4"
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "custom"


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    model: str
    artist_prompt: str
    negative_prompt: str | None = None
    preview_prompt: str = "solo, 1girl, portrait, looking at viewer"
    preview_size: str = "832x1216"
    description: str = ""

    @property
    def family(self) -> str:
        return model_family(self.model)


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    prompt: str
    negative_prompt: str
    width: int = 832
    height: int = 1216
    steps: int = 28
    scale: float = 5.0
    sampler: str = "k_euler_ancestral"
    noise_schedule: str = "karras"
    cfg_rescale: float = 0.0
    quality: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,120}", self.model):
            raise ValueError("模型名格式无效")
        if not self.prompt.strip() or len(self.prompt) > 16000:
            raise ValueError("提示词应为 1-16000 字符")
        if any(v < 64 or v > 2048 or v % 64 for v in (self.width, self.height)):
            raise ValueError("宽高必须是 64-2048 范围内的 64 倍数")
        if self.width * self.height > 2097152:
            raise ValueError("图片面积不得超过 2097152 像素")
        if (
            not 1 <= self.steps <= 50
            or not 1 <= self.scale <= 20
            or not 0 <= self.cfg_rescale <= 20
        ):
            raise ValueError("steps/CFG/rescale 参数超出范围")

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class SessionState:
    model: str = "nai-diffusion-4-5-full"
    presets: dict[str, str] = field(default_factory=dict)
    last_request: dict[str, Any] | None = None


def _as_dict(item: Any) -> dict[str, Any] | None:
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        raw = item.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            if ":" in raw:
                name, prompt = raw.split(":", 1)
                return {"name": name.strip(), "artist_prompt": prompt.strip()}
    return None


def parse_presets(value: Any) -> list[Preset]:
    """Accept JSON text, list objects, or legacy name:prompt entries."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except json.JSONDecodeError as exc:
            raise ValueError("预设 JSON 无效") from exc
    if not isinstance(value, list):
        raise ValueError("预设必须是列表")
    result: list[Preset] = []
    for index, raw_item in enumerate(value):
        item = _as_dict(raw_item)
        if not item:
            continue
        name = str(item.get("name") or item.get("id") or f"preset-{index + 1}").strip()
        preset_id = str(item.get("id") or name).strip()
        artist = str(
            item.get("artist_prompt") or item.get("prompt") or item.get("artist") or ""
        ).strip()
        if not name:
            raise ValueError("预设名称不能为空")
        result.append(
            Preset(
                id=preset_id,
                name=name,
                model=normalize_model(str(item.get("model") or item.get("model_name") or "nai4")),
                artist_prompt=artist,
                negative_prompt=str(item["negative_prompt"]).strip()
                if item.get("negative_prompt") is not None
                else None,
                preview_prompt=str(
                    item.get("preview_prompt") or "solo, 1girl, portrait, looking at viewer"
                ).strip(),
                preview_size=str(item.get("preview_size") or "832x1216").strip(),
                description=str(item.get("description") or "").strip(),
            )
        )
    identities = [(p.family, p.id.casefold()) for p in result]
    names = [(p.family, p.name.casefold()) for p in result]
    if len(set(identities)) != len(identities) or len(set(names)) != len(names):
        raise ValueError("同一模型族内预设名称和 ID 不得重复")
    return result


def find_preset(presets: list[Preset], query: str, model: str | None = None) -> Preset | None:
    needle = query.strip().casefold()
    candidates = [p for p in presets if model is None or p.family == model_family(model)]
    matches = [p for p in candidates if p.id.casefold() == needle or p.name.casefold() == needle]
    if len(matches) > 1:
        raise ValueError("预设重名, 请指定 nai4 或 nai5")
    return matches[0] if matches else None
