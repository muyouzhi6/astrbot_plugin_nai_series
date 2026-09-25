"""Preset operations kept separate from model routing and translation."""

from __future__ import annotations

from .models import Preset, find_preset, model_family, normalize_model, parse_presets


def preset_lines(presets: list[Preset]) -> list[str]:
    return [
        f"{preset.name} [{preset.family}] - {preset.description or '无说明'}" for preset in presets
    ]


def compose_prompt(preset: Preset | None, prompt: str) -> str:
    pieces = []
    if preset and preset.artist_prompt:
        pieces.append(preset.artist_prompt)
    if prompt.strip():
        pieces.append(prompt.strip())
    return ", ".join(piece for piece in pieces if piece)


__all__ = [
    "Preset",
    "compose_prompt",
    "model_family",
    "normalize_model",
    "parse_presets",
    "preset_lines",
    "find_preset",
]
