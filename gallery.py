"""Render saved preview images, never generate paid samples on gallery reads."""

import io
import os
import textwrap
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def open_image(data):
    if len(data) > 24 * 1024 * 1024:
        raise ValueError("图片不得超过 24 MB")
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > 24000000:
        raise ValueError("图片分辨率过大")
    image.load()
    return ImageOps.exif_transpose(image).convert("RGB")


def save_preview(data, path):
    image = open_image(data)
    image.thumbnail((720, 960))
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def font(size, custom=""):
    paths = [
        custom,
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for name in paths:
        if name and Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def render_gallery(presets, store, page, pages, custom_font=""):
    columns, card_w, card_h, gap = 2, 410, 470, 20
    rows = (len(presets) + 1) // columns
    image = Image.new("RGB", (900, 142 + rows * (card_h + gap)), "#f5f1e8")
    draw = ImageDraw.Draw(image)
    draw.text((30, 24), "Nai / 预设图册", font=font(32, custom_font), fill="#263c3a")
    draw.text(
        (30, 72),
        f"第 {page}/{pages} 页  |  /预设 nai4 或 nai5 预设名",
        font=font(20, custom_font),
        fill="#50605b",
    )
    for index, preset in enumerate(presets):
        x, y = 30 + (index % 2) * (card_w + gap), 120 + (index // 2) * (card_h + gap)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=12, fill="#fffdf8")
        path = store.preview_path(preset)
        if path.exists():
            with Image.open(path) as source:
                thumbnail = ImageOps.contain(source.convert("RGB"), (card_w - 24, 325))
                image.paste(
                    thumbnail,
                    (x + (card_w - thumbnail.width) // 2, y + 12 + (325 - thumbnail.height) // 2),
                )
        else:
            draw.text(
                (x + 24, y + 150), "尚未设置展示图", font=font(23, custom_font), fill="#7f8981"
            )
        label = f"{preset.family} / {preset.name}"
        title_font = font(23, custom_font)
        while title_font.getlength(label) > card_w - 36:
            label = label[:-2] + "…"
        draw.text((x + 16, y + 345), label, font=title_font, fill="#263c3a")
        for row, line in enumerate(
            textwrap.wrap(preset.description or "附图发送 nai预设展示 预设名", width=19)[:2]
        ):
            draw.text(
                (x + 16, y + 383 + 26 * row), line, font=font(18, custom_font), fill="#68766e"
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
