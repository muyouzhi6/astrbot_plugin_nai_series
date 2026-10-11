"""Resolve only images attached to this event or its explicit replies."""

import asyncio
import base64
import io
import re
from pathlib import Path

import aiohttp
from PIL import Image as PILImage
from PIL import ImageOps

from .providers import GenerationError, _validate_image, download_image

LIMIT = 24 * 1024 * 1024


def _data(value):
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return vars(value) if hasattr(value, "__dict__") else {}


def _scan(value, depth=0):
    """Keep each image's URL/file alternatives together; never inspect text URLs."""
    images, replies, forwards = [], [], []
    if depth > 6:
        raise ValueError("参考消息嵌套过深")
    if isinstance(value, str):
        for kind, fields in re.findall(r"\[CQ:(image|reply|forward),([^\]]+)\]", value):
            values = {}
            for field in fields.split(","):
                key, sep, item = field.partition("=")
                if sep:
                    values[key] = (
                        item.replace("&#44;", ",")
                        .replace("&#91;", "[")
                        .replace("&#93;", "]")
                        .replace("&amp;", "&")
                    )
            nested = _scan({"type": kind, "data": values}, depth + 1)
            images.extend(nested[0])
            replies.extend(nested[1])
            forwards.extend(nested[2])
    elif isinstance(value, (list, tuple)):
        if len(value) > 100:
            raise ValueError("参考消息过长")
        for item in value:
            nested = _scan(item, depth + 1)
            images.extend(nested[0])
            replies.extend(nested[1])
            forwards.extend(nested[2])
    else:
        item = _data(value)
        kind = str(item.get("type", "")).lower().split(".")[-1]
        body = item.get("data") if isinstance(item.get("data"), dict) else item
        if kind == "image":
            refs = tuple(
                dict.fromkeys(str(body[k]) for k in ("url", "file", "path") if body.get(k))
            )
            if refs:
                images.append(refs)
        elif kind == "reply":
            replies.append((body.get("id"), body.get("chain") or item.get("chain")))
        elif kind == "forward":
            if body.get("id"):
                forwards.append(str(body["id"]))
        else:
            for key in ("message", "messages", "content", "chain"):
                if key in body:
                    nested = _scan(body[key], depth + 1)
                    images.extend(nested[0])
                    replies.extend(nested[1])
                    forwards.extend(nested[2])
                    break
    if len(images) > 16 or len(replies) > 4 or len(forwards) > 4:
        raise ValueError("参考消息中的图片或引用过多")
    return images, replies, forwards


async def _action(event, action, **params):
    bot = getattr(event, "bot", None)
    for client in (bot, getattr(bot, "api", None)):
        call = getattr(client, "call_action", None)
        if not callable(call):
            continue
        try:
            result = await asyncio.wait_for(call(action, **params), 8)
            result = _data(result)
            return result.get("data", result)
        except (Exception, asyncio.TimeoutError):
            continue
    return {}


async def _expand(event, scanned, strict=False):
    images, replies, forwards = scanned
    result = list(images)
    for message_id, embedded in replies:
        nested = _scan(embedded)[0] if embedded else []
        if not nested and message_id is not None:
            payload = await _action(event, "get_msg", message_id=message_id)
            if strict and not payload:
                raise ValueError("引用消息读取失败, 请重新引用或直接附图")
            nested, _, nested_forwards = _scan(payload)
            forwards.extend(nested_forwards)
        result.extend(nested)
    if len(forwards) > 4:
        raise ValueError("参考消息中的转发过多")
    for message_id in dict.fromkeys(forwards):
        payload = await _action(event, "get_forward_msg", message_id=message_id)
        if strict and not payload:
            raise ValueError("合并转发读取失败, 请直接附图")
        result.extend(_scan(payload)[0])
    return result


async def collect_images(event, *, strict=False, fetch_current=True):
    message = getattr(event, "message_obj", None)
    getter = getattr(event, "get_messages", None)
    chain = getter() if callable(getter) else getattr(message, "message", [])
    current = _scan(chain)
    raw = _scan(getattr(message, "raw_message", None))
    # Prefer original platform references when another plugin replaced temp paths.
    embedded = {str(key): chain for key, chain in current[1] if chain}
    replies = [(key, chain or embedded.get(str(key))) for key, chain in (raw[1] or current[1])]
    combined = (raw[0] or current[0], replies, raw[2] or current[2])
    result = await _expand(event, combined, strict=strict)
    if fetch_current and not result and getattr(message, "message_id", None):
        fetched = await _action(event, "get_msg", message_id=message.message_id)
        result = await _expand(event, _scan(fetched))
    unique, seen = [], set()
    for refs in result:
        if seen.intersection(refs):
            continue
        seen.update(refs)
        unique.append(refs)
    if len(unique) > 16:
        raise ValueError("每次最多选择 16 张候选参考图")
    return unique


def _read_file(path):
    # Paths are obtained from platform image components, never tool arguments.
    path = Path(path)
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError("参考图文件已失效或超过 24 MB")
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    if len(data) > LIMIT:
        raise ValueError("参考图超过 24 MB")
    return data


async def _read_reference(event, ref, resolve=True):
    if len(ref) > LIMIT * 4 // 3 + 200:
        raise ValueError("参考图超过 24 MB")
    if ref.startswith(("http://", "https://")):
        return await download_image(ref, aiohttp.ClientTimeout(total=30))
    if ref.startswith("base64://"):
        return base64.b64decode(ref[9:], validate=True)
    if ref.startswith("data:image/") and ";base64," in ref:
        return base64.b64decode(ref.split(";base64,", 1)[1], validate=True)
    path = ref.removeprefix("file://")
    if Path(path).is_absolute() and Path(path).is_file():
        return await asyncio.to_thread(_read_file, path)
    if resolve:
        payload = await _action(event, "get_image", file=ref)
        for key in ("url", "file", "path"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if value and str(value) != ref:
                try:
                    return await _read_reference(event, str(value), False)
                except (ValueError, OSError, GenerationError, aiohttp.ClientError):
                    continue
    raise ValueError("参考图已失效, 请重新发图或引用原图消息")


def normalize_image(data, size=None):
    _validate_image(data)
    with PILImage.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if size and image.size != size:
            image = image.resize(size, PILImage.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG")
        value = output.getvalue()
        if len(value) > LIMIT:
            raise ValueError("参考图转换后超过 24 MB")
        return value


async def resolve_image(event, index=None, *, candidates=None):
    async with asyncio.timeout(60):
        if candidates is None:
            candidates = await collect_images(event)
        for attempt in range(2):
            if not candidates:
                raise ValueError("请在同一条消息附图, 或引用一条图片消息")
            if index is None and len(candidates) != 1:
                raise ValueError(
                    f"找到 {len(candidates)} 张图, 请用 --image 序号选择一张; 同消息图片在前, 引用图片在后"
                )
            selected = 1 if index is None else index
            if (
                isinstance(selected, bool)
                or not str(selected).isdigit()
                or not 1 <= int(selected) <= len(candidates)
            ):
                raise ValueError("参考图序号超出范围, 从 1 开始")
            for ref in candidates[int(selected) - 1]:
                try:
                    data = await _read_reference(event, ref)
                    return await asyncio.to_thread(normalize_image, data)
                except (ValueError, OSError, GenerationError, aiohttp.ClientError, TimeoutError):
                    continue
            message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
            if attempt or not message_id:
                break
            fresh = await _action(event, "get_msg", message_id=message_id)
            recovered = await _expand(event, _scan(fresh))
            if not recovered or recovered == candidates:
                break
            candidates = recovered
        raise ValueError("参考图读取失败, 请重新发图或引用原图消息")
