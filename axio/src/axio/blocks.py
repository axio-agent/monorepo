"""Content blocks: TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

from axio.types import ToolCallID, ToolName


class ContentBlock:
    """Base class for all content blocks."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TextBlock(ContentBlock):
    text: str


@dataclass(frozen=True, slots=True)
class ImageBlock(ContentBlock):
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: bytes


@dataclass(frozen=True, slots=True)
class ToolUseBlock(ContentBlock):
    id: ToolCallID
    name: ToolName
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock(ContentBlock):
    tool_use_id: ToolCallID
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False


@singledispatch
def to_dict(block: ContentBlock) -> dict[str, Any]:
    """Serialize a ContentBlock to a plain dict."""
    msg = f"Unknown block type: {type(block).__name__}"
    raise TypeError(msg)


@to_dict.register(TextBlock)
def _text_to_dict(block: TextBlock) -> dict[str, Any]:
    return {"type": "text", "text": block.text}


@to_dict.register(ImageBlock)
def _image_to_dict(block: ImageBlock) -> dict[str, Any]:
    return {"type": "image", "media_type": block.media_type, "data": base64.b64encode(block.data).decode()}


@to_dict.register(ToolUseBlock)
def _tool_use_to_dict(block: ToolUseBlock) -> dict[str, Any]:
    return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}


@to_dict.register(ToolResultBlock)
def _tool_result_to_dict(block: ToolResultBlock) -> dict[str, Any]:
    if isinstance(block.content, str):
        serialized_content: str | list[dict[str, Any]] = block.content
    else:
        serialized_content = [to_dict(b) for b in block.content]
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": serialized_content,
        "is_error": block.is_error,
    }


def from_dict(data: dict[str, Any]) -> ContentBlock:
    """Deserialize a plain dict to a ContentBlock."""
    match data["type"]:
        case "text":
            return TextBlock(text=data["text"])
        case "image":
            return ImageBlock(media_type=data["media_type"], data=base64.b64decode(data["data"]))
        case "tool_use":
            return ToolUseBlock(id=data["id"], name=data["name"], input=data["input"])
        case "tool_result":
            raw = data["content"]
            if isinstance(raw, str):
                content: str | list[TextBlock | ImageBlock] = raw
            else:
                content = [from_dict(b) for b in raw]  # type: ignore[misc]
            return ToolResultBlock(tool_use_id=data["tool_use_id"], content=content, is_error=data["is_error"])
        case _:
            msg = f"Unknown block type: {data['type']}"
            raise ValueError(msg)
