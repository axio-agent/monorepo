"""Content blocks: TextBlock, ImageBlock, AudioBlock, ToolUseBlock, ToolResultBlock."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any, Literal

from .types import ToolCallID, ToolName

logger = logging.getLogger(__name__)

type ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
type AudioMediaType = Literal[
    "audio/x-aac",
    "audio/flac",
    "audio/mp3",
    "audio/m4a",
    "audio/mpeg",
    "audio/mpga",
    "audio/mp4",
    "audio/ogg",
    "audio/pcm",
    "audio/wav",
    "audio/webm",
]
type VideoMediaType = Literal[
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
]


class ContentBlock:
    """Base class for all content blocks."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class TextBlock(ContentBlock):
    text: str
    #: Opaque proof for this text, where the provider signs a text part. Replayed with it.
    #: Kept out of ``repr`` with the rest of them: see ``provider`` below.
    signature: str = field(default="", repr=False)
    #: Which protocol issued ``signature``. See ``ReasoningBlock.provider``.
    provider: str = ""


@dataclass(frozen=True, slots=True)
class ImageBlock(ContentBlock):
    media_type: ImageMediaType
    data: bytes


@dataclass(frozen=True, slots=True)
class AudioBlock(ContentBlock):
    media_type: AudioMediaType
    data: bytes


@dataclass(frozen=True, slots=True)
class VideoBlock(ContentBlock):
    media_type: VideoMediaType
    data: bytes


@dataclass(frozen=True, slots=True)
class ReasoningBlock(ContentBlock):
    """The model's own reasoning, kept so the turn can be sent back unaltered.

    ``signature`` is the provider's proof that the block is its own. Anthropic refuses a returned
    thinking block whose signature is missing or changed, and Google reports
    ``MISSING_THOUGHT_SIGNATURE`` for the same failure. A stored turn that dropped the signature
    cannot be replayed. Never inspect, decode or truncate it.

    ``redacted`` marks a block whose reasoning the provider withheld. The signature still has to
    travel, and ``text`` is empty.
    """

    text: str = ""
    #: Out of ``repr`` on purpose: this is the one field documented as never to be inspected, and a
    #: debug log of the block printed it in full beside everything else.
    signature: str = field(default="", repr=False)
    redacted: bool = False
    #: How the provider names this block, where it names them. Required beside the signature.
    id: str = ""
    #: Which protocol issued ``signature``: ``"anthropic"``, ``"google"``, ``"openai"``, the same
    #: names ``ProviderEvent`` uses. The value means nothing outside the protocol that made it —
    #: Anthropic sends it as a thinking signature, Google as ``thoughtSignature``, Responses as
    #: ``encrypted_content`` — so a session that changes transport must not replay it to the next
    #: one. Empty says nobody recorded it, which is what a turn stored before this field existed
    #: looks like; ``proof()`` lets that through rather than dropping proofs it cannot judge.
    provider: str = ""


@dataclass(frozen=True, slots=True)
class ToolUseBlock(ContentBlock):
    id: ToolCallID
    name: ToolName
    input: dict[str, Any]
    #: Opaque proof for this call. Stored rather than held in the transport, which a restart empties.
    signature: str = field(default="", repr=False)
    #: Which protocol issued ``signature``. See ``ReasoningBlock.provider``.
    provider: str = ""


@dataclass(frozen=True, slots=True)
class ProviderBlock(ContentBlock):
    """One item of the turn that only the protocol which produced it can read.

    An endpoint that runs its own tools answers with items this vocabulary has no shape for: a web
    search it ran, a file it read, code it executed, and whatever it adds next. Where the
    application keeps the history rather than the provider — ``store=False`` on the Responses API —
    the provider expects every one of those items back, in order and exactly once. Dropped, the
    next request is missing what the model was answering from, and reasoning continuity, tool-call
    association and anything built on a newer item type break with nothing said.

    ``data`` is the item exactly as it arrived and is never interpreted here. ``provider`` names the
    protocol that issued it, so it is replayed only to an endpoint that speaks the same one, and
    ``kind`` is the item's own type, which is all a caller needs to render or count it.
    """

    provider: str
    kind: str
    data: dict[str, Any]
    #: How the provider names this item, where it names them.
    id: str = ""


@dataclass(frozen=True, slots=True)
class ToolResultBlock(ContentBlock):
    tool_use_id: ToolCallID
    content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock]
    is_error: bool = False


def proof(block: TextBlock | ReasoningBlock | ToolUseBlock, provider: str) -> str:
    """This block's opaque proof, where the protocol about to receive it is the one that issued it.

    Each converter reads the same ``signature`` field by its own rules, so a session that changed
    transport sent a proof one provider made to another that never made it. At best the request is
    refused; at worst the value is accepted somewhere it was never meant to sit, or printed by
    whatever logs the request.

    A block with no recorded provider was stored before anyone recorded one, and the transport
    reading it is almost always the one that wrote it. Those are replayed, because dropping them
    loses proofs that are valid and breaks the sessions that already exist.
    """
    if not block.signature:
        return ""
    if block.provider and block.provider != provider:
        logger.debug("Not replaying a %s proof to %s", block.provider, provider)
        return ""
    return block.signature


def replayable(block: ProviderBlock, provider: str) -> bool:
    """Whether this opaque item goes back to the endpoint about to be asked.

    Unlike a proof, an item with no provider recorded is not replayed: every one of them was made
    by a reader that names itself, so an empty name is a block from somewhere else entirely.
    """
    if block.provider == provider:
        return True
    logger.debug("Not replaying a %s %s item to %s", block.provider or "unattributed", block.kind, provider)
    return False


@singledispatch
def to_dict(block: ContentBlock) -> dict[str, Any]:
    """Serialize a ContentBlock to a plain dict."""
    msg = f"Unknown block type: {type(block).__name__}"
    raise TypeError(msg)


@to_dict.register(TextBlock)
def _text_to_dict(block: TextBlock) -> dict[str, Any]:
    # Only when signed: text is the commonest block, and an empty key on each grows every session.
    out: dict[str, Any] = {"type": "text", "text": block.text}
    if block.signature:
        out["signature"] = block.signature
        # Beside the proof and never without it: a signature restored with no provider is one
        # nothing can judge, which is the state this field exists to end.
        out["provider"] = block.provider
    return out


@to_dict.register(ImageBlock)
def _image_to_dict(block: ImageBlock) -> dict[str, Any]:
    return {"type": "image", "media_type": block.media_type, "data": base64.b64encode(block.data).decode()}


@to_dict.register(AudioBlock)
def _audio_to_dict(block: AudioBlock) -> dict[str, Any]:
    return {"type": "audio", "media_type": block.media_type, "data": base64.b64encode(block.data).decode()}


@to_dict.register(VideoBlock)
def _video_to_dict(block: VideoBlock) -> dict[str, Any]:
    return {"type": "video", "media_type": block.media_type, "data": base64.b64encode(block.data).decode()}


@to_dict.register(ReasoningBlock)
def _reasoning_to_dict(block: ReasoningBlock) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "text": block.text,
        "signature": block.signature,
        "redacted": block.redacted,
        "id": block.id,
        "provider": block.provider,
    }


@to_dict.register(ToolUseBlock)
def _tool_use_to_dict(block: ToolUseBlock) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": block.id,
        "name": block.name,
        "input": block.input,
        "signature": block.signature,
        "provider": block.provider,
    }


@to_dict.register(ProviderBlock)
def _provider_to_dict(block: ProviderBlock) -> dict[str, Any]:
    return {
        "type": "provider",
        "provider": block.provider,
        "kind": block.kind,
        "data": block.data,
        "id": block.id,
    }


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
            return TextBlock(
                text=data["text"],
                signature=data.get("signature", ""),
                provider=data.get("provider", ""),
            )
        case "image":
            return ImageBlock(media_type=data["media_type"], data=base64.b64decode(data["data"]))
        case "audio":
            return AudioBlock(media_type=data["media_type"], data=base64.b64decode(data["data"]))
        case "video":
            return VideoBlock(media_type=data["media_type"], data=base64.b64decode(data["data"]))
        case "reasoning":
            return ReasoningBlock(
                text=data.get("text", ""),
                signature=data.get("signature", ""),
                redacted=data.get("redacted", False),
                id=data.get("id", ""),
                provider=data.get("provider", ""),
            )
        case "tool_use":
            return ToolUseBlock(
                id=data["id"],
                name=data["name"],
                input=data["input"],
                signature=data.get("signature", ""),
                provider=data.get("provider", ""),
            )
        case "provider":
            return ProviderBlock(
                provider=data["provider"],
                kind=data["kind"],
                data=data["data"],
                id=data.get("id", ""),
            )
        case "tool_result":
            raw = data["content"]
            if isinstance(raw, str):
                content: str | list[TextBlock | ImageBlock | AudioBlock | VideoBlock] = raw
            else:
                content = [from_dict(b) for b in raw]  # type: ignore[misc]
            return ToolResultBlock(tool_use_id=data["tool_use_id"], content=content, is_error=data["is_error"])
        case _:
            msg = f"Unknown block type: {data['type']}"
            raise ValueError(msg)
