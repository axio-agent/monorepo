"""OpenAI Realtime WebSocket transport.

Implements :class:`axio.transport.RealtimeTransport` against the OpenAI
Realtime API (``wss://api.openai.com/v1/realtime``).  Provider events are
mapped to axio :class:`StreamEvent` variants.  The session's
``send`` / ``commit`` / ``interrupt`` / ``send_tool_result`` methods are
mapped to the corresponding client events.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from axio.blocks import AudioBlock, ContentBlock, TextBlock
from axio.events import (
    AudioOutputDelta,
    Error,
    SpeechStarted,
    SpeechStopped,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    TranscriptDelta,
    TurnComplete,
)
from axio.exceptions import StreamError
from axio.tool import Tool
from axio.transport import RealtimeSession, RealtimeTransport
from axio.types import StopReason, ToolCallID, ToolName, Usage

logger = logging.getLogger(__name__)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_PCM_MEDIA_TYPE = "audio/pcm;rate=24000"


def _convert_realtime_tools(tools: list[Tool[Any]]) -> list[dict[str, Any]]:
    """Convert axio ``Tool``s to OpenAI realtime tool entries (flat ``function`` shape)."""
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        }
        for t in tools
    ]


def _flat_audio_format(media_type: str) -> str:
    """Translate axio audio media-type strings to OpenAI realtime v1 format strings.

    realtime=v1 takes a flat string (``"pcm16"`` / ``"g711_ulaw"`` / ``"g711_alaw"``).
    24 kHz PCM16 mono is the only sample rate the API currently supports.
    """
    if media_type.startswith("audio/pcm"):
        return "pcm16"
    if media_type.startswith("audio/pcmu"):
        return "g711_ulaw"
    if media_type.startswith("audio/pcma"):
        return "g711_alaw"
    raise ValueError(f"OpenAI realtime: unsupported audio format {media_type!r}")


@dataclass(slots=True)
class OpenAIRealtimeSession(RealtimeSession):
    """RealtimeSession backed by an aiohttp WebSocket.

    ``ws`` may be any object that quacks like
    :class:`aiohttp.ClientWebSocketResponse` (``send_str``, async iteration
    yielding :class:`aiohttp.WSMessage`-like objects, ``close``, ``closed``).
    Tests inject a stub.
    """

    ws: Any
    output_audio_media_type: str = DEFAULT_PCM_MEDIA_TYPE
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False)
    own_http_session: bool = False
    _tool_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _next_tool_index: int = field(default=0, init=False, repr=False)
    _audio_appended: bool = field(default=False, init=False, repr=False)

    async def _send_event(self, payload: dict[str, Any]) -> None:
        await self.ws.send_str(json.dumps(payload))

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            match block:
                case AudioBlock(data=data):
                    self._audio_appended = True
                    await self._send_event(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(data).decode(),
                        }
                    )
                case TextBlock(text=text):
                    await self._send_event(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": text}],
                            },
                        }
                    )
                case _:
                    raise TypeError(f"OpenAI realtime: send() does not support {type(block).__name__}")

    async def commit(self) -> None:
        # OpenAI rejects input_audio_buffer.commit on an empty buffer
        # ("input_audio_buffer_commit_empty").  Skip it for text-only turns.
        if self._audio_appended:
            await self._send_event({"type": "input_audio_buffer.commit"})
            self._audio_appended = False
        await self._send_event({"type": "response.create"})

    async def interrupt(self) -> None:
        await self._send_event({"type": "response.cancel"})

    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None:
        if isinstance(content, list):
            parts: list[str] = []
            for b in content:
                if isinstance(b, TextBlock):
                    parts.append(b.text)
                else:
                    raise TypeError(
                        f"OpenAI realtime tool result only supports str / TextBlock, got {type(b).__name__}"
                    )
            output = "".join(parts)
        else:
            output = content
        await self._send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": tool_use_id,
                    "output": output,
                },
            }
        )
        await self._send_event({"type": "response.create"})

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for msg in self.ws:
            mtype = getattr(msg, "type", None)
            if mtype is aiohttp.WSMsgType.TEXT:
                payload = json.loads(msg.data)
                for ev in self._translate(payload):
                    yield ev
            elif mtype is aiohttp.WSMsgType.ERROR:
                logger.error("OpenAI realtime ws error: %s", self.ws.exception())
                break
            elif mtype in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                break

    def _translate(self, ev: dict[str, Any]) -> list[StreamEvent]:
        """Map a raw OpenAI realtime server event to axio StreamEvents."""
        out: list[StreamEvent] = []
        t = ev.get("type")
        # Both the v1 (realtime=v1) and the newer GA schemas appear on this
        # endpoint depending on the model id.  Accept both event-name variants.
        if t in ("response.audio.delta", "response.output_audio.delta"):
            out.append(
                AudioOutputDelta(
                    data=base64.b64decode(ev["delta"]),
                    media_type=self.output_audio_media_type,
                )
            )
        elif t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
            out.append(TranscriptDelta(role="assistant", delta=ev["delta"]))
        elif t == "conversation.item.input_audio_transcription.delta":
            out.append(TranscriptDelta(role="user", delta=ev["delta"]))
        elif t == "response.text.delta":
            out.append(TextDelta(index=0, delta=ev["delta"]))
        elif t == "input_audio_buffer.speech_started":
            out.append(SpeechStarted())
        elif t == "input_audio_buffer.speech_stopped":
            out.append(SpeechStopped())
        elif t == "response.output_item.added":
            item = ev.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item["call_id"]
                idx = self._next_tool_index
                self._next_tool_index += 1
                self._tool_index[call_id] = idx
                out.append(ToolUseStart(index=idx, tool_use_id=call_id, name=item["name"]))
        elif t == "response.function_call_arguments.delta":
            call_id = ev["call_id"]
            idx = self._tool_index.get(call_id, 0)
            out.append(ToolInputDelta(index=idx, tool_use_id=call_id, partial_json=ev["delta"]))
        elif t == "response.done":
            resp = ev.get("response") or {}
            items = resp.get("output") or []
            stop_reason = (
                StopReason.tool_use if any(it.get("type") == "function_call" for it in items) else StopReason.end_turn
            )
            usage_data = resp.get("usage")
            usage: Usage | None = None
            if usage_data:
                usage = Usage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                )
            out.append(TurnComplete(stop_reason=stop_reason, usage=usage))
            self._tool_index.clear()
            self._next_tool_index = 0
        elif t == "error":
            err = ev.get("error") or {}
            msg = err.get("message") or str(err)
            code = err.get("code") or err.get("type") or "unknown"
            logger.error("OpenAI realtime error [%s]: %s", code, msg)
            out.append(Error(exception=StreamError(f"OpenAI realtime [{code}]: {msg}")))
        else:
            logger.debug("OpenAI realtime: ignoring event %s", t)
        return out

    async def close(self) -> None:
        if not self.ws.closed:
            await self.ws.close()
        if self.own_http_session and self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()


@dataclass(slots=True)
class OpenAIRealtimeTransport(RealtimeTransport):
    """RealtimeTransport for OpenAI's Realtime WebSocket API.

    Server-VAD knobs live here so callers can tune barge-in sensitivity
    without touching the transport internals.  When AEC is imperfect or
    the room is loud, the defaults can cause the model's own audio to
    trip ``interrupt_response`` and cancel its response.  Bump
    ``vad_threshold`` higher, lengthen ``vad_silence_duration_ms``, or
    set ``vad_interrupt_response=False`` to require an explicit
    ``agent.interrupt()``.
    """

    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_REALTIME_URL", OPENAI_REALTIME_URL))
    model: str = DEFAULT_REALTIME_MODEL
    http_session: aiohttp.ClientSession | None = field(default=None, repr=False, compare=False)

    vad_threshold: float = 0.5
    """Server VAD activation threshold (0..1).  Higher → less sensitive to
    quiet sounds and bleed-through.  Default 0.5."""

    vad_prefix_padding_ms: int = 300
    """Audio retained before a detected speech start."""

    vad_silence_duration_ms: int = 500
    """Silence required before the server considers a turn complete."""

    vad_create_response: bool = True
    """Whether the server auto-creates a response after each user turn."""

    vad_interrupt_response: bool = True
    """Whether incoming user speech cancels the model's in-flight response.
    Setting this to False makes interruption explicit (call
    ``agent.interrupt()``) — useful when AEC isn't strong enough and the
    model's own voice would otherwise loop back through the mic and
    interrupt itself."""

    input_noise_reduction: str | None = "near_field"
    """Server-side noise reduction profile applied to mic audio.  Options:
    ``"near_field"`` (close-talking mic / headset), ``"far_field"`` (laptop
    or conference mic), or ``None`` to disable."""

    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=24000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession:
        own_session = False
        if self.http_session is None or self.http_session.closed:
            self.http_session = aiohttp.ClientSession()
            own_session = True
        url = f"{self.base_url}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        ws = await self.http_session.ws_connect(url, headers=headers)
        # The realtime=v1 (gpt-4o-realtime / gpt-realtime) WebSocket schema is
        # flat — instructions / voice / *_audio_format / turn_detection / tools
        # live directly on session.  The nested {audio: {input, output}} shape
        # belongs to the newer HTTP session-create REST endpoint.  Sent here it
        # is rejected with "Unknown parameter: 'session.audio'".
        session_payload: dict[str, Any] = {
            "instructions": system,
            "input_audio_format": _flat_audio_format(input_audio_format),
            "output_audio_format": _flat_audio_format(output_audio_format),
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.vad_prefix_padding_ms,
                "silence_duration_ms": self.vad_silence_duration_ms,
                "create_response": self.vad_create_response,
                "interrupt_response": self.vad_interrupt_response,
            },
            "tools": _convert_realtime_tools(tools),
        }
        if self.input_noise_reduction is not None:
            session_payload["input_audio_noise_reduction"] = {"type": self.input_noise_reduction}
        if voice:
            session_payload["voice"] = voice
        await ws.send_str(json.dumps({"type": "session.update", "session": session_payload}))
        return OpenAIRealtimeSession(
            ws=ws,
            output_audio_media_type=output_audio_format,
            http_session=self.http_session,
            own_http_session=own_session,
        )
