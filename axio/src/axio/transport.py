"""Transport protocols: completion, image gen, TTS, STT.

Transports should be stateless - all request state lives in the arguments
passed to each method call.  This allows multiple agents to share a single
transport instance and call it concurrently without interference.

The one allowed exception is a reusable connection pool (e.g. an
``aiohttp.ClientSession``), which is safe to share across concurrent calls.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .blocks import ContentBlock
from .events import StreamEvent
from .messages import Message
from .tool import Tool
from .types import ToolCallID, ToolName

logger = logging.getLogger(__name__)


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class ImageGenTransport(Protocol):
    """Generate ``n`` image samples for a text prompt.  Returns raw image bytes
    (PNG / JPEG / WebP — provider-defined)."""

    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]: ...


@runtime_checkable
class VideoGenTransport(Protocol):
    """Generate ``n`` video samples for a text prompt.  Returns raw video bytes
    (MP4 / WebM — provider-defined).  Provider-specific knobs (duration,
    aspect ratio, seed image, etc.) live as extra kwargs on the implementation."""

    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]: ...


@runtime_checkable
class AudioGenTransport(Protocol):
    """Generate ``n`` non-speech audio samples for a text prompt — music,
    sound effects, ambient.  Returns raw audio bytes (MP3 / WAV / OGG —
    provider-defined).  Distinct from :class:`TTSTransport`, which is
    text-to-speech."""

    async def generate_audios(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]: ...


@runtime_checkable
class RealtimeSession(Protocol):
    """Active duplex realtime session — bidirectional audio / text / tools.

    Returned by :meth:`RealtimeTransport.connect`.  Events from the provider
    arrive on :meth:`events`.  User input is pushed via :meth:`send`.
    """

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        """Append user content (audio chunk, text, image) to the input buffer."""

    async def commit(self) -> None:
        """Signal end-of-utterance for manual VAD; no-op with server VAD."""

    async def interrupt(self) -> None:
        """Abort in-flight assistant generation (e.g. user interrupted)."""

    async def send_tool_result(
        self, tool_use_id: ToolCallID, name: ToolName, content: str | list[ContentBlock]
    ) -> None:
        """Deliver a tool's result to the provider so generation can resume.

        ``name`` is included because some providers (e.g. Gemini Live) require
        the tool name alongside the call id.  OpenAI realtime can ignore it.
        """

    def events(self) -> AsyncIterator[StreamEvent]:
        """Async iterator over server events for the lifetime of this session."""

    async def close(self) -> None:
        """Tear down the session and release resources."""


@runtime_checkable
class RealtimeTransport(Protocol):
    """Provider that supports duplex realtime sessions (e.g. OpenAI Realtime,
    Gemini Live).  Distinct from :class:`CompletionTransport` because the
    interaction is bidirectional, not request/response."""

    async def connect(
        self,
        *,
        system: str,
        tools: list[Tool[Any]],
        voice: str | None = None,
        input_audio_format: str = "audio/pcm;rate=16000",
        output_audio_format: str = "audio/pcm;rate=24000",
    ) -> RealtimeSession: ...


@runtime_checkable
class TTSTransport(Protocol):
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]: ...


@runtime_checkable
class STTTransport(Protocol):
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str: ...


@runtime_checkable
class EmbeddingTransport(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class DummyTransport:
    """Placeholder transport that fails loudly if actually used.

    Assign this as the default transport when constructing agent prototypes
    that will be configured later via ``agent.copy(transport=real_transport)``.

    Example::

        from axio.agent import Agent
        from axio.transport import DummyCompletionTransport

        researcher = Agent(
            system="You are a research assistant...",
            transport=DummyCompletionTransport(),
        )

        # At runtime, swap in the real transport:
        active = researcher.copy(transport=OpenAITransport())
        result = await active.run(task, context)
    """

    @staticmethod
    def _do_fail() -> None:
        logger.warning(
            "DummyCompletionTransport.stream() called - this agent has no real transport. "
            "Use agent.copy(transport=<real_transport>) before running."
        )
        raise RuntimeError(
            "DummyCompletionTransport is a placeholder. Configure a real transport with agent.copy(transport=...)."
        )


class DummyCompletionTransport(DummyTransport, CompletionTransport):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyImageGenTransport(DummyTransport, ImageGenTransport):
    async def generate_images(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyVideoGenTransport(DummyTransport, VideoGenTransport):
    async def generate_videos(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        image: bytes | None = None,
        duration_seconds: int | None = None,
        aspect_ratio: str | None = None,
    ) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyAudioGenTransport(DummyTransport, AudioGenTransport):
    async def generate_audios(self, prompt: str, *, model: str | None = None, n: int = 1) -> list[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyTTSTransport(DummyTransport, TTSTransport):
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        self._do_fail()
        raise AssertionError("unreachable")


class DummySTTTransport(DummyTransport, STTTransport):
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str:
        self._do_fail()
        raise AssertionError("unreachable")


class DummyEmbeddingTransport(DummyTransport, EmbeddingTransport):
    def embed(self, texts: list[str]) -> Any:
        self._do_fail()
