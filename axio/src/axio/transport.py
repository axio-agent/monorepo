"""Transport protocols: completion, image gen, TTS, STT."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]: ...


@runtime_checkable
class ImageGenTransport(Protocol):
    async def generate(self, prompt: str, *, size: tuple[int, int] | None = None, n: int = 1) -> list[bytes]: ...


@runtime_checkable
class TTSTransport(Protocol):
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]: ...


@runtime_checkable
class STTTransport(Protocol):
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str: ...


@runtime_checkable
class EmbeddingTransport(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
