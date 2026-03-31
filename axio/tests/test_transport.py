"""Tests for transport protocols — structural protocol verification."""

from __future__ import annotations

from collections.abc import AsyncIterator

from axio.events import StreamEvent
from axio.messages import Message
from axio.tool import Tool
from axio.transport import CompletionTransport, ImageGenTransport, STTTransport, TTSTransport


class _MockCompletion:
    def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]:
        raise NotImplementedError

    stream.__doc__ = "Mock"


class _MockImageGen:
    async def generate(self, prompt: str, *, size: tuple[int, int] | None = None, n: int = 1) -> list[bytes]:
        return [b""]


class _MockTTS:
    def synthesize(self, text: str, *, voice: str | None = None) -> AsyncIterator[bytes]:
        raise NotImplementedError


class _MockSTT:
    async def transcribe(self, audio: bytes, media_type: str = "audio/wav") -> str:
        return ""


class TestProtocolConformance:
    def test_completion_transport(self) -> None:
        assert isinstance(_MockCompletion(), CompletionTransport)

    def test_image_gen_transport(self) -> None:
        assert isinstance(_MockImageGen(), ImageGenTransport)

    def test_tts_transport(self) -> None:
        assert isinstance(_MockTTS(), TTSTransport)

    def test_stt_transport(self) -> None:
        assert isinstance(_MockSTT(), STTTransport)
