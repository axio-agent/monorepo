"""Transport protocols: completion, image gen, TTS, STT.

Transports should be stateless — all request state lives in the arguments
passed to each method call.  This allows multiple agents to share a single
transport instance and call it concurrently without interference.

The one allowed exception is a reusable connection pool (e.g. an
``aiohttp.ClientSession``), which is safe to share across concurrent calls.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .events import StreamEvent
from .messages import Message
from .tool import Tool

logger = logging.getLogger(__name__)


@runtime_checkable
class CompletionTransport(Protocol):
    def stream(self, messages: list[Message], tools: list[Tool[Any]], system: str) -> AsyncIterator[StreamEvent]: ...


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
            "DummyCompletionTransport.stream() called — this agent has no real transport. "
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
    async def generate(self, prompt: str, *, size: tuple[int, int] | None = None, n: int = 1) -> list[bytes]:
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
