"""AgentStream: async iterator wrapper over the agent event generator."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from axio.events import Error, SessionEndEvent, StreamEvent, TextDelta
from axio.exceptions import StreamError


class AgentStream:
    def __init__(self, generator: AsyncGenerator[StreamEvent, None]) -> None:
        self._generator = generator
        self._closed = False

    def __aiter__(self) -> AgentStream:
        return self

    async def __anext__(self) -> StreamEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._generator.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._generator.aclose()

    async def get_final_text(self) -> str:
        parts: list[str] = []
        try:
            async for event in self:
                if isinstance(event, Error):
                    raise StreamError(str(event.exception)) from event.exception
                if isinstance(event, TextDelta):
                    parts.append(event.delta)
        finally:
            await self.aclose()
        return "".join(parts)

    async def get_session_end(self) -> SessionEndEvent:
        result: SessionEndEvent | None = None
        try:
            async for event in self:
                if isinstance(event, Error):
                    raise StreamError(str(event.exception)) from event.exception
                if isinstance(event, SessionEndEvent):
                    result = event
        finally:
            await self.aclose()
        if result is None:
            raise StreamError("Stream ended without SessionEndEvent")
        return result
