"""AgentStream: async iterator wrapper over the agent event generator."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from .events import Error, Refusal, SessionEndEvent, StreamEvent, TextDelta
from .exceptions import StreamError
from .types import INCOMPLETE

logger = logging.getLogger(__name__)


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
        """Everything the turn said, and nothing about whether it finished saying it.

        A run ending on one of :data:`~axio.types.INCOMPLETE` returns a truncated answer that
        reads exactly like a whole one, because a ``str`` has nowhere to put the reason. It is
        logged as a warning here, and :meth:`get_session_end` carries it for a caller that needs
        to branch on it. ``Error`` still raises, so a broken turn is never returned as an answer.
        """
        parts: list[str] = []
        try:
            async for event in self:
                if isinstance(event, SessionEndEvent) and event.stop_reason in INCOMPLETE:
                    logger.warning(
                        "Returning an answer the model did not finish: the run ended on %s",
                        event.stop_reason,
                    )
                if isinstance(event, Error):
                    raise StreamError(str(event.exception)) from event.exception
                if isinstance(event, TextDelta):
                    parts.append(event.delta)
                if isinstance(event, Refusal):
                    # A refusal arrives instead of the answer, never beside it, so it is what the
                    # turn said. Collected nowhere, run() returned an empty string for a turn that
                    # had text the caller needed to see.
                    parts.append(event.text)
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
