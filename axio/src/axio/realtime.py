"""Realtime duplex agent: drives a :class:`RealtimeSession`, dispatches
tools concurrently with audio output, exposes a single event stream to the
caller."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from .blocks import ContentBlock, ToolUseBlock
from .events import (
    Error,
    StreamEvent,
    ToolInputDelta,
    ToolUseStart,
    TurnComplete,
)
from .tool import Tool
from .transport import RealtimeSession, RealtimeTransport
from .types import StopReason

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RealtimeAgent:
    """Drives a duplex realtime session against a :class:`RealtimeTransport`.

    The agent intercepts ``ToolUseStart``/``ToolInputDelta``/``TurnComplete``
    events to assemble tool calls and dispatches them as background tasks so
    that streaming audio output from the provider is not blocked.  Each tool
    result is sent back to the session via :meth:`RealtimeSession.send_tool_result`
    as soon as the handler returns.  ``interrupt()`` cancels in-flight tasks.

    Lifecycle::

        async with RealtimeAgent(system="...", transport=t, tools=[...]) as agent:
            await agent.send(AudioBlock(data=mic_chunk, media_type="audio/pcm"))
            async for event in agent.events():
                match event:
                    case AudioOutputDelta(data=pcm):
                        speaker.feed(pcm)
                    ...
    """

    system: str
    transport: RealtimeTransport
    tools: list[Tool[Any]] = field(default_factory=list)
    voice: str | None = None
    input_audio_format: str = "audio/pcm;rate=16000"
    output_audio_format: str = "audio/pcm;rate=24000"
    raise_on_error: bool = True
    """Re-raise the exception wrapped in any :class:`Error` event yielded by
    the session.  Set to ``False`` to receive ``Error`` events as data and
    decide what to do per error (transient retry, log-and-continue, etc.)."""

    _session: RealtimeSession | None = field(default=None, init=False, repr=False)
    _pending: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _tool_tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False, repr=False)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is not None:
            raise RuntimeError("Session already connected.")
        self._session = await self.transport.connect(
            system=self.system,
            tools=self.tools,
            voice=self.voice,
            input_audio_format=self.input_audio_format,
            output_audio_format=self.output_audio_format,
        )

    async def close(self) -> None:
        # Wait for in-flight tools to settle (close = graceful).
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> RealtimeSession:
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() or use 'async with'.")
        return self._session

    async def send(self, content: ContentBlock | list[ContentBlock]) -> None:
        await self.session.send(content)

    async def commit(self) -> None:
        await self.session.commit()

    async def interrupt(self) -> None:
        # Cancel running tool tasks AND drop any half-streamed pending tool
        # fragments. Once the user has interrupted, neither the in-flight
        # results nor a TurnComplete arriving afterwards should be allowed to
        # finalize stale tool calls.
        self._pending.clear()
        for task in list(self._tool_tasks):
            task.cancel()
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        await self.session.interrupt()

    def _find_tool(self, name: str) -> Tool[Any] | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    async def _dispatch_tool(self, block: ToolUseBlock) -> None:
        """Execute one tool and forward its result back to the session."""
        session = self._session
        if session is None:
            return
        tool = self._find_tool(block.name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", block.name)
            await self._safe_send_tool_result(block.id, block.name, f"Unknown tool: {block.name}")
            return
        try:
            result = await tool(**block.input)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Tool %s raised %s: %s", block.name, type(exc).__name__, exc, exc_info=True)
            await self._safe_send_tool_result(block.id, block.name, f"error: {exc}")
            return
        if isinstance(result, str):
            content: str | list[ContentBlock] = result
        elif isinstance(result, list):
            content = result
        else:
            content = str(result)
        await self._safe_send_tool_result(block.id, block.name, content)

    async def _safe_send_tool_result(
        self, tool_use_id: str, tool_name: str, content: str | list[ContentBlock]
    ) -> None:
        """Deliver a tool result, logging (instead of raising) any transport
        error so a wedged WebSocket cannot silently swallow tool output via
        the dispatch task's ``return_exceptions=True`` gather."""
        session = self._session
        if session is None:
            return
        try:
            await session.send_tool_result(tool_use_id, tool_name, content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Failed to deliver tool result %s (%s): %s: %s",
                tool_use_id,
                tool_name,
                type(exc).__name__,
                exc,
                exc_info=True,
            )

    def _finalize_pending_tools(self) -> list[ToolUseBlock]:
        """Convert accumulated streaming fragments into ``ToolUseBlock``s."""
        blocks: list[ToolUseBlock] = []
        for tid, info in self._pending.items():
            raw = "".join(info["json_parts"])
            try:
                inp = json.loads(raw) if raw else {}
            except json.JSONDecodeError as exc:
                logger.warning("Tool %s (id=%s) malformed JSON: %s", info["name"], tid, exc)
                inp = {}
            blocks.append(ToolUseBlock(id=tid, name=info["name"], input=inp))
        self._pending.clear()
        return blocks

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Yield events from the session, dispatching tool calls as side-effects.

        Tool dispatch runs as background tasks — slow handlers do not block
        the stream of ``AudioOutputDelta`` from the provider.
        """
        session = self.session
        try:
            async for event in session.events():
                if isinstance(event, Error) and self.raise_on_error:
                    # Surface the wrapped exception so the caller's ``async for``
                    # terminates with a real exception instead of silently
                    # waiting for events that will never arrive.
                    raise event.exception
                yield event
                match event:
                    case ToolUseStart(tool_use_id=tid, name=name):
                        self._pending[tid] = {"name": name, "json_parts": []}
                    case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                        if tid in self._pending:
                            self._pending[tid]["json_parts"].append(pj)
                    case TurnComplete(stop_reason=sr) if sr is StopReason.tool_use and self._pending:
                        for block in self._finalize_pending_tools():
                            task = asyncio.create_task(
                                self._dispatch_tool(block),
                                name=f"realtime-tool-{block.name}-{block.id}",
                            )
                            self._tool_tasks.add(task)
                            task.add_done_callback(self._tool_tasks.discard)
                    case _:
                        pass
        finally:
            # Drain in-flight tool tasks so callers observe deterministic
            # completion when they exit the loop.
            if self._tool_tasks:
                await asyncio.gather(*self._tool_tasks, return_exceptions=True)
