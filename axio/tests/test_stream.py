"""Tests for axon.stream: AgentStream lifecycle and collectors."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from axio.events import Error, SessionEndEvent, StreamEvent, TextDelta
from axio.exceptions import StreamError
from axio.stream import AgentStream
from axio.types import StopReason, Usage


async def _make_gen(events: list[StreamEvent]) -> AsyncGenerator[StreamEvent, None]:
    for e in events:
        yield e


def _simple_events() -> list[StreamEvent]:
    return [
        TextDelta(0, "Hello"),
        TextDelta(0, " world"),
        SessionEndEvent(StopReason.end_turn, Usage(10, 5)),
    ]


class TestAsyncIteration:
    async def test_yields_all_events(self) -> None:
        events = _simple_events()
        stream = AgentStream(_make_gen(events))
        collected = [e async for e in stream]
        assert collected == events

    async def test_aclose_stops_iteration(self) -> None:
        stream = AgentStream(_make_gen(_simple_events()))
        await stream.aclose()
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    async def test_break_mid_stream(self) -> None:
        stream = AgentStream(_make_gen(_simple_events()))
        async for event in stream:
            if isinstance(event, TextDelta):
                break
        await stream.aclose()


class TestGetFinalText:
    async def test_returns_concatenated_text(self) -> None:
        stream = AgentStream(_make_gen(_simple_events()))
        assert await stream.get_final_text() == "Hello world"

    async def test_raises_on_error_event(self) -> None:
        events: list[StreamEvent] = [
            Error(RuntimeError("boom")),
            SessionEndEvent(StopReason.error, Usage(0, 0)),
        ]
        stream = AgentStream(_make_gen(events))
        with pytest.raises(StreamError, match="boom"):
            await stream.get_final_text()


class TestGetSessionEnd:
    async def test_returns_session_end(self) -> None:
        stream = AgentStream(_make_gen(_simple_events()))
        end = await stream.get_session_end()
        assert isinstance(end, SessionEndEvent)
        assert end.stop_reason == StopReason.end_turn
        assert end.total_usage == Usage(10, 5)

    async def test_raises_on_error_event(self) -> None:
        events: list[StreamEvent] = [
            Error(RuntimeError("fail")),
            SessionEndEvent(StopReason.error, Usage(0, 0)),
        ]
        stream = AgentStream(_make_gen(events))
        with pytest.raises(StreamError, match="fail"):
            await stream.get_session_end()


class TestMultipleLoops:
    async def test_second_loop_empty(self) -> None:
        stream = AgentStream(_make_gen(_simple_events()))
        _ = [e async for e in stream]
        second = [e async for e in stream]
        assert second == []
