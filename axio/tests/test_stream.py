"""Tests for axio.stream: AgentStream lifecycle and collectors."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import Error, IterationEnd, Refusal, SessionEndEvent, StreamEvent, TextDelta
from axio.exceptions import StreamError
from axio.stream import AgentStream
from axio.testing import StubTransport
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


class TestRefusalReachesTheResult:
    async def test_run_returns_the_refusal_text(self) -> None:
        # A refusal arrives instead of the answer, never beside it. Collected nowhere, the public
        # run() API returned an empty string for a turn that had text the caller needed to see.
        transport = StubTransport(
            [
                [
                    Refusal(index=0, text="I cannot help with that"),
                    IterationEnd(1, StopReason.refusal, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="", tools=[], transport=transport)
        assert await agent.run("hi", MemoryContextStore()) == "I cannot help with that"

    async def test_a_refusal_that_arrives_in_fragments_is_joined(self) -> None:
        transport = StubTransport(
            [
                [
                    Refusal(index=0, text="I cannot "),
                    Refusal(index=0, text="help with that"),
                    IterationEnd(1, StopReason.refusal, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="", tools=[], transport=transport)
        assert await agent.run("hi", MemoryContextStore()) == "I cannot help with that"
