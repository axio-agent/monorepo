"""Tests for AgentStream: interrupt, aclose, SessionEndEvent guarantee."""

from __future__ import annotations

from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import Error, IterationEnd, SessionEndEvent, StreamEvent, TextDelta
from axio.testing import StubTransport
from axio.types import StopReason, Usage


class TestSessionEndGuarantee:
    async def test_end_turn_has_session_end(self) -> None:
        """C1: SessionEndEvent is always last on end_turn."""
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    TextDelta(0, " there"),
                    TextDelta(0, "!"),
                    *[TextDelta(0, f" {i}") for i in range(10)],
                    *[TextDelta(0, f" extra{i}") for i in range(5)],
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", MemoryContextStore()):
            events.append(e)
        assert isinstance(events[-1], SessionEndEvent)

    async def test_error_has_session_end(self) -> None:
        """C1: SessionEndEvent is last even on error."""
        transport = StubTransport(
            [
                [
                    TextDelta(0, "partial"),
                    IterationEnd(1, StopReason.max_tokens, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", MemoryContextStore()):
            events.append(e)

        assert isinstance(events[-1], SessionEndEvent)
        assert events[-1].stop_reason == StopReason.error
        assert any(isinstance(e, Error) for e in events)


class TestAclose:
    async def test_aclose_mid_stream(self) -> None:
        """Verify aclose doesn't cause ResourceWarning."""
        transport = StubTransport(
            [
                [
                    TextDelta(0, "Hello"),
                    TextDelta(0, " world"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        stream = agent.run_stream("hi", MemoryContextStore())
        event = await stream.__anext__()
        assert isinstance(event, TextDelta)
        await stream.aclose()


class TestMultipleLoopsOnStream:
    async def test_second_loop_empty(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(1, 1)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        stream = agent.run_stream("hi", MemoryContextStore())
        first = [e async for e in stream]
        assert len(first) > 0
        second = [e async for e in stream]
        assert second == []
