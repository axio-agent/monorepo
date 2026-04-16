"""Tests for Agent.run_stream() and run(): core loop, stop reasons, usage."""

from __future__ import annotations

from collections.abc import AsyncIterator

from axio.agent import Agent
from axio.blocks import TextBlock
from axio.context import MemoryContextStore
from axio.events import IterationEnd, ReasoningDelta, SessionEndEvent, StreamEvent, TextDelta, ToolResult
from axio.messages import Message
from axio.testing import MsgInput, StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool, ToolHandler
from axio.types import StopReason, Usage


class CapturingTransport:
    """Records messages passed to each stream() call."""

    def __init__(self, responses: list[list[StreamEvent]]) -> None:
        self._responses = responses
        self._call_count = 0
        self.calls: list[list[Message]] = []

    async def _generate(self, events: list[StreamEvent]) -> AsyncIterator[StreamEvent]:
        for event in events:
            yield event

    def stream(self, messages: list[Message], tools: list[Tool], system: str) -> AsyncIterator[StreamEvent]:
        self.calls.append(list(messages))
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._generate(self._responses[idx])


class OkHandler(ToolHandler):
    msg: str

    async def __call__(self) -> str:
        return "ok"


class TestRunStream:
    async def test_end_turn_yields_text_and_session_end(self) -> None:
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
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", MemoryContextStore()):
            events.append(e)

        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_session_end_total_usage(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        end = await agent.run_stream("hi", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(10, 5)


class TestRun:
    async def test_returns_concatenated_text(self) -> None:
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
        result = await agent.run("hi", MemoryContextStore())
        assert result == "Hello world"


class TestMultiIteration:
    async def test_tool_use_then_end_turn(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("do it", MemoryContextStore()):
            events.append(e)

        iteration_ends = [e for e in events if isinstance(e, IterationEnd)]
        assert len(iteration_ends) == 2
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.end_turn

    async def test_total_usage_across_iterations(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        end = await agent.run_stream("go", MemoryContextStore()).get_session_end()
        assert end.total_usage == Usage(13, 12)


class TestContextTokenTracking:
    async def test_agent_updates_context_tokens(self) -> None:
        transport = StubTransport(
            [
                [
                    TextDelta(0, "hi"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (10, 5)

    async def test_agent_accumulates_context_tokens_across_iterations(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1, Usage(10, 5)),
                make_text_response("Done", 2, Usage(3, 7)),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        context = MemoryContextStore()
        await agent.run("go", context)
        assert await context.get_context_tokens() == (13, 12)


class TestReasoningPassthrough:
    async def test_reasoning_delta_yielded_but_not_stored(self) -> None:
        """ReasoningDelta events pass through the stream but are NOT stored in context."""
        transport = StubTransport(
            [
                [
                    ReasoningDelta(0, "thinking..."),
                    TextDelta(0, "answer"),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ]
            ]
        )
        agent = Agent(system="test", tools=[], transport=transport)
        context = MemoryContextStore()
        events: list[StreamEvent] = []
        async for e in agent.run_stream("hi", context):
            events.append(e)

        # ReasoningDelta is yielded
        reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
        assert len(reasoning) == 1
        assert reasoning[0].delta == "thinking..."

        # TextDelta is yielded
        text = [e for e in events if isinstance(e, TextDelta)]
        assert len(text) == 1
        assert text[0].delta == "answer"

        # Only text is stored in assistant message, not reasoning
        history = await context.get_history()
        assistant_msgs = [m for m in history if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        from axio.blocks import TextBlock

        text_blocks = [b for b in assistant_msgs[0].content if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "answer"


class TestMaxIterations:
    async def test_max_iterations_reached(self) -> None:
        """C7: max_iterations emits SessionEndEvent(stop_reason=error)."""
        tool = Tool(name="echo", description="echo", handler=OkHandler)
        transport = StubTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_tool_use_response("echo", "c2", {"msg": "hi"}, 2),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, max_iterations=1)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        last = events[-1]
        assert isinstance(last, SessionEndEvent)
        assert last.stop_reason == StopReason.error


class TestLastIterationMessage:
    async def test_injected_only_on_last_iteration(self) -> None:
        """last_iteration_message is appended to history only on the final iteration."""
        tool = Tool(name="echo", description="echo", handler=OkHandler)
        hint = Message(role="system", content=[TextBlock(text="wrap up now")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        await agent.run("go", MemoryContextStore())

        # iteration 1: hint NOT in history
        assert hint not in transport.calls[0]
        # iteration 2 (last): hint IS the final message
        assert transport.calls[1][-1] is hint

    async def test_not_injected_when_none(self) -> None:
        """No injection when last_iteration_message is None (default)."""
        transport = CapturingTransport([make_text_response("hi", 1)])
        agent = Agent(system="test", tools=[], transport=transport)
        await agent.run("go", MemoryContextStore())

        history = transport.calls[0]
        assert all(m.role != "system" for m in history)

    async def test_not_stored_in_context(self) -> None:
        """last_iteration_message is injected into the stream but not persisted."""
        tool = Tool(name="echo", description="echo", handler=OkHandler)
        hint = Message(role="system", content=[TextBlock(text="wrap up")])
        transport = CapturingTransport(
            [
                make_tool_use_response("echo", "c1", {"msg": "hi"}, 1),
                make_text_response("Done", 2),
            ]
        )
        agent = Agent(
            system="test",
            tools=[tool],
            transport=transport,
            max_iterations=2,
            last_iteration_message=hint,
        )
        context = MemoryContextStore()
        await agent.run("go", context)

        history = await context.get_history()
        assert hint not in history
