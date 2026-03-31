"""Tests for Agent tool dispatch: invocation, errors, parallel execution."""

from __future__ import annotations

import json
from typing import Any

from axio.agent import Agent
from axio.blocks import ToolResultBlock
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    SessionEndEvent,
    StreamEvent,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.testing import MsgInput, StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool, ToolHandler
from axio.types import StopReason, Usage

_calls: list[dict[str, Any]] = []


class TrackingHandler(ToolHandler):
    msg: str

    async def __call__(self) -> str:
        _calls.append(self.model_dump())
        return self.model_dump_json()


class XHandler(ToolHandler):
    x: int

    async def __call__(self) -> str:
        return "a"


class YHandler(ToolHandler):
    y: int

    async def __call__(self) -> str:
        return "b"


class BadHandler(ToolHandler):
    async def __call__(self) -> str:
        raise ValueError("boom")


class TestToolInvocation:
    async def test_handler_called(self) -> None:
        _calls.clear()
        tool = Tool(name="echo", description="echo", handler=TrackingHandler)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(_calls) == 1
        assert _calls[0] == {"msg": "hi"}

    async def test_result_in_context(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        ctx = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", ctx)
        history = await ctx.get_history()
        user_msgs = [m for m in history if m.role == "user"]
        tool_results = [b for m in user_msgs for b in m.content if isinstance(b, ToolResultBlock)]
        assert len(tool_results) == 1
        assert tool_results[0].tool_use_id == "c1"
        assert not tool_results[0].is_error


class TestTwoToolsOneResponse:
    async def test_both_called(self) -> None:
        """C2: every ToolUseBlock has a corresponding ToolResultBlock."""
        calls: list[str] = []

        class AHandler(ToolHandler):
            x: int

            async def __call__(self) -> str:
                calls.append("a")
                return "a"

        class BHandler(ToolHandler):
            y: int

            async def __call__(self) -> str:
                calls.append("b")
                return "b"

        tool_a = Tool(name="a", description="a", handler=AHandler)
        tool_b = Tool(name="b", description="b", handler=BHandler)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "a"),
                    ToolInputDelta(0, "c1", json.dumps({"x": 1})),
                    ToolUseStart(1, "c2", "b"),
                    ToolInputDelta(1, "c2", json.dumps({"y": 2})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool_a, tool_b], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert set(calls) == {"a", "b"}


class TestUnknownTool:
    async def test_produces_error_result(self) -> None:
        """C9: unknown tool produces is_error=True, loop continues."""
        transport = StubTransport([make_tool_use_response("nonexistent", "c1", {}), make_text_response("Done")])
        agent = Agent(system="test", tools=[], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error


class TestHandlerException:
    async def test_exception_wrapped_as_error_result(self) -> None:
        tool = Tool(name="bad", description="bad", handler=BadHandler)
        transport = StubTransport([make_tool_use_response("bad", "c1", {}), make_text_response("Done")])
        ctx = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", ctx):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert isinstance(events[-1], SessionEndEvent)


class TestMalformedJson:
    async def test_malformed_json_returns_error_result(self) -> None:
        """Truncated JSON → ToolResult(is_error=True), loop continues."""
        tool = Tool(name="list_files", description="list files", handler=MsgInput)
        # Truncated JSON: '{"directory": ".'  (missing closing quote and brace)
        truncated = '{"msg": ".'
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "list_files"),
                    ToolInputDelta(0, "c1", truncated),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert tool_results[0].tool_use_id == "c1"

        # Loop should continue — we get a SessionEndEvent with end_turn
        session_ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert len(session_ends) == 1
        assert session_ends[0].stop_reason == StopReason.end_turn

    async def test_mixed_valid_and_malformed_tools(self) -> None:
        """Two parallel tool calls: one valid, one malformed. Valid runs, malformed errors."""
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        valid_args = json.dumps({"msg": "hello"})
        malformed_args = '{"msg": "trunc'

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", valid_args),
                    ToolUseStart(1, "c2", "echo"),
                    ToolInputDelta(1, "c2", malformed_args),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 2

        valid_result = next(r for r in tool_results if r.tool_use_id == "c1")
        malformed_result = next(r for r in tool_results if r.tool_use_id == "c2")

        assert not valid_result.is_error
        assert malformed_result.is_error


class TestToolResultCarriesData:
    async def test_content_and_input_populated(self) -> None:
        """ToolResult events carry the tool input dict and result content string."""
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        r = tool_results[0]
        assert r.input == {"msg": "hi"}
        assert r.content != ""
        assert not r.is_error

    async def test_error_result_has_content(self) -> None:
        """Error ToolResult events carry the error message as content."""
        tool = Tool(name="bad", description="bad", handler=BadHandler)
        transport = StubTransport([make_tool_use_response("bad", "c1", {}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        r = tool_results[0]
        assert r.is_error
        assert "boom" in r.content


class TestStopReasonOverride:
    async def test_stop_reason_override_when_tool_blocks_present(self) -> None:
        """Transport returns end_turn with tool calls → agent overrides to tool_use and dispatches."""
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        # Transport returns end_turn but includes tool call events
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.end_turn, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        # Tool should have been dispatched despite end_turn
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error

        # Session should end with end_turn (from the second iteration's text response)
        session_ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert len(session_ends) == 1
        assert session_ends[0].stop_reason == StopReason.end_turn
