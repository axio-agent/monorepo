"""Tests for Agent tool dispatch: invocation, errors, parallel execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from axio.agent import Agent
from axio.blocks import ImageBlock, ToolResultBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    SessionEndEvent,
    StreamEvent,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.testing import StubTransport, make_echo_tool, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio.types import StopReason, Usage

calls_log: list[dict[str, Any]] = []


async def _tracking(msg: str) -> str:
    data = {"msg": msg}
    calls_log.append(data)
    return json.dumps(data)


async def _handler_x(x: int) -> str:
    return "a"


async def _handler_y(y: int) -> str:
    return "b"


async def _bad(**kwargs: object) -> str:
    raise ValueError("boom")


class TestToolInvocation:
    async def test_handler_called(self) -> None:
        calls_log.clear()
        tool: Tool[Any] = Tool(name="echo", description="echo", handler=_tracking)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(calls_log) == 1
        assert calls_log[0] == {"msg": "hi"}

    async def test_result_in_context(self) -> None:
        tool = make_echo_tool()
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

        async def _a(x: int) -> str:
            calls.append("a")
            return "a"

        async def _b(y: int) -> str:
            calls.append("b")
            return "b"

        tool_a: Tool[Any] = Tool(name="a", description="a", handler=_a)
        tool_b: Tool[Any] = Tool(name="b", description="b", handler=_b)
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
        tool: Tool[Any] = Tool(name="bad", description="bad", handler=_bad)
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
        tool = make_echo_tool()
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

        # Loop should continue - we get a SessionEndEvent with end_turn
        session_ends = [e for e in events if isinstance(e, SessionEndEvent)]
        assert len(session_ends) == 1
        assert session_ends[0].stop_reason == StopReason.end_turn

    async def test_mixed_valid_and_malformed_tools(self) -> None:
        """Two parallel tool calls: one valid, one malformed. Valid runs, malformed errors."""
        tool = make_echo_tool()
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
        tool = make_echo_tool()
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
        tool: Tool[Any] = Tool(name="bad", description="bad", handler=_bad)
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
        tool = make_echo_tool()
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


class TestRefusedDispatch:
    async def test_a_refused_call_is_kept_in_history_with_a_result_saying_why(self) -> None:
        """A turn whose reason does not vouch for its calls still has to say so to the model.

        Stripped from the history instead, the next request cannot tell the attempt from a turn
        that called nothing, and the model makes the same call again.
        """
        ran: list[str] = []

        async def handler(msg: str) -> str:
            ran.append(msg)
            return msg

        tool: Tool[Any] = Tool(name="echo", description="echo", handler=handler)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.max_tokens, Usage(10, 5)),
                ]
            ]
        )
        context = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        events = [e async for e in agent.run_stream("go", context)]

        assert ran == [], "a turn that does not vouch for its calls must not run them"

        history = await context.get_history()
        calls = [b for m in history for b in m.content if isinstance(b, ToolUseBlock)]
        results = [b for m in history for b in m.content if isinstance(b, ToolResultBlock)]
        assert [b.id for b in calls] == ["c1"], "the attempted call is what the next turn reads"
        assert [b.tool_use_id for b in results] == ["c1"], "a stored call with no result is refused next"
        assert results[0].is_error
        assert "max_tokens" in str(results[0].content)

        # The call was reported as started, so a caller with no result for it waits forever.
        reported = [e for e in events if isinstance(e, ToolResult)]
        assert [(e.tool_use_id, e.is_error) for e in reported] == [("c1", True)]
        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.max_tokens]


class TestStreamingToolDispatch:
    async def test_streaming_handler_emits_keyed_output_deltas(self) -> None:
        """A tool with .stream attribute emits ToolOutputDelta events with key per field."""

        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("stdout", "line1\n")
            yield ("stderr", "warn\n")

        async def streaming_handler(msg: str) -> str:
            parts = []
            async for _, t in _stream(msg):
                parts.append(t)
            return "".join(parts)

        streaming_handler.stream = _stream  # type: ignore[attr-defined]

        tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)
        transport = StubTransport(
            [make_tool_use_response("streamer", "c1", {"msg": "hi"}), make_text_response("Done")]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 2
        assert deltas[0].key == "stdout"
        assert deltas[0].delta == "line1\n"
        assert deltas[1].key == "stderr"
        assert deltas[1].delta == "warn\n"
        assert deltas[0].name == "streamer"

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 1
        assert results[0].content == "line1\nwarn\n"
        assert not results[0].is_error

    async def test_non_streaming_tool_no_output_deltas(self) -> None:
        """A normal tool (no .stream) produces no ToolOutputDelta events."""
        tool: Tool[object] = Tool(name="echo", description="echo", handler=_tracking)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 0

    async def test_mixed_streaming_and_non_streaming(self) -> None:
        """Parallel dispatch: one streaming, one normal."""

        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("output", "s1")
            yield ("output", "s2")

        async def streaming_handler(msg: str) -> str:
            parts = []
            async for _, t in _stream(msg):
                parts.append(t)
            return "".join(parts)

        streaming_handler.stream = _stream  # type: ignore[attr-defined]

        stream_tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)
        normal_tool: Tool[object] = Tool(name="echo", description="echo", handler=_tracking)

        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "streamer"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    ToolUseStart(1, "c2", "echo"),
                    ToolInputDelta(1, "c2", json.dumps({"msg": "world"})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[stream_tool, normal_tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        deltas = [e for e in events if isinstance(e, ToolOutputDelta)]
        assert len(deltas) == 2
        assert all(d.name == "streamer" for d in deltas)

        results = [e for e in events if isinstance(e, ToolResult)]
        assert len(results) == 2


class TestDispatchThatBreaksInsteadOfFinishing:
    """A streaming dispatch talks to the reader through a queue, ended by one sentinel."""

    async def test_a_dispatch_that_raises_does_not_hang_the_turn(self) -> None:
        # Put only on the way out of a normal return, the sentinel never arrived when the dispatch
        # raised, and the reader waited on a queue nothing would fill again.
        async def _stream(msg: str) -> AsyncGenerator[tuple[str, str], None]:
            yield ("stdout", msg)

        async def streaming_handler(msg: str) -> str:
            return msg

        streaming_handler.stream = _stream  # type: ignore[attr-defined]
        tool: Tool[object] = Tool(name="streamer", description="streams", handler=streaming_handler)

        class _Broken(Agent):
            async def _dispatch_tools_streaming(
                self,
                blocks: list[ToolUseBlock],
                iteration: int,
                output_queue: asyncio.Queue[Any],
            ) -> list[ToolResultBlock]:
                raise RuntimeError("the dispatch itself broke")

        agent = _Broken(
            system="",
            tools=[tool],
            transport=StubTransport([make_tool_use_response("streamer", "c1", {"msg": "hi"})]),
        )

        events: list[StreamEvent] = []
        # The timeout is the assertion: the reader waited on a queue nothing would fill again, and
        # the turn hung with no error and no timeout of its own.
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError, match="the dispatch itself broke"):
                async for event in agent.run_stream("go", MemoryContextStore()):
                    events.append(event)

        assert [e.stop_reason for e in events if isinstance(e, SessionEndEvent)] == [StopReason.error]


class TestTheMediaNudge:
    """Gemini ends a turn holding media in about twenty tokens, so the agent asks it to go on."""

    @staticmethod
    async def _stored() -> list[Message]:
        async def picture() -> list[ImageBlock]:
            return [ImageBlock(media_type="image/png", data=b"\x89PNG")]

        tool: Tool[Any] = Tool(name="picture", description="a picture", handler=picture)

        class _Nudging(StubTransport):
            nudge_on_media_tool_result = True

        transport = _Nudging(
            [
                make_tool_use_response("picture", "c1", {}),
                make_text_response("done"),
            ]
        )
        context = MemoryContextStore()
        async for _ in Agent(system="", tools=[tool], transport=transport).run_stream("go", context):
            pass
        return await context.get_history()

    async def test_it_travels_in_the_message_the_results_are_in(self) -> None:
        # Appended as a message of its own it made two user turns in a row, which Anthropic
        # refuses outright and Gemini survives only because its converter merges them.
        history = await self._stored()

        user_turns = [m for m in history if m.role == "user"]
        results = [m for m in user_turns if any(isinstance(b, ToolResultBlock) for b in m.content)]
        assert len(results) == 1
        assert [type(b).__name__ for b in results[0].content] == ["ToolResultBlock", "TextBlock"]

    async def test_no_two_user_turns_ever_land_in_a_row(self) -> None:
        history = await self._stored()

        roles = [m.role for m in history]
        assert all(a != b or a != "user" for a, b in zip(roles, roles[1:], strict=False))
