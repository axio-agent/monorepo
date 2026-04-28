"""Tests for axio.testing: StubTransport and response builders."""

from __future__ import annotations

import json

import pytest

from axio.events import IterationEnd, TextDelta, ToolInputDelta, ToolUseStart
from axio.testing import (
    StubTransport,
    make_echo_tool,
    make_ephemeral_context,
    make_stub_transport,
    make_text_response,
    make_tool_use_response,
)
from axio.types import StopReason, Usage


class TestMakeTextResponse:
    def test_default_text(self) -> None:
        events = make_text_response()
        assert any(isinstance(e, TextDelta) and e.delta == "Done" for e in events)

    def test_custom_text(self) -> None:
        events = make_text_response("hello")
        assert any(isinstance(e, TextDelta) and e.delta == "hello" for e in events)

    def test_ends_with_iteration_end(self) -> None:
        events = make_text_response()
        assert isinstance(events[-1], IterationEnd)
        assert events[-1].stop_reason == StopReason.end_turn

    def test_custom_iteration(self) -> None:
        events = make_text_response(iteration=5)
        end = events[-1]
        assert isinstance(end, IterationEnd)
        assert end.iteration == 5

    def test_custom_usage(self) -> None:
        u = Usage(1, 2)
        events = make_text_response(usage=u)
        end = events[-1]
        assert isinstance(end, IterationEnd)
        assert end.usage == u


class TestMakeToolUseResponse:
    def test_default_tool_name(self) -> None:
        events = make_tool_use_response()
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert len(starts) == 1
        assert starts[0].name == "echo"

    def test_custom_tool_name(self) -> None:
        events = make_tool_use_response("my_tool")
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert starts[0].name == "my_tool"

    def test_tool_input_delta_contains_json(self) -> None:
        events = make_tool_use_response(tool_input={"x": 1})
        deltas = [e for e in events if isinstance(e, ToolInputDelta)]
        assert len(deltas) == 1
        assert json.loads(deltas[0].partial_json) == {"x": 1}

    def test_ends_with_tool_use_stop_reason(self) -> None:
        events = make_tool_use_response()
        assert isinstance(events[-1], IterationEnd)
        assert events[-1].stop_reason == StopReason.tool_use

    def test_custom_tool_id(self) -> None:
        events = make_tool_use_response(tool_id="call_abc")
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        assert starts[0].tool_use_id == "call_abc"


class TestStubTransport:
    async def test_yields_configured_events(self) -> None:
        events = make_text_response("hi")
        transport = StubTransport([events])
        received = [e async for e in transport.stream([], [], "")]
        assert received == events

    async def test_pops_next_sequence_per_call(self) -> None:
        r1 = make_text_response("first")
        r2 = make_text_response("second")
        transport = StubTransport([r1, r2])

        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]

        assert any(isinstance(e, TextDelta) and e.delta == "first" for e in first)
        assert any(isinstance(e, TextDelta) and e.delta == "second" for e in second)

    async def test_repeats_last_sequence_when_exhausted(self) -> None:
        events = make_text_response("only")
        transport = StubTransport([events])

        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]

        assert first == second

    def test_call_count_increments(self) -> None:
        transport = StubTransport([make_text_response()])
        assert transport._call_count == 0
        _ = transport.stream([], [], "")
        assert transport._call_count == 1

    async def test_empty_responses_list(self) -> None:
        transport = StubTransport([])
        with pytest.raises((IndexError, Exception)):
            _ = [e async for e in transport.stream([], [], "")]


class TestMakeStubTransport:
    async def test_yields_hello_world(self) -> None:
        transport = make_stub_transport()
        events = [e async for e in transport.stream([], [], "")]
        text = "".join(e.delta for e in events if isinstance(e, TextDelta))
        assert text == "Hello world"

    async def test_repeats_on_second_call(self) -> None:
        transport = make_stub_transport()
        first = [e async for e in transport.stream([], [], "")]
        second = [e async for e in transport.stream([], [], "")]
        assert first == second


class TestMakeEphemeralContext:
    async def test_returns_empty_context(self) -> None:
        ctx = make_ephemeral_context()
        assert await ctx.get_history() == []

    def test_each_call_returns_new_instance(self) -> None:
        a = make_ephemeral_context()
        b = make_ephemeral_context()
        assert a is not b


class TestMakeEchoTool:
    def test_name(self) -> None:
        tool = make_echo_tool()
        assert tool.name == "echo"

    async def test_returns_json_with_msg(self) -> None:
        tool = make_echo_tool()
        result = await tool(msg="hello")
        assert json.loads(result) == {"msg": "hello"}
