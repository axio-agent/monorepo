"""Tests for ToolArgStream and Agent parse_tool_args integration."""

from __future__ import annotations

import json

from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import (
    IterationEnd,
    StreamEvent,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.testing import MsgInput, StubTransport, make_text_response
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio.types import StopReason, Usage


class TestToolArgStreamBasic:
    def test_single_string_field(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"path": "/tmp/foo.py"}')
        assert events == [
            ToolFieldStart(tool_use_id="c1", key="path"),
            ToolFieldDelta(tool_use_id="c1", key="path", text="/tmp/foo.py"),
            ToolFieldEnd(tool_use_id="c1", key="path"),
        ]

    def test_two_fields(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"path": "/a", "content": "hello"}')
        assert events == [
            ToolFieldStart(tool_use_id="c1", key="path"),
            ToolFieldDelta(tool_use_id="c1", key="path", text="/a"),
            ToolFieldEnd(tool_use_id="c1", key="path"),
            ToolFieldStart(tool_use_id="c1", key="content"),
            ToolFieldDelta(tool_use_id="c1", key="content", text="hello"),
            ToolFieldEnd(tool_use_id="c1", key="content"),
        ]

    def test_empty_object(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed("{}")
        assert events == []

    def test_current_key(self) -> None:
        s = ToolArgStream("c1")
        assert s.current_key == ""
        s.feed('{"path": "/a"}')
        assert s.current_key == "path"


class TestToolArgStreamChunked:
    def test_field_split_across_chunks(self) -> None:
        s = ToolArgStream("c1")
        e1 = s.feed('{"path":"/tmp/f')
        e2 = s.feed('oo.py"}')
        assert e1 == [
            ToolFieldStart(tool_use_id="c1", key="path"),
            ToolFieldDelta(tool_use_id="c1", key="path", text="/tmp/f"),
        ]
        assert e2 == [
            ToolFieldDelta(tool_use_id="c1", key="path", text="oo.py"),
            ToolFieldEnd(tool_use_id="c1", key="path"),
        ]

    def test_key_split_across_chunks(self) -> None:
        s = ToolArgStream("c1")
        e1 = s.feed('{"pa')
        e2 = s.feed('th": "v"}')
        assert e1 == []
        assert e2 == [
            ToolFieldStart(tool_use_id="c1", key="path"),
            ToolFieldDelta(tool_use_id="c1", key="path", text="v"),
            ToolFieldEnd(tool_use_id="c1", key="path"),
        ]

    def test_char_by_char(self) -> None:
        s = ToolArgStream("c1")
        all_events = []
        for ch in '{"k": "v"}':
            all_events.extend(s.feed(ch))
        starts = [e for e in all_events if isinstance(e, ToolFieldStart)]
        deltas = [e for e in all_events if isinstance(e, ToolFieldDelta)]
        ends = [e for e in all_events if isinstance(e, ToolFieldEnd)]
        assert len(starts) == 1
        assert starts[0].key == "k"
        assert "".join(d.text for d in deltas) == "v"
        assert len(ends) == 1
        assert ends[0].key == "k"


class TestToolArgStreamEscapes:
    def test_string_escapes_decoded(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"msg": "a\\nb\\tc"}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "a\nb\tc"

    def test_escaped_quote(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"msg": "say \\"hi\\""}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == 'say "hi"'

    def test_escaped_backslash(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"msg": "a\\\\b"}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "a\\b"

    def test_unicode_escape(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"msg": "\\u0041"}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "A"

    def test_surrogate_pair(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"msg": "\\uD83D\\uDE00"}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "\U0001f600"


class TestToolArgStreamNonStringValues:
    def test_number(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"count": 42}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "42"

    def test_negative_float(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"val": -3.14}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "-3.14"

    def test_boolean(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"flag": true}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "true"

    def test_null(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"val": null}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "null"

    def test_nested_object_as_raw_json(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"meta": {"a": 1}}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == '{"a": 1}'

    def test_array_as_raw_json(self) -> None:
        s = ToolArgStream("c1")
        events = s.feed('{"items": [1, 2, 3]}')
        deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        text = "".join(d.text for d in deltas)
        assert text == "[1, 2, 3]"


class TestAgentParseToolArgs:
    async def test_produces_field_events(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", '{"msg":'),
                    ToolInputDelta(0, "c1", '"hello"}'),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, parse_tool_args=True)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        field_starts = [e for e in events if isinstance(e, ToolFieldStart)]
        field_deltas = [e for e in events if isinstance(e, ToolFieldDelta)]
        field_ends = [e for e in events if isinstance(e, ToolFieldEnd)]
        input_deltas = [e for e in events if isinstance(e, ToolInputDelta)]

        assert len(field_starts) == 1
        assert field_starts[0].key == "msg"
        assert "".join(d.text for d in field_deltas) == "hello"
        assert len(field_ends) == 1
        assert input_deltas == []

    async def test_default_no_field_events(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        field_starts = [e for e in events if isinstance(e, ToolFieldStart)]
        input_deltas = [e for e in events if isinstance(e, ToolInputDelta)]

        assert field_starts == []
        assert len(input_deltas) == 1

    async def test_tool_still_dispatched(self) -> None:
        tool = Tool(name="echo", description="echo", handler=MsgInput)
        transport = StubTransport(
            [
                [
                    ToolUseStart(0, "c1", "echo"),
                    ToolInputDelta(0, "c1", json.dumps({"msg": "hi"})),
                    IterationEnd(1, StopReason.tool_use, Usage(10, 5)),
                ],
                make_text_response("Done"),
            ]
        )
        agent = Agent(system="test", tools=[tool], transport=transport, parse_tool_args=True)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert not tool_results[0].is_error
