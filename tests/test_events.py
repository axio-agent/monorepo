"""Tests for axon.events: all stream event types."""

import pytest

from axio.events import (
    Error,
    IterationEnd,
    SessionEndEvent,
    TextDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.types import StopReason, Usage


class TestTextDeltaEvent:
    def test_frozen(self) -> None:
        e = TextDelta(index=0, delta="hi")
        with pytest.raises(AttributeError):
            e.delta = "bye"  # type: ignore[misc]

    def test_fields(self) -> None:
        e = TextDelta(index=0, delta="hi")
        assert e.index == 0
        assert e.delta == "hi"


class TestToolUseStartEvent:
    def test_frozen(self) -> None:
        e = ToolUseStart(index=0, tool_use_id="c1", name="echo")
        with pytest.raises(AttributeError):
            e.name = "other"  # type: ignore[misc]


class TestToolInputDeltaEvent:
    def test_frozen(self) -> None:
        e = ToolInputDelta(index=0, tool_use_id="c1", partial_json="{}")
        with pytest.raises(AttributeError):
            e.partial_json = "[]"  # type: ignore[misc]


class TestToolResultEvent:
    def test_frozen(self) -> None:
        e = ToolResult(tool_use_id="c1", name="echo", is_error=False)
        with pytest.raises(AttributeError):
            e.is_error = True  # type: ignore[misc]

    def test_defaults(self) -> None:
        e = ToolResult(tool_use_id="c1", name="echo", is_error=False)
        assert e.content == ""
        assert e.input == {}

    def test_with_content_and_input(self) -> None:
        e = ToolResult(
            tool_use_id="c1",
            name="shell",
            is_error=False,
            content="hello world",
            input={"command": "echo hello"},
        )
        assert e.content == "hello world"
        assert e.input == {"command": "echo hello"}

    def test_default_factory_isolation(self) -> None:
        e1 = ToolResult(tool_use_id="c1", name="a", is_error=False)
        e2 = ToolResult(tool_use_id="c2", name="b", is_error=False)
        assert e1.input is not e2.input


class TestIterationEndEvent:
    def test_frozen(self) -> None:
        e = IterationEnd(iteration=1, stop_reason=StopReason.end_turn, usage=Usage(10, 5))
        with pytest.raises(AttributeError):
            e.iteration = 2  # type: ignore[misc]


class TestErrorEvent:
    def test_frozen(self) -> None:
        e = Error(exception=RuntimeError("boom"))
        with pytest.raises(AttributeError):
            e.exception = RuntimeError("other")  # type: ignore[misc]

    def test_holds_exception(self) -> None:
        exc = ValueError("test")
        e = Error(exception=exc)
        assert e.exception is exc


class TestSessionEndEvent:
    def test_frozen(self) -> None:
        e = SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=Usage(10, 5))
        with pytest.raises(AttributeError):
            e.stop_reason = StopReason.error  # type: ignore[misc]

    def test_usage_accumulation(self) -> None:
        total = Usage(10, 5) + Usage(3, 7)
        end = SessionEndEvent(stop_reason=StopReason.end_turn, total_usage=total)
        assert end.total_usage == Usage(13, 12)


class TestStreamEventUnion:
    def test_all_seven_types(self) -> None:
        events = [
            TextDelta(0, "hi"),
            ToolUseStart(0, "c1", "echo"),
            ToolInputDelta(0, "c1", "{}"),
            ToolResult("c1", "echo", False),
            IterationEnd(1, StopReason.end_turn, Usage(0, 0)),
            Error(RuntimeError("e")),
            SessionEndEvent(StopReason.end_turn, Usage(0, 0)),
        ]
        assert len(events) == 7
