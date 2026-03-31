"""Tests for axon.exceptions: hierarchy verification."""

from __future__ import annotations

from axio.exceptions import AxioError, GuardError, HandlerError, StreamError, ToolError


class TestHierarchy:
    def test_tool_error_is_axon_error(self) -> None:
        assert issubclass(ToolError, AxioError)

    def test_guard_error_is_tool_error(self) -> None:
        assert issubclass(GuardError, ToolError)

    def test_handler_error_is_tool_error(self) -> None:
        assert issubclass(HandlerError, ToolError)

    def test_stream_error_is_axon_error(self) -> None:
        assert issubclass(StreamError, AxioError)

    def test_stream_error_not_tool_error(self) -> None:
        assert not issubclass(StreamError, ToolError)


class TestInstantiation:
    def test_guard_error_message(self) -> None:
        exc = GuardError("denied")
        assert str(exc) == "denied"

    def test_handler_error_message(self) -> None:
        exc = HandlerError("boom")
        assert str(exc) == "boom"

    def test_stream_error_message(self) -> None:
        exc = StreamError("no data")
        assert str(exc) == "no data"

    def test_catch_axon_error_catches_guard(self) -> None:
        try:
            raise GuardError("test")
        except AxioError:
            pass

    def test_catch_tool_error_catches_handler(self) -> None:
        try:
            raise HandlerError("test")
        except ToolError:
            pass
