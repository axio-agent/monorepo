"""Tests for axio_tui.plugin — entry-point-based discovery functions."""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import MagicMock, patch

from axio.permission import PermissionGuard
from axio.tool import ToolHandler

from axio_tui.plugin import discover_guards, discover_tools, discover_transports


class _EchoHandler(ToolHandler):
    """Echo back input text."""

    text: str = ""

    async def __call__(self) -> str:
        return self.text


class _NoDocHandler(ToolHandler):
    text: str = ""

    async def __call__(self) -> str:
        return self.text


class _ConcurrentHandler(ToolHandler):
    """Handler with concurrency limit."""

    _tool_concurrency: ClassVar[int | None] = 2
    text: str = ""

    async def __call__(self) -> str:
        return self.text


class _FakeGuard(PermissionGuard):
    async def check(self, handler: object) -> object:
        return handler


class _FakeTransport:
    pass


def _make_entry_point(name: str, obj: object) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = obj
    return ep


class TestDiscoverTools:
    @patch("axio_tui.plugin.entry_points")
    def test_builds_tool_from_handler(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("echo", _EchoHandler)]
        tools = discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].description == "Echo back input text."
        assert tools[0].handler is _EchoHandler
        assert tools[0].concurrency is None

    @patch("axio_tui.plugin.entry_points")
    def test_uses_class_name_when_no_docstring(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("nodoc", _NoDocHandler)]
        tools = discover_tools()
        assert tools[0].description == "_NoDocHandler"

    @patch("axio_tui.plugin.entry_points")
    def test_respects_tool_concurrency(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("concurrent", _ConcurrentHandler)]
        tools = discover_tools()
        assert tools[0].concurrency == 2

    @patch("axio_tui.plugin.entry_points")
    def test_skips_non_handler(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("bad", str)]
        tools = discover_tools()
        assert tools == []

    @patch("axio_tui.plugin.entry_points")
    def test_skips_on_load_error(self, mock_eps: MagicMock) -> None:
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("no module")
        mock_eps.return_value = [ep]
        tools = discover_tools()
        assert tools == []

    @patch("axio_tui.plugin.entry_points")
    def test_empty_entry_points(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = []
        assert discover_tools() == []


class TestDiscoverTransports:
    @patch("axio_tui.plugin.entry_points")
    def test_loads_transport_class(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("fake", _FakeTransport)]
        transports = discover_transports()
        assert transports == {"fake": _FakeTransport}

    @patch("axio_tui.plugin.entry_points")
    def test_skips_on_load_error(self, mock_eps: MagicMock) -> None:
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("no module")
        mock_eps.return_value = [ep]
        assert discover_transports() == {}

    @patch("axio_tui.plugin.entry_points")
    def test_empty_entry_points(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = []
        assert discover_transports() == {}


class TestDiscoverGuards:
    @patch("axio_tui.plugin.entry_points")
    def test_loads_guard_class(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("fake", _FakeGuard)]
        guards = discover_guards()
        assert guards == {"fake": _FakeGuard}

    @patch("axio_tui.plugin.entry_points")
    def test_skips_non_guard(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("bad", str)]
        guards = discover_guards()
        assert guards == {}

    @patch("axio_tui.plugin.entry_points")
    def test_skips_on_load_error(self, mock_eps: MagicMock) -> None:
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("no module")
        mock_eps.return_value = [ep]
        assert discover_guards() == {}

    @patch("axio_tui.plugin.entry_points")
    def test_empty_entry_points(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = []
        assert discover_guards() == {}
