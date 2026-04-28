"""Tests for axio_tui.plugin - entry-point-based discovery functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from axio.permission import PermissionGuard
from axio.tool import Tool

from axio_tui.plugin import discover_guards, discover_tools, discover_transports


async def _echo_handler(text: str = "") -> str:
    """Echo back input text."""
    return text


async def _no_doc_handler(text: str = "") -> str:
    return text


async def _concurrent_handler(text: str = "") -> str:
    """Handler with concurrency limit."""
    return text


_concurrent_handler._tool_concurrency = 2  # type: ignore[attr-defined]


class _FakeGuard(PermissionGuard):
    async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
        return kwargs


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
        mock_eps.return_value = [_make_entry_point("echo", _echo_handler)]
        tools = discover_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert tools[0].description == "Echo back input text."
        assert tools[0].handler is _echo_handler
        assert tools[0].concurrency is None

    @patch("axio_tui.plugin.entry_points")
    def test_uses_empty_string_when_no_docstring(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("nodoc", _no_doc_handler)]
        tools = discover_tools()
        assert tools[0].description == ""

    @patch("axio_tui.plugin.entry_points")
    def test_respects_tool_concurrency(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("concurrent", _concurrent_handler)]
        tools = discover_tools()
        assert tools[0].concurrency == 2

    @patch("axio_tui.plugin.entry_points")
    def test_skips_non_callable(self, mock_eps: MagicMock) -> None:
        mock_eps.return_value = [_make_entry_point("bad", 42)]
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
