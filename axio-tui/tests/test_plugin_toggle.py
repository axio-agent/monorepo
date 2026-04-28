"""Tests for ToolSelectScreen - enable/disable tool toggling."""

from __future__ import annotations

from typing import Any

import pytest
from axio.tool import Tool

from axio_tui.screens import ToolSelectScreen


async def _stub_handler() -> str:
    """Stub tool handler."""
    return ""


def _make_tools(*names: str) -> list[Tool[Any]]:
    return [Tool(name=n, description=f"Description for {n}", handler=_stub_handler) for n in names]


def _screen(tools: list[Tool[Any]], disabled: set[str]) -> ToolSelectScreen:
    return ToolSelectScreen(groups={"pkg": tools}, disabled=disabled)


class TestToolSelectScreen:
    def test_screen_shows_all_tools(self) -> None:
        tools = _make_tools("shell", "read_file", "write_file")
        screen = _screen(tools, disabled=set())
        assert screen._all_tools == tools

    def test_toggle_disables_tool(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = _screen(tools, disabled=set())
        screen._select(tools[0])
        assert "shell" in screen._disabled

    def test_toggle_enables_tool(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = _screen(tools, disabled={"shell"})
        screen._select(tools[0])
        assert "shell" not in screen._disabled

    def test_toggle_preserves_other(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = _screen(tools, disabled={"read_file"})
        screen._select(tools[0])
        assert screen._disabled == {"shell", "read_file"}

    def test_dismiss_returns_disabled_set(self) -> None:
        tools = _make_tools("shell", "read_file")
        disabled: set[str] = {"shell"}
        screen = _screen(tools, disabled=disabled)
        assert screen._disabled == {"shell"}
        # Original set must not be mutated
        screen._select(tools[1])
        assert disabled == {"shell"}
        assert screen._disabled == {"shell", "read_file"}

    def test_filter_narrows_items(self) -> None:
        tools = _make_tools("shell", "read_file", "write_file")
        screen = _screen(tools, disabled=set())
        screen._filter_query = "file"
        screen._rebuild_items()
        assert all(isinstance(i, Tool) and "file" in i.name for i in screen._items)
        assert len(screen._items) == 2

    def test_format_enabled(self) -> None:
        tools = _make_tools("shell")
        screen = _screen(tools, disabled=set())
        fmt = screen._format(tools[0])
        assert "[*]" in fmt
        assert "shell" in fmt
        assert "Description for shell" in fmt

    def test_format_disabled(self) -> None:
        tools = _make_tools("shell")
        screen = _screen(tools, disabled={"shell"})
        fmt = screen._format(tools[0])
        assert "[ ]" in fmt

    def test_does_not_mutate_original_disabled(self) -> None:
        original: set[str] = {"shell"}
        tools = _make_tools("shell", "read_file")
        screen = _screen(tools, disabled=original)
        screen._select(tools[1])
        assert original == {"shell"}, "Original set must not be mutated"

    @pytest.mark.parametrize("disabled", [set(), {"a"}, {"a", "b", "c"}])
    def test_roundtrip_disabled_state(self, disabled: set[str]) -> None:
        tools = _make_tools("a", "b", "c")
        screen = _screen(tools, disabled=disabled)
        assert screen._disabled == disabled

    def test_group_header_bulk_toggle_on(self) -> None:
        tools = _make_tools("a", "b")
        screen = _screen(tools, disabled=set())
        # All on → bulk toggle disables all
        screen._select("pkg")
        assert screen._disabled == {"a", "b"}

    def test_group_header_bulk_toggle_off(self) -> None:
        tools = _make_tools("a", "b")
        screen = _screen(tools, disabled={"a", "b"})
        # Not all on → bulk toggle enables all
        screen._select("pkg")
        assert screen._disabled == set()

    def test_group_header_partial_enables_all(self) -> None:
        tools = _make_tools("a", "b")
        screen = _screen(tools, disabled={"a"})
        # Partial → bulk toggle enables all
        screen._select("pkg")
        assert screen._disabled == set()
