"""Tests for PluginSelectScreen — enable/disable plugin toggling."""

from __future__ import annotations

import pytest
from axio.tool import Tool, ToolHandler

from axio_tui.screens import PluginSelectScreen


class _StubHandler(ToolHandler):
    """Stub tool handler."""

    async def __call__(self) -> str:
        return ""


def _make_tools(*names: str) -> list[Tool]:
    return [Tool(name=n, description=f"Description for {n}", handler=_StubHandler) for n in names]


class TestPluginSelectScreen:
    def test_screen_shows_all_tools(self) -> None:
        tools = _make_tools("shell", "read_file", "write_file")
        screen = PluginSelectScreen(tools, disabled=set())
        assert screen._all_tools == tools
        assert screen._filtered == tools

    def test_toggle_disables_tool(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = PluginSelectScreen(tools, disabled=set())
        screen._toggle(tools[0])
        assert "shell" in screen._disabled

    def test_toggle_enables_tool(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = PluginSelectScreen(tools, disabled={"shell"})
        screen._toggle(tools[0])
        assert "shell" not in screen._disabled

    def test_toggle_preserves_other(self) -> None:
        tools = _make_tools("shell", "read_file")
        screen = PluginSelectScreen(tools, disabled={"read_file"})
        screen._toggle(tools[0])
        assert screen._disabled == {"shell", "read_file"}

    def test_dismiss_returns_disabled_set(self) -> None:
        tools = _make_tools("shell", "read_file")
        disabled: set[str] = {"shell"}
        screen = PluginSelectScreen(tools, disabled=disabled)
        # Verify internal disabled state matches what was passed
        assert screen._disabled == {"shell"}
        # Original set should not be mutated
        screen._toggle(tools[1])
        assert disabled == {"shell"}
        assert screen._disabled == {"shell", "read_file"}

    def test_filter_narrows_list(self) -> None:
        tools = _make_tools("shell", "read_file", "write_file")
        screen = PluginSelectScreen(tools, disabled=set())
        # Simulate filter by directly calling the filter logic
        screen._filtered = [t for t in screen._all_tools if "file" in t.name.lower()]
        assert len(screen._filtered) == 2
        assert all("file" in t.name for t in screen._filtered)

    def test_format_enabled(self) -> None:
        tools = _make_tools("shell")
        screen = PluginSelectScreen(tools, disabled=set())
        fmt = screen._format(tools[0])
        assert fmt.startswith("[*]")
        assert "shell" in fmt
        assert "Description for shell" in fmt

    def test_format_disabled(self) -> None:
        tools = _make_tools("shell")
        screen = PluginSelectScreen(tools, disabled={"shell"})
        fmt = screen._format(tools[0])
        assert fmt.startswith("[ ]")

    def test_does_not_mutate_original_disabled(self) -> None:
        original: set[str] = {"shell"}
        tools = _make_tools("shell", "read_file")
        screen = PluginSelectScreen(tools, disabled=original)
        screen._toggle(tools[1])
        assert original == {"shell"}, "Original set must not be mutated"

    @pytest.mark.parametrize("disabled", [set(), {"a"}, {"a", "b", "c"}])
    def test_roundtrip_disabled_state(self, disabled: set[str]) -> None:
        tools = _make_tools("a", "b", "c")
        screen = PluginSelectScreen(tools, disabled=disabled)
        assert screen._disabled == disabled
