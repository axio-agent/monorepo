"""Tests for GuardSelectScreen, GuardToolsScreen, PluginHubScreen, and ToolDetailScreen."""

from __future__ import annotations

from axio.tool import Tool, ToolHandler

from axio_tui.screens import GuardSelectScreen, GuardToolsScreen, PluginHubScreen, ToolDetailScreen


class _StubHandler(ToolHandler):
    """Stub tool handler."""

    async def __call__(self) -> str:
        return ""


def _make_tools(*names: str) -> list[Tool]:
    return [Tool(name=n, description=f"Description for {n}", handler=_StubHandler) for n in names]


GUARD_NAMES = {"path": "Ask user about path access", "llm": "Agent-based safety review"}


class TestGuardToolsScreen:
    def test_initial_state(self) -> None:
        screen = GuardToolsScreen("path", "desc", True, {"shell", "read_file"}, ["shell", "read_file", "write_file"])
        assert screen._enabled is True
        assert screen._assigned_tools == {"shell", "read_file"}

    def test_toggle_enabled(self) -> None:
        screen = GuardToolsScreen("path", "desc", True, set(), ["shell"])
        assert screen._enabled is True
        screen._enabled = not screen._enabled
        assert screen._enabled is False

    def test_toggle_tool(self) -> None:
        screen = GuardToolsScreen("path", "desc", True, {"shell"}, ["shell", "read_file"])
        # Add read_file
        screen._assigned_tools.add("read_file")
        assert "read_file" in screen._assigned_tools
        # Remove shell
        screen._assigned_tools.discard("shell")
        assert "shell" not in screen._assigned_tools

    def test_format_entries_enabled(self) -> None:
        screen = GuardToolsScreen("path", "desc", True, {"shell"}, ["shell", "read_file"])
        entries = screen._format_entries()
        assert entries[0] == "[*] Enabled"
        assert entries[1] == "───"
        assert entries[2] == "[*] shell"
        assert entries[3] == "[ ] read_file"

    def test_format_entries_disabled(self) -> None:
        screen = GuardToolsScreen("path", "desc", False, set(), ["shell"])
        entries = screen._format_entries()
        assert entries[0] == "[ ] Enabled"

    def test_does_not_mutate_input_set(self) -> None:
        original: set[str] = {"shell"}
        screen = GuardToolsScreen("path", "desc", True, original, ["shell", "read_file"])
        screen._assigned_tools.add("read_file")
        assert original == {"shell"}, "Original set must not be mutated"


class TestGuardSelectScreen:
    def test_initial_state(self) -> None:
        screen = GuardSelectScreen(GUARD_NAMES, set(), {"path": {"shell"}}, ["shell"])
        assert screen._guard_order == ["path", "llm"]
        assert screen._disabled_guards == set()
        assert screen._guard_tool_map["path"] == {"shell"}

    def test_format_enabled(self) -> None:
        screen = GuardSelectScreen(GUARD_NAMES, set(), {"path": {"shell", "read_file"}}, ["shell", "read_file"])
        fmt = screen._format("path")
        assert fmt.startswith("[*]")
        assert "2 tools" in fmt

    def test_format_disabled(self) -> None:
        screen = GuardSelectScreen(GUARD_NAMES, {"llm"}, {"llm": set()}, [])
        fmt = screen._format("llm")
        assert fmt.startswith("[ ]")
        assert "disabled" in fmt

    def test_update_state_from_tools_screen(self) -> None:
        screen = GuardSelectScreen(GUARD_NAMES, {"llm"}, {"path": {"shell"}}, ["shell"])
        # Simulate what _on_guard_tools_dismissed does (without DOM refresh)
        enabled, tools = True, {"shell", "read_file"}
        if enabled:
            screen._disabled_guards.discard("path")
        else:
            screen._disabled_guards.add("path")
        screen._guard_tool_map["path"] = tools
        assert screen._guard_tool_map["path"] == {"shell", "read_file"}
        assert "path" not in screen._disabled_guards

    def test_disable_guard_state(self) -> None:
        screen = GuardSelectScreen(GUARD_NAMES, set(), {"path": {"shell"}}, ["shell"])
        screen._disabled_guards.add("path")
        assert "path" in screen._disabled_guards

    def test_does_not_mutate_input(self) -> None:
        original_disabled: set[str] = {"llm"}
        original_map: dict[str, set[str]] = {"path": {"shell"}}
        screen = GuardSelectScreen(GUARD_NAMES, original_disabled, original_map, ["shell"])
        screen._disabled_guards.add("path")
        screen._guard_tool_map["path"] = set()
        assert original_disabled == {"llm"}, "Original disabled set must not be mutated"
        assert original_map["path"] == {"shell"}, "Original map must not be mutated"


class TestPluginHubScreen:
    def test_format_entries(self) -> None:
        tools = _make_tools("shell", "read_file", "write_file")
        screen = PluginHubScreen(
            tools=tools,
            disabled_plugins={"shell"},
            guard_names=GUARD_NAMES,
            disabled_guards={"llm"},
            guard_tool_map={"path": {"shell"}, "llm": set()},
            on_plugins_changed=lambda d: None,
            on_guards_changed=lambda d, m: None,
        )
        entries = screen._format_entries()
        assert "2/3 enabled" in entries[0]
        assert "1/2 enabled" in entries[1]

    def test_plugin_callback_updates_state(self) -> None:
        tools = _make_tools("shell", "read_file")
        captured_plugins: list[set[str]] = []

        screen = PluginHubScreen(
            tools=tools,
            disabled_plugins=set(),
            guard_names=GUARD_NAMES,
            disabled_guards=set(),
            guard_tool_map={"path": {"shell"}, "llm": set()},
            on_plugins_changed=lambda d: captured_plugins.append(d),
            on_guards_changed=lambda d, m: None,
        )

        # Directly update state (avoids DOM _refresh_list)
        screen._disabled_plugins = {"shell"}
        screen._on_plugins_changed({"shell"})  # type: ignore[operator]
        assert captured_plugins == [{"shell"}]

    def test_guard_callback_updates_state(self) -> None:
        tools = _make_tools("shell", "read_file")
        captured_guards: list[tuple[set[str], dict[str, set[str]]]] = []

        screen = PluginHubScreen(
            tools=tools,
            disabled_plugins=set(),
            guard_names=GUARD_NAMES,
            disabled_guards=set(),
            guard_tool_map={"path": {"shell"}, "llm": set()},
            on_plugins_changed=lambda d: None,
            on_guards_changed=lambda d, m: captured_guards.append((d, m)),
        )

        # Directly update state
        screen._disabled_guards = {"llm"}
        screen._guard_tool_map = {"path": {"shell", "read_file"}, "llm": set()}
        screen._on_guards_changed(screen._disabled_guards, screen._guard_tool_map)  # type: ignore[operator]
        assert len(captured_guards) == 1
        assert captured_guards[0][0] == {"llm"}

    def test_none_callbacks_ignored(self) -> None:
        tools = _make_tools("shell")
        screen = PluginHubScreen(
            tools=tools,
            disabled_plugins=set(),
            guard_names=GUARD_NAMES,
            disabled_guards=set(),
            guard_tool_map={},
            on_plugins_changed=lambda d: None,
            on_guards_changed=lambda d, m: None,
        )
        # None results should be ignored (no state change)
        screen._on_tool_screen_dismissed(None)
        assert screen._disabled_plugins == set()
        screen._on_guard_screen_dismissed(None)
        assert screen._disabled_guards == set()


class TestToolDetailScreen:
    def test_initial_state(self) -> None:
        screen = ToolDetailScreen("shell", {"command": "ls"}, "file.txt", is_error=False)
        assert screen._name == "shell"
        assert screen._tool_input == {"command": "ls"}
        assert screen._content == "file.txt"
        assert screen._is_error is False

    def test_error_state(self) -> None:
        screen = ToolDetailScreen("bad_tool", {}, "boom", is_error=True)
        assert screen._is_error is True
        assert screen._content == "boom"

    def test_empty_input_and_content(self) -> None:
        screen = ToolDetailScreen("noop", {}, "", is_error=False)
        assert screen._tool_input == {}
        assert screen._content == ""
