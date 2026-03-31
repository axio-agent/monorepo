"""Modal screens for the TUI: model select, session select, plugin select, quit dialog."""

from __future__ import annotations

from typing import Any

from axio.context import SessionInfo
from axio.models import ModelSpec
from axio.tool import Tool
from rich.text import Text
from rich.tree import Tree as RichTree
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static


class ModelSelectScreen(ModalScreen[tuple[str, ModelSpec] | None]):
    """Modal screen for selecting a model from a filterable list.

    Each entry is a (transport_name, ModelSpec) tuple.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ModelSelectScreen {
        align: center middle;
    }
    #model-select {
        width: 90;
        height: 80%;
        border: heavy $accent;
        background: $panel;
        padding: 1 2;
    }
    #model-filter {
        margin-bottom: 1;
    }
    #model-list {
        height: 1fr;
    }
    """

    def __init__(self, models: list[tuple[str, ModelSpec]]) -> None:
        super().__init__()
        self._all_models = models
        self._filtered: list[tuple[str, ModelSpec]] = list(models)

    @staticmethod
    def _format(entry: tuple[str, ModelSpec]) -> str:
        name, spec = entry
        return f"[{name}] {spec.id}  (ctx:{spec.context_window:,} out:{spec.max_output_tokens:,})"

    def compose(self) -> ComposeResult:
        with Container(id="model-select"):
            yield Static("[bold]Select Model[/]")
            yield Input(placeholder="Filter models...", id="model-filter")
            yield OptionList(*[self._format(m) for m in self._filtered], id="model-list")

    def on_mount(self) -> None:
        self.query_one("#model-filter", Input).focus()

    def on_input_changed(self, message: Input.Changed) -> None:
        query = message.value.lower()
        self._filtered = [(n, m) for n, m in self._all_models if query in m.id.lower() or query in n.lower()]
        ol = self.query_one("#model-list", OptionList)
        ol.clear_options()
        for entry in self._filtered:
            ol.add_option(self._format(entry))

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if self._filtered:
            self.dismiss(self._filtered[0])

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        focused = self.focused
        ol = self.query_one("#model-list", OptionList)
        flt = self.query_one("#model-filter", Input)
        if event.key == "down" and focused is flt:
            if self._filtered:
                ol.focus()
                ol.highlighted = 0
            event.prevent_default()
        elif event.key == "up" and focused is ol and ol.highlighted == 0:
            flt.focus()
            event.prevent_default()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if 0 <= idx < len(self._filtered):
            self.dismiss(self._filtered[idx])

    def action_cancel(self) -> None:
        self.dismiss(None)


class SessionSelectScreen(ModalScreen[SessionInfo | None]):
    """Modal screen for selecting a previous session from a filterable list."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    SessionSelectScreen { align: center middle; }
    #session-select { width: 80; height: 80%; border: heavy $accent; background: $panel; padding: 1 2; }
    #session-filter { margin-bottom: 1; }
    #session-list { height: 1fr; }
    """

    def __init__(self, sessions: list[SessionInfo]) -> None:
        super().__init__()
        self._all_sessions = sessions
        self._filtered: list[SessionInfo] = list(sessions)

    @staticmethod
    def _format(s: SessionInfo) -> str:
        tokens = f"  [{s.input_tokens + s.output_tokens:,} tok]" if s.input_tokens else ""
        return f"{s.created_at}  ({s.message_count} msgs){tokens}  {s.preview}"

    def compose(self) -> ComposeResult:
        with Container(id="session-select"):
            yield Static("[bold]Restore Session[/]")
            yield Input(placeholder="Filter sessions...", id="session-filter")
            yield OptionList(*[self._format(s) for s in self._filtered], id="session-list")

    def on_mount(self) -> None:
        self.query_one("#session-filter", Input).focus()

    def on_input_changed(self, message: Input.Changed) -> None:
        query = message.value.lower()
        self._filtered = [s for s in self._all_sessions if query in s.preview.lower()]
        ol = self.query_one("#session-list", OptionList)
        ol.clear_options()
        for s in self._filtered:
            ol.add_option(self._format(s))

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if self._filtered:
            self.dismiss(self._filtered[0])

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        focused = self.focused
        ol = self.query_one("#session-list", OptionList)
        flt = self.query_one("#session-filter", Input)
        if event.key == "down" and focused is flt:
            if self._filtered:
                ol.focus()
                ol.highlighted = 0
            event.prevent_default()
        elif event.key == "up" and focused is ol and ol.highlighted == 0:
            flt.focus()
            event.prevent_default()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if 0 <= idx < len(self._filtered):
            self.dismiss(self._filtered[idx])

    def action_cancel(self) -> None:
        self.dismiss(None)


class PluginSelectScreen(ModalScreen[set[str] | None]):
    """Enable / disable discovered plugins."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    PluginSelectScreen { align: center middle; }
    #plugin-select { width: 80; height: 80%; border: heavy $accent; background: $panel; padding: 1 2; }
    #plugin-filter { margin-bottom: 1; }
    #plugin-list { height: 1fr; }
    """

    def __init__(self, tools: list[Tool], disabled: set[str]) -> None:
        super().__init__()
        self._all_tools = tools
        self._disabled = set(disabled)
        self._filtered: list[Tool] = list(tools)

    def _format(self, tool: Tool) -> str:
        mark = "[ ]" if tool.name in self._disabled else "[*]"
        return f"{mark} {tool.name:<20} — {tool.description}"

    def compose(self) -> ComposeResult:
        with Container(id="plugin-select"):
            yield Static("[bold]Manage Plugins[/]")
            yield Input(placeholder="Filter plugins...", id="plugin-filter")
            yield OptionList(*[self._format(t) for t in self._filtered], id="plugin-list")

    def on_mount(self) -> None:
        self.query_one("#plugin-filter", Input).focus()

    def _refresh_list(self) -> None:
        ol = self.query_one("#plugin-list", OptionList)
        ol.clear_options()
        for t in self._filtered:
            ol.add_option(self._format(t))

    def on_input_changed(self, message: Input.Changed) -> None:
        query = message.value.lower()
        self._filtered = [t for t in self._all_tools if query in t.name.lower()]
        self._refresh_list()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        if self._filtered:
            self._toggle(self._filtered[0])
            self._refresh_list()

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        focused = self.focused
        ol = self.query_one("#plugin-list", OptionList)
        flt = self.query_one("#plugin-filter", Input)
        if event.key == "down" and focused is flt:
            if self._filtered:
                ol.focus()
                ol.highlighted = 0
            event.prevent_default()
        elif event.key == "up" and focused is ol and ol.highlighted == 0:
            flt.focus()
            event.prevent_default()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if 0 <= idx < len(self._filtered):
            self._toggle(self._filtered[idx])
            self._refresh_list()

    def _toggle(self, tool: Tool) -> None:
        if tool.name in self._disabled:
            self._disabled.discard(tool.name)
        else:
            self._disabled.add(tool.name)

    def action_cancel(self) -> None:
        self.dismiss(self._disabled)


class PluginHubScreen(ModalScreen[None]):
    """Top-level plugin management hub with Tools and Guards subcategories."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    PluginHubScreen { align: center middle; }
    #plugin-hub { width: 70; height: auto; border: heavy $accent; background: $panel; padding: 1 2; }
    #hub-list { height: auto; }
    """

    def __init__(
        self,
        tools: list[Tool],
        disabled_plugins: set[str],
        guard_names: dict[str, str],
        disabled_guards: set[str],
        guard_tool_map: dict[str, set[str]],
        on_plugins_changed: object,
        on_guards_changed: object,
    ) -> None:
        super().__init__()
        self._tools = tools
        self._disabled_plugins = set(disabled_plugins)
        self._guard_names = guard_names
        self._disabled_guards = set(disabled_guards)
        self._guard_tool_map = {k: set(v) for k, v in guard_tool_map.items()}
        self._on_plugins_changed = on_plugins_changed
        self._on_guards_changed = on_guards_changed

    def _format_entries(self) -> list[str]:
        enabled_tools = sum(1 for t in self._tools if t.name not in self._disabled_plugins)
        total_tools = len(self._tools)
        enabled_guards = sum(1 for g in self._guard_names if g not in self._disabled_guards)
        total_guards = len(self._guard_names)
        return [
            f"Tools           {enabled_tools}/{total_tools} enabled",
            f"Guards          {enabled_guards}/{total_guards} enabled",
        ]

    def compose(self) -> ComposeResult:
        with Container(id="plugin-hub"):
            yield Static("[bold]Manage Plugins[/]")
            yield OptionList(*self._format_entries(), id="hub-list")

    def on_mount(self) -> None:
        self.query_one("#hub-list", OptionList).focus()

    def _refresh_list(self) -> None:
        ol = self.query_one("#hub-list", OptionList)
        ol.clear_options()
        for entry in self._format_entries():
            ol.add_option(entry)

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        if message.option_index == 0:
            self.app.push_screen(
                PluginSelectScreen(self._tools, self._disabled_plugins),
                self._on_tool_screen_dismissed,
            )
        elif message.option_index == 1:
            all_tool_names = [t.name for t in self._tools]
            self.app.push_screen(
                GuardSelectScreen(
                    self._guard_names,
                    self._disabled_guards,
                    self._guard_tool_map,
                    all_tool_names,
                ),
                self._on_guard_screen_dismissed,
            )

    def _on_tool_screen_dismissed(self, disabled: set[str] | None) -> None:
        if disabled is not None:
            self._disabled_plugins = disabled
            self._on_plugins_changed(disabled)  # type: ignore[operator]
            self._refresh_list()

    def _on_guard_screen_dismissed(self, result: tuple[set[str], dict[str, set[str]]] | None) -> None:
        if result is not None:
            self._disabled_guards, self._guard_tool_map = result
            self._on_guards_changed(result[0], result[1])  # type: ignore[operator]
            self._refresh_list()

    def action_cancel(self) -> None:
        self.dismiss(None)


class GuardSelectScreen(ModalScreen[tuple[set[str], dict[str, set[str]]] | None]):
    """Lists discovered guards. Enter on a guard opens GuardToolsScreen."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    GuardSelectScreen { align: center middle; }
    #guard-select { width: 80; height: 80%; border: heavy $accent; background: $panel; padding: 1 2; }
    #guard-list { height: 1fr; }
    """

    def __init__(
        self,
        guard_names: dict[str, str],
        disabled_guards: set[str],
        guard_tool_map: dict[str, set[str]],
        all_tool_names: list[str],
    ) -> None:
        super().__init__()
        self._guard_names = guard_names
        self._guard_order = list(guard_names.keys())
        self._disabled_guards = set(disabled_guards)
        self._guard_tool_map = {k: set(v) for k, v in guard_tool_map.items()}
        self._all_tool_names = all_tool_names

    def _format(self, name: str) -> str:
        desc = self._guard_names[name]
        tool_count = len(self._guard_tool_map.get(name, set()))
        if name in self._disabled_guards:
            return f"[ ] {name:<12} — {desc} (disabled)"
        return f"[*] {name:<12} — {desc} ({tool_count} tools)"

    def compose(self) -> ComposeResult:
        with Container(id="guard-select"):
            yield Static("[bold]Manage Guards[/]")
            yield OptionList(*[self._format(n) for n in self._guard_order], id="guard-list")

    def on_mount(self) -> None:
        self.query_one("#guard-list", OptionList).focus()

    def _refresh_list(self) -> None:
        ol = self.query_one("#guard-list", OptionList)
        ol.clear_options()
        for name in self._guard_order:
            ol.add_option(self._format(name))

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if 0 <= idx < len(self._guard_order):
            name = self._guard_order[idx]
            enabled = name not in self._disabled_guards
            tools = self._guard_tool_map.get(name, set())
            self.app.push_screen(
                GuardToolsScreen(
                    name,
                    self._guard_names[name],
                    enabled,
                    tools,
                    self._all_tool_names,
                ),
                lambda result, n=name: self._on_guard_tools_dismissed(n, result),
            )

    def _on_guard_tools_dismissed(self, name: str, result: tuple[bool, set[str]] | None) -> None:
        if result is not None:
            enabled, tools = result
            if enabled:
                self._disabled_guards.discard(name)
            else:
                self._disabled_guards.add(name)
            self._guard_tool_map[name] = tools
            self._refresh_list()

    def action_cancel(self) -> None:
        self.dismiss((self._disabled_guards, self._guard_tool_map))


class GuardToolsScreen(ModalScreen[tuple[bool, set[str]] | None]):
    """Per-guard config: enable/disable toggle + tool checkboxes."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    GuardToolsScreen { align: center middle; }
    #guard-tools { width: 80; height: 80%; border: heavy $accent; background: $panel; padding: 1 2; }
    #guard-tools-list { height: 1fr; }
    """

    def __init__(
        self,
        guard_name: str,
        guard_description: str,
        enabled: bool,
        assigned_tools: set[str],
        all_tool_names: list[str],
    ) -> None:
        super().__init__()
        self._guard_name = guard_name
        self._guard_description = guard_description
        self._enabled = enabled
        self._assigned_tools = set(assigned_tools)
        self._all_tool_names = all_tool_names

    def _format_entries(self) -> list[str]:
        entries: list[str] = []
        mark = "[*]" if self._enabled else "[ ]"
        entries.append(f"{mark} Enabled")
        entries.append("───")
        for name in self._all_tool_names:
            mark = "[*]" if name in self._assigned_tools else "[ ]"
            entries.append(f"{mark} {name}")
        return entries

    def compose(self) -> ComposeResult:
        with Container(id="guard-tools"):
            yield Static(f"[bold]Guard: {self._guard_name}[/] — {self._guard_description}")
            yield OptionList(*self._format_entries(), id="guard-tools-list")

    def on_mount(self) -> None:
        self.query_one("#guard-tools-list", OptionList).focus()

    def _refresh_list(self) -> None:
        ol = self.query_one("#guard-tools-list", OptionList)
        ol.clear_options()
        for entry in self._format_entries():
            ol.add_option(entry)

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        idx = message.option_index
        if idx == 0:
            self._enabled = not self._enabled
            self._refresh_list()
        elif idx == 1:
            return  # separator
        else:
            tool_idx = idx - 2
            if 0 <= tool_idx < len(self._all_tool_names):
                name = self._all_tool_names[tool_idx]
                if name in self._assigned_tools:
                    self._assigned_tools.discard(name)
                else:
                    self._assigned_tools.add(name)
                self._refresh_list()

    def action_cancel(self) -> None:
        self.dismiss((self._enabled, self._assigned_tools))


def _truncate_display(content: str, max_chars: int = 10_000) -> str:
    """Truncate large content to prevent TUI freezes during rendering."""
    if len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}\n\n[Truncated: showing {max_chars:,} of {len(content):,} chars]"


def _add_tree_nodes(parent: RichTree, data: object, max_value_len: int = 200) -> None:
    """Recursively populate a Rich Tree from a dict/list structure."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                branch = parent.add(Text(str(key), style="bold"))
                _add_tree_nodes(branch, value, max_value_len)
            else:
                val_repr = repr(value)
                if len(val_repr) > max_value_len:
                    val_repr = val_repr[:max_value_len] + "..."
                parent.add(Text.assemble((f"{key}: ", "bold"), val_repr))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                branch = parent.add(Text(str(i), style="dim"))
                _add_tree_nodes(branch, item, max_value_len)
            else:
                val_repr = repr(item)
                if len(val_repr) > max_value_len:
                    val_repr = val_repr[:max_value_len] + "..."
                parent.add(Text(val_repr))


class ToolDetailScreen(ModalScreen[None]):
    """Shows tool call input parameters and result output."""

    AUTO_FOCUS = "#tool-detail-scroll"
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "cancel", "Close", show=False),
    ]
    CSS = """
    ToolDetailScreen { align: center middle; }
    #tool-detail {
        width: 90%;
        height: 80%;
        border: heavy $accent;
        background: $panel;
        padding: 1 2;
    }
    #tool-detail-scroll { height: 1fr; }
    .tool-section-label { margin-top: 1; color: $text-muted; }
    .tool-section-content {
        margin: 0 1;
        padding: 0 1;
        border: solid $accent;
        height: auto;
    }
    """

    def __init__(self, name: str, tool_input: dict[str, Any], content: str, is_error: bool) -> None:
        super().__init__()
        self._name = name
        self._tool_input = tool_input
        self._content = content
        self._is_error = is_error

    def compose(self) -> ComposeResult:
        status_mark = "[red]✗ Error[/]" if self._is_error else "[green]✓ Success[/]"
        output_text = _truncate_display(self._content or "(empty)")
        with Container(id="tool-detail"):
            yield Static(f"[bold]Tool: {self._name}[/]    {status_mark}")
            with VerticalScroll(id="tool-detail-scroll"):
                yield Static("Input:", classes="tool-section-label")
                if self._tool_input:
                    tree = RichTree(f"[bold]{self._name}[/bold]")
                    _add_tree_nodes(tree, self._tool_input)
                    yield Static(tree, classes="tool-section-content")
                else:
                    yield Static("(none)", classes="tool-section-content")
                yield Static("Output:", classes="tool-section-label")
                yield Static(output_text, markup=False, classes="tool-section-content")

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.key in ("alt+up", "alt+down"):
            action = "action_nav_up" if event.key == "alt+up" else "action_nav_down"
            self.dismiss(None)
            nav = getattr(self.app, action, None)
            if nav is not None:
                nav()
            event.prevent_default()
            event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuitDialog(ModalScreen[bool]):
    """Confirmation dialog before quitting."""

    BINDINGS = [
        Binding("y", "confirm", "Yes", show=False),
        Binding("n", "cancel", "No", show=False),
        Binding("escape", "cancel", "No", show=False),
        Binding("left", "focus_prev_button", show=False),
        Binding("right", "focus_next_button", show=False),
    ]
    CSS = """
    QuitDialog { align: center middle; }
    #quit-dialog {
        width: 50;
        height: auto;
        border: heavy $error;
        background: $panel;
        padding: 1 2;
    }
    .guard-buttons { height: auto; margin-top: 1; }
    .guard-buttons Button { margin: 0 1; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="quit-dialog"):
            yield Static("[bold]Quit?[/]")
            yield Static("Are you sure you want to exit?")
            with Horizontal(classes="guard-buttons"):
                yield Button("Quit", id="btn-quit", variant="error")
                yield Button("Cancel", id="btn-cancel", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#btn-cancel", Button).focus()

    def _cycle_buttons(self, direction: int) -> None:
        buttons = list(self.query(Button))
        if not buttons:
            return
        try:
            idx = buttons.index(self.focused)  # type: ignore[arg-type]
        except ValueError:
            buttons[0].focus()
            return
        buttons[(idx + direction) % len(buttons)].focus()

    def action_focus_next_button(self) -> None:
        self._cycle_buttons(1)

    def action_focus_prev_button(self) -> None:
        self._cycle_buttons(-1)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-quit")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
