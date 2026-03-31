"""TUI settings screens for MCP server management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import Button, Input, OptionList, Select, Static

    _HAS_TEXTUAL = True
except ImportError:
    _HAS_TEXTUAL = False

if TYPE_CHECKING:
    from .config import MCPServerConfig
    from .registry import MCPRegistry

_DELETE_SENTINEL = "__DELETE__"


if _HAS_TEXTUAL:

    class MCPServerEditScreen(ModalScreen["MCPServerConfig | None"]):
        """Per-server config: name, type, command/url, args, headers."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]
        CSS = """
        MCPServerEditScreen { align: center middle; }
        #mcp-edit {
            width: 80;
            height: auto;
            max-height: 90%;
            border: heavy $accent;
            background: $panel;
            padding: 1 2;
        }
        #mcp-edit Input { margin-bottom: 1; }
        #mcp-edit Select { margin-bottom: 1; }
        .mcp-buttons { height: auto; margin-top: 1; }
        .mcp-buttons Button { margin: 0 1; }
        """

        def __init__(self, config: MCPServerConfig | None = None, scope: str = "global") -> None:
            super().__init__()
            self._editing = config
            self._scope = scope

        def compose(self) -> ComposeResult:
            title = "Edit MCP Server" if self._editing else "Add MCP Server"
            with Container(id="mcp-edit"):
                yield Static(f"[bold]{title}[/]")
                yield Static("Name:")
                yield Input(
                    value=self._editing.name if self._editing else "",
                    placeholder="server name (e.g. filesystem)",
                    id="mcp-name",
                )
                yield Static("Type:")
                is_http = self._editing is not None and self._editing.url is not None
                yield Select(
                    [("stdio (subprocess)", "stdio"), ("HTTP", "http")],
                    value="http" if is_http else "stdio",
                    id="mcp-type",
                )
                yield Static("Command (stdio):")
                yield Input(
                    value=self._editing.command or "" if self._editing else "",
                    placeholder="e.g. npx, uvx, python",
                    id="mcp-command",
                )
                yield Static("Args (comma-separated):")
                yield Input(
                    value=", ".join(self._editing.args) if self._editing and self._editing.args else "",
                    placeholder="e.g. -m, mcp_server",
                    id="mcp-args",
                )
                yield Static("URL (HTTP):")
                yield Input(
                    value=self._editing.url or "" if self._editing else "",
                    placeholder="e.g. http://localhost:8000/mcp",
                    id="mcp-url",
                )
                yield Static("Headers (key:value, comma-separated):")
                yield Input(
                    value=_headers_to_str(self._editing.headers) if self._editing else "",
                    placeholder="e.g. Authorization:Bearer xxx",
                    id="mcp-headers",
                )
                yield Static("Scope:")
                yield Select(
                    [("Global", "global"), ("Project", "project")],
                    value=self._scope,
                    id="mcp-scope",
                )
                with Horizontal(classes="mcp-buttons"):
                    yield Button("Save", id="btn-save", variant="primary")
                    if self._editing:
                        yield Button("Delete", id="btn-delete", variant="error")
                    yield Button("Cancel", id="btn-cancel")

        def on_mount(self) -> None:
            self.query_one("#mcp-name", Input).focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-cancel":
                self.dismiss(None)
            elif event.button.id == "btn-delete":
                self.dismiss(_make_delete_sentinel())  # type: ignore[arg-type]
            elif event.button.id == "btn-save":
                self._save()

        def _save(self) -> None:
            from .config import MCPServerConfig

            name = self.query_one("#mcp-name", Input).value.strip()
            if not name:
                self.notify("Name is required", severity="error")
                return

            transport_type = self.query_one("#mcp-type", Select).value

            command: str | None = None
            url: str | None = None
            args: list[str] = []
            headers: dict[str, str] = {}

            if transport_type == "stdio":
                command = self.query_one("#mcp-command", Input).value.strip()
                if not command:
                    self.notify("Command is required for stdio transport", severity="error")
                    return
                raw_args = self.query_one("#mcp-args", Input).value.strip()
                if raw_args:
                    args = [a.strip() for a in raw_args.split(",") if a.strip()]
            else:
                url = self.query_one("#mcp-url", Input).value.strip()
                if not url:
                    self.notify("URL is required for HTTP transport", severity="error")
                    return

            raw_headers = self.query_one("#mcp-headers", Input).value.strip()
            if raw_headers:
                headers = _parse_headers(raw_headers)

            scope = str(self.query_one("#mcp-scope", Select).value)

            try:
                config = MCPServerConfig(
                    name=name,
                    command=command,
                    url=url,
                    args=args,
                    headers=headers,
                    scope=scope,
                )
            except ValueError as exc:
                self.notify(str(exc), severity="error")
                return

            self.dismiss(config)

        def action_cancel(self) -> None:
            self.dismiss(None)

    class MCPHubScreen(ModalScreen[None]):
        """Command palette entry point: lists configured MCP server instances."""

        BINDINGS = [Binding("escape", "cancel", "Cancel")]
        CSS = """
        MCPHubScreen { align: center middle; }
        #mcp-hub {
            width: 80;
            height: 80%;
            border: heavy $accent;
            background: $panel;
            padding: 1 2;
        }
        #mcp-list { height: 1fr; }
        """

        def __init__(
            self,
            registry: MCPRegistry,
            config: Any = None,
            global_config: Any = None,
        ) -> None:
            super().__init__()
            self._registry = registry
            self._config = config
            self._global_config = global_config

        def _format_entries(self) -> list[str]:
            entries = ["+ Add MCP Server"]
            for name in self._registry.server_names:
                status = self._registry.server_status(name)
                count = self._registry.server_tool_count(name)
                config = self._registry.server_config(name)
                endpoint = config.command or config.url or ""
                scope_db = self._registry.get_server_scope(name)
                badge = "[G]" if scope_db is not self._config else "[P]"
                if status == "connected":
                    entries.append(f"{badge} [*] {name:<20} ({count} tools)  {endpoint}")
                elif status == "error":
                    entries.append(f"{badge} [!] {name:<20} (error)    {endpoint}")
                else:
                    entries.append(f"{badge} [-] {name:<20} (disconnected)  {endpoint}")
            return entries

        def compose(self) -> ComposeResult:
            with Container(id="mcp-hub"):
                yield Static("[bold]Manage MCP Servers[/]")
                yield OptionList(*self._format_entries(), id="mcp-list")

        def on_mount(self) -> None:
            self.query_one("#mcp-list", OptionList).focus()

        def _refresh_list(self) -> None:
            ol = self.query_one("#mcp-list", OptionList)
            ol.clear_options()
            for entry in self._format_entries():
                ol.add_option(entry)

        def _scope_label(self, name: str) -> str:
            scope_db = self._registry.get_server_scope(name)
            return "project" if scope_db is self._config else "global"

        def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
            idx = message.option_index
            if idx == 0:
                self.app.push_screen(MCPServerEditScreen(), self._on_add_dismissed)
            else:
                server_idx = idx - 1
                names = self._registry.server_names
                if 0 <= server_idx < len(names):
                    name = names[server_idx]
                    config = self._registry.server_config(name)
                    scope = self._scope_label(name)
                    self.app.push_screen(
                        MCPServerEditScreen(config, scope=scope),
                        lambda result, n=name: self._on_edit_dismissed(n, result),
                    )

        def _on_add_dismissed(self, result: Any) -> None:
            if result is None:
                return
            self.app.run_worker(self._do_add(result))

        def _scope_db(self, scope: str) -> Any:
            return self._config if scope == "project" else self._global_config

        async def _do_add(self, config: MCPServerConfig) -> None:
            await self._registry.add_server(config, scope=self._scope_db(config.scope))
            self._refresh_list()

        def _on_edit_dismissed(self, name: str, result: Any) -> None:
            if result is None:
                return
            if _is_delete_sentinel(result):
                self.app.run_worker(self._do_remove(name))
            else:
                self.app.run_worker(self._do_update(name, result))

        async def _do_remove(self, name: str) -> None:
            await self._registry.remove_server(name)
            self._refresh_list()

        async def _do_update(self, name: str, config: MCPServerConfig) -> None:
            await self._registry.update_server(name, config, scope=self._scope_db(config.scope))
            self._refresh_list()

        def action_cancel(self) -> None:
            self.dismiss(None)

else:

    class MCPHubScreen:  # type: ignore[no-redef]
        """Stub when textual is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("textual is required for MCP settings screens")

    class MCPServerEditScreen:  # type: ignore[no-redef]
        """Stub when textual is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("textual is required for MCP settings screens")


def _headers_to_str(headers: dict[str, str]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in headers.items())


def _parse_headers(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            key, value = pair.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class _DeleteSentinel:
    """Sentinel object to signal a delete action from the edit screen."""


def _make_delete_sentinel() -> _DeleteSentinel:
    return _DeleteSentinel()


def _is_delete_sentinel(obj: object) -> bool:
    return isinstance(obj, _DeleteSentinel)
