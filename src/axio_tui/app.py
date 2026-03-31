"""AgentApp: Textual TUI for the Axio coding-assist agent."""

import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiohttp
from axio.agent import Agent
from axio.context import ContextStore, MemoryContextStore, SessionInfo, compact_context
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    TextDelta,
    ToolResult,
    ToolUseStart,
)
from axio.messages import Message
from axio.models import Capability, ModelSpec
from axio.permission import PermissionGuard
from axio.selector import EmbeddingToolSelector, ToolFilteringTransport
from axio.tool import Tool
from pygments.token import Token  # type: ignore[import-untyped]
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.content import Content
from textual.events import Key
from textual.highlight import HighlightTheme
from textual.highlight import highlight as hl_highlight
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Markdown, OptionList, RichLog, Static
from textual.widgets._markdown import MarkdownFence

from axio_tui.plugin import ToolsPlugin, discover_guards, discover_tools, discover_tools_plugins
from axio_tui.transport_registry import RoleBinding, TransportRegistry

from .prompt import SYSTEM_PROMPT
from .screens import ModelSelectScreen, PluginHubScreen, QuitDialog, SessionSelectScreen, ToolDetailScreen
from .sqlite_context import GLOBAL_PROJECT, ProjectConfig, SQLiteContextStore
from .tools import SUBAGENT_SYSTEM_PROMPT, Confirm, StatusLine, SubAgent, VisionAnalyze

logger = logging.getLogger(__name__)


class _MonokaiTheme(HighlightTheme):
    """Monokai-inspired syntax highlighting for code blocks."""

    STYLES: dict[tuple[str, ...], str] = {
        Token.Comment: "#75715e italic",
        Token.Error: "#f92672 on #49483e",
        Token.Generic.Strong: "bold",
        Token.Generic.Emph: "italic",
        Token.Generic.Error: "#f92672",
        Token.Generic.Heading: "#e6db74 bold",
        Token.Generic.Subheading: "#e6db74",
        Token.Keyword: "#f92672",
        Token.Keyword.Constant: "#66d9ef italic",
        Token.Keyword.Namespace: "#f92672",
        Token.Keyword.Type: "#66d9ef italic",
        Token.Literal.Number: "#ae81ff",
        Token.Literal.String: "#e6db74",
        Token.Literal.String.Backtick: "#e6db74",
        Token.Literal.String.Doc: "#e6db74 italic",
        Token.Literal.String.Double: "#e6db74",
        Token.Name: "#f8f8f2",
        Token.Name.Attribute: "#a6e22e",
        Token.Name.Builtin: "#66d9ef",
        Token.Name.Builtin.Pseudo: "#66d9ef italic",
        Token.Name.Class: "#a6e22e bold",
        Token.Name.Constant: "#66d9ef",
        Token.Name.Decorator: "#a6e22e bold",
        Token.Name.Function: "#a6e22e",
        Token.Name.Function.Magic: "#a6e22e",
        Token.Name.Tag: "#f92672",
        Token.Name.Variable: "#f8f8f2",
        Token.Number: "#ae81ff",
        Token.Operator: "#f92672",
        Token.Operator.Word: "#f92672",
        Token.String: "#e6db74",
        Token.Whitespace: "",
    }


class _MonokaiFence(MarkdownFence):
    """MarkdownFence with Monokai highlighting."""

    @classmethod
    def highlight(cls, code: str, language: str) -> Content:
        return hl_highlight(code, language=language or None, theme=_MonokaiTheme)


class _Markdown(Markdown):
    """Markdown widget with Monokai code-block highlighting."""

    BLOCKS = {**Markdown.BLOCKS, "fence": _MonokaiFence, "code_block": _MonokaiFence}


@dataclass
class _ToolCallInfo:
    name: str = ""
    status: bool | None = None  # None=pending, True=ok, False=error
    input: dict[str, Any] = field(default_factory=dict)
    content: str = ""


class _ToolStatusWidget(Static):
    """Single-line live-updating tool status tracker with clickable completed tools."""

    DEFAULT_CSS = """
    _ToolStatusWidget {
        height: auto;
        margin: 0;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("", classes="meta")
        self._tools: dict[str, _ToolCallInfo] = {}  # keyed by tool_use_id

    def track(self, tool_use_id: str, name: str) -> None:
        self._tools[tool_use_id] = _ToolCallInfo(name=name)
        self._refresh_content()

    def complete(
        self, tool_use_id: str, *, is_error: bool, content: str = "", tool_input: dict[str, Any] | None = None
    ) -> None:
        if tool_use_id in self._tools:
            info = self._tools[tool_use_id]
            info.status = not is_error
            info.content = content
            info.input = tool_input or {}
        self._refresh_content()

    def _refresh_content(self) -> None:
        parts: list[str] = []
        for tid, info in self._tools.items():
            if info.status is None:
                parts.append(f"[yellow]{info.name} ⏳[/]")
            elif info.status:
                parts.append(f"[@click=show_detail('{tid}')][green]{info.name} ✓[/][/]")
            else:
                parts.append(f"[@click=show_detail('{tid}')][red]{info.name} ✗[/][/]")
        self.update("[dim]tools:[/] " + "  ".join(parts))

    def action_show_detail(self, tool_use_id: str) -> None:
        info = self._tools.get(tool_use_id)
        if info is None:
            return
        is_error = info.status is not None and not info.status
        self.app.push_screen(ToolDetailScreen(info.name, info.input, info.content, is_error))


class ModelRole(StrEnum):
    CHAT = "chat"
    COMPACT = "compact"
    SUBAGENT = "subagent"
    GUARD = "guard"
    VISION = "vision"
    EMBEDDING = "embedding"
    REASONING = "reasoning"


class ModelRoleSelectScreen(ModalScreen[ModelRole | None]):
    """Modal screen for selecting which model role to configure."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    CSS = """
    ModelRoleSelectScreen { align: center middle; }
    #role-select {
        width: 70;
        height: auto;
        border: heavy $accent;
        background: $panel;
        padding: 1 2;
    }
    #role-list { height: auto; }
    """

    def __init__(self, role_bindings: dict[ModelRole, RoleBinding]) -> None:
        super().__init__()
        self._role_bindings = role_bindings

    def compose(self) -> ComposeResult:
        with Container(id="role-select"):
            yield Static("[bold]Select Model Role[/]")
            options: list[str] = []
            for role in ModelRole:
                label = _ROLE_LABELS[role]
                if role in self._role_bindings:
                    b = self._role_bindings[role]
                    options.append(f"{label:<12} [{b.transport}] {b.model.id}")
                else:
                    options.append(f"{label:<12} (not configured)")
            yield OptionList(*options, id="role-list")

    def on_mount(self) -> None:
        self.query_one("#role-list", OptionList).focus()

    def on_option_list_option_selected(self, message: OptionList.OptionSelected) -> None:
        roles = list(ModelRole)
        idx = message.option_index
        if 0 <= idx < len(roles):
            self.dismiss(roles[idx])

    def action_cancel(self) -> None:
        self.dismiss(None)


_ROLE_LABELS: dict[ModelRole, str] = {
    ModelRole.CHAT: "Chat",
    ModelRole.COMPACT: "Compaction",
    ModelRole.SUBAGENT: "Subagent",
    ModelRole.GUARD: "Guard",
    ModelRole.VISION: "Vision",
    ModelRole.EMBEDDING: "Embedding",
    ModelRole.REASONING: "Reasoning",
}

DB_PATH = Path.home() / ".local" / "share" / "axio-tui.db"


class _AxioLogFilter(logging.Filter):
    """Only pass log records from axio-related loggers.

    Third-party loggers (aiosqlite, markdown_it, asyncio, …) produce
    thousands of DEBUG messages that flood the RichLog widget and
    starve the event loop when routed through _RichLogHandler.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return name.startswith("axio") or name == "root"


class _RichLogHandler(logging.Handler):
    """Routes log records to a Textual RichLog widget."""

    def __init__(self, widget: RichLog) -> None:
        super().__init__()
        self._widget = widget
        self.addFilter(_AxioLogFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._widget.write(self.format(record))
        except Exception:
            self.handleError(record)


class AgentApp(App[None]):
    """Axio Agent TUI."""

    TITLE = "Axio Agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("escape", "stop_agent", "Stop", show=False),
        Binding("f12", "toggle_dev_console", "DevConsole", show=False),
        Binding("alt+up", "nav_up", "Prev Msg"),
        Binding("alt+down", "nav_down", "Next Msg"),
    ]
    CSS = """
    #log {
        height: 1fr;
    }
    #log > .nav-selected {
        background: $accent 10%;
        border-left: thick $accent;
    }
    #log > Markdown {
        margin: 0;
        padding: 0 1;
    }
    #log > .meta {
        height: auto;
        margin: 0;
        padding: 0 1;
    }
    #log > .user-msg {
        height: auto;
        margin: 1 0 0 0;
        padding: 0 1;
        background: $surface;
        border-left: thick $accent;
    }
    #dev-console {
        height: 1fr;
        border-top: solid $accent;
        display: none;
    }
    #dev-console.visible {
        display: block;
    }
    #tui-status {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
        display: none;
    }
    #tui-status.visible {
        display: block;
    }
    #status {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._session: aiohttp.ClientSession | None = None
        self._agent: Agent | None = None
        self._transports = TransportRegistry()
        self._role_bindings: dict[ModelRole, RoleBinding] = {}
        self._chat_context = SQLiteContextStore(DB_PATH, uuid4().hex)
        self._config = ProjectConfig(DB_PATH)
        self._global_config = ProjectConfig(DB_PATH, project=GLOBAL_PROJECT)
        self._is_running = False
        self._worker: Any = None
        self._pending_input: asyncio.Queue[tuple[str, Static]] = asyncio.Queue()
        self._guard_lock = asyncio.Lock()
        self._current_md: Markdown | None = None
        self._response_text = ""
        self._text_dirty = False
        self._last_input_tokens = 0
        self._chat_model: ModelSpec | None = None
        self._model_name = ""
        self._all_tools: list[Tool] = []
        self._disabled_plugins: set[str] = set()
        self._disabled_guards: set[str] = set()
        self._guard_tool_map: dict[str, set[str]] = {}
        self._guard_instances: dict[str, PermissionGuard] = {}
        self._guard_classes: dict[str, type[PermissionGuard]] = {}
        self._active_tools: list[Tool] = []
        self._chat_transport: Any = None
        self._tool_selector: EmbeddingToolSelector | None = None
        self._embed_cache: Any = None
        self._log_handler: _RichLogHandler | None = None
        self._tool_status: _ToolStatusWidget | None = None
        self._agent_emoji = ""
        self._tools_plugins: dict[str, ToolsPlugin] = {}
        self._nav_index: int | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        log = VerticalScroll(id="log")
        log.can_focus = False
        yield log
        yield RichLog(highlight=True, markup=True, id="dev-console")
        yield Static("", id="tui-status")
        yield Static("", id="status")
        yield Input(placeholder="Enter a coding task...", id="input")
        yield Footer()

    async def on_mount(self) -> None:
        console = self.query_one("#dev-console", RichLog)
        handler = _RichLogHandler(console)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s[%(name)s]: %(message)s"))
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

        StatusLine._callback = self._on_agent_status
        SubAgent._factory = self._make_subagent

        self._session = aiohttp.ClientSession()
        await self._transports.init(self._session, config=self._config, global_config=self._global_config)

        if not self._transports.available:
            await self._write_meta("[red]No transports available — set an API key env variable[/]")
            return

        # Resolve model for each role: persisted config -> transport defaults -> skip
        for role in ModelRole:
            config_value = await self._config.get(f"model.{role}")
            binding: RoleBinding | None = None
            if config_value:
                binding = self._transports.resolve(config_value)
            if binding is None:
                binding = self._transports.resolve_default(role.value)
            if binding is not None:
                self._role_bindings[role] = binding

        # Set up chat transport
        if ModelRole.CHAT in self._role_bindings:
            chat_binding = self._role_bindings[ModelRole.CHAT]
            chat_transport = self._transports.make_transport(chat_binding.transport, chat_binding.model)
            self._chat_model = chat_binding.model
        else:
            # Fallback: first available transport with its default model
            first = self._transports.available[0]
            t = self._transports.get_transport(first)
            chat_transport = self._transports.make_transport(first, t.model)
            self._chat_model = t.model

        self._model_name = self._chat_model.id
        self.sub_title = self._model_name

        if ModelRole.VISION in self._role_bindings:
            vb = self._role_bindings[ModelRole.VISION]
            VisionAnalyze._transport = self._transports.make_transport(vb.transport, vb.model)

        self._chat_transport = chat_transport
        if ModelRole.EMBEDDING in self._role_bindings:
            eb = self._role_bindings[ModelRole.EMBEDDING]
            raw_transport = self._transports.make_transport(eb.transport, eb.model)
            try:
                from axio_tui_rag.embedding_cache import CachedEmbeddingTransport

                embed_cache = CachedEmbeddingTransport(raw_transport, DB_PATH, eb.model.id)
                self._embed_cache = embed_cache
                self._tool_selector = EmbeddingToolSelector(
                    transport=embed_cache,
                    top_k=6,
                    pinned=frozenset({"status_line"}),
                )
                chat_transport = ToolFilteringTransport(chat_transport, self._tool_selector)
            except ImportError:
                pass

        # Discover tools and guards via entry points
        tools: list[Tool] = discover_tools()
        self._guard_classes = discover_guards()

        # Load disabled plugins from config
        raw = await self._config.get("plugins.disabled")
        if raw:
            self._disabled_plugins = {n.strip() for n in raw.split(",") if n.strip()}

        # Load disabled guards from config
        raw = await self._config.get("guards.disabled")
        if raw:
            self._disabled_guards = {n.strip() for n in raw.split(",") if n.strip()}

        # Load guard-tool assignments from config (seed defaults for path guard)
        _DEFAULT_PATH_TOOLS = frozenset(
            {"shell", "run_python", "write_file", "read_file", "list_files", "patch_file", "vision", "index_files"}
        )
        for guard_name in self._guard_classes:
            raw = await self._config.get(f"guards.{guard_name}.tools")
            if raw is not None:
                self._guard_tool_map[guard_name] = {n.strip() for n in raw.split(",") if n.strip()}
            elif guard_name == "path":
                self._guard_tool_map[guard_name] = set(_DEFAULT_PATH_TOOLS)
            else:
                self._guard_tool_map[guard_name] = set()

        # Instantiate enabled guards
        for guard_name, cls in self._guard_classes.items():
            if guard_name in self._disabled_guards:
                continue
            instance = self._instantiate_guard(guard_name, cls)
            if instance is not None:
                self._guard_instances[guard_name] = instance

        # Load tools from plugins (e.g. MCP servers)
        self._tools_plugins = discover_tools_plugins()
        for plugin in self._tools_plugins.values():
            try:
                await plugin.init(config=self._config, global_config=self._global_config)
                tools.extend(plugin.all_tools)
            except Exception:
                logger.warning("Tools plugin init failed", exc_info=True)

        self._all_tools = tools
        self._rebuild_agent_tools()

        self._agent = Agent(
            system=SYSTEM_PROMPT,
            tools=self._active_tools,
            transport=chat_transport,
        )

        await self._update_status()
        self.set_interval(0.1, self._flush_text)
        self.query_one("#input", Input).focus()
        await self._show_welcome()

    async def _show_welcome(self) -> None:
        lines: list[str] = [
            "# Axio Agent",
            "",
            "AI coding assistant. Type a task below to get started.",
            "Press `Ctrl+P` to open the command palette.",
            "",
        ]

        # Transports
        lines.append("**Transports**")
        lines.append("")
        for name in self._transports.discovered:
            meta = self._transports.get_meta(name)
            if name in self._transports.available:
                t = self._transports.get_transport(name)
                count = len(t.models)
                lines.append(f"- {meta.label} ({count} models)")
            else:
                lines.append(f"- ~~{meta.label}~~ — no API key")

        # Model roles
        lines.append("")
        lines.append("**Models**")
        lines.append("")
        lines.append("| Role | Model |")
        lines.append("|------|-------|")
        for role in ModelRole:
            label = _ROLE_LABELS[role]
            if role in self._role_bindings:
                b = self._role_bindings[role]
                lines.append(f"| {label} | `{b.model.id}` ({b.transport}) |")
            else:
                lines.append(f"| {label} | *not configured* |")

        # Tools
        lines.append("")
        enabled = [t for t in self._all_tools if t.name not in self._disabled_plugins]
        disabled_count = len(self._all_tools) - len(enabled)
        suffix = f" ({disabled_count} disabled)" if disabled_count else ""
        lines.append(f"**Tools ({len(enabled)})**{suffix}")
        lines.append("")
        lines.append(", ".join(f"`{t.name}`" for t in enabled))

        # Guards
        if self._guard_classes:
            lines.append("")
            enabled_guards = [g for g in self._guard_classes if g not in self._disabled_guards]
            disabled_guards = [g for g in self._guard_classes if g in self._disabled_guards]
            parts: list[str] = [f"`{g}`" for g in enabled_guards]
            parts.extend(f"~~`{g}`~~" for g in disabled_guards)
            lines.append(f"**Guards:** {', '.join(parts)}")

        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(_Markdown("\n".join(lines)))
        scroll.scroll_end(animate=False)

    async def on_unmount(self) -> None:
        # Cancel the active agent worker first so the SSE connection is
        # released before we close the aiohttp session.  Without this the
        # executor shutdown hangs waiting for orphan threads.
        if self._worker is not None and self._worker.is_running:
            self._worker.cancel()
            try:
                await self._worker.wait()
            except Exception:
                pass

        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        StatusLine._callback = None
        SubAgent._factory = None
        VisionAnalyze._transport = None
        for plugin in self._tools_plugins.values():
            try:
                await plugin.close()
            except Exception:
                pass
        if self._embed_cache is not None:
            await self._embed_cache.close()
        await self._chat_context.close()
        await self._config.close()
        await self._global_config.close()
        if self._session is not None:
            await self._session.close()
            # Give aiohttp time to close underlying SSL transports
            await asyncio.sleep(0.25)

    async def _make_subagent(self) -> tuple[Agent, ContextStore]:
        context = await MemoryContextStore.from_context(self._chat_context)
        assert self._agent is not None
        sub_tools = [
            Tool(name=t.name, description=t.description, handler=t.handler)
            for t in self._agent.tools
            if t.name != "subagent"
        ]
        transport = self._chat_transport if self._chat_transport is not None else self._agent.transport
        sub_agent = replace(
            self._agent,
            system=SUBAGENT_SYSTEM_PROMPT,
            tools=sub_tools,
            max_iterations=25,
            transport=transport,
        )
        if ModelRole.SUBAGENT in self._role_bindings:
            sb = self._role_bindings[ModelRole.SUBAGENT]
            sub_transport = self._transports.make_transport(sb.transport, sb.model)
            sub_agent = replace(sub_agent, transport=sub_transport)
        return sub_agent, context

    def _on_agent_status(self, message: str) -> None:
        widget = self.query_one("#tui-status", Static)
        widget.update(message)
        widget.add_class("visible")

    def _hide_agent_status(self) -> None:
        self.query_one("#tui-status", Static).remove_class("visible")

    async def _update_status(self) -> None:
        total_in, total_out = await self._chat_context.get_context_tokens()
        parts: list[str] = []
        if self._agent_emoji:
            parts.append(self._agent_emoji)
        parts.append(f"Model: {self._model_name}")
        if self._chat_model is not None and self._last_input_tokens > 0:
            ctx_win = self._chat_model.context_window
            pct = self._last_input_tokens * 100 // ctx_win
            parts.append(f"Context: {self._last_input_tokens:,} / {ctx_win:,} ({pct}%)")
        parts.append(f"Tokens: {total_in:,} in / {total_out:,} out")
        self.query_one("#status", Static).update("  |  ".join(parts))

    async def _flush_text(self) -> None:
        if not self._text_dirty or self._current_md is None:
            return
        self._text_dirty = False
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await self._current_md.update(self._response_text)
        scroll.scroll_end(animate=False)

    async def _ensure_md(self) -> Markdown:
        if self._current_md is None:
            scroll = self.query_one("#log", VerticalScroll)
            if not scroll.is_attached:
                self._current_md = _Markdown("")
                return self._current_md
            self._current_md = _Markdown("")
            await scroll.mount(self._current_md)
        return self._current_md

    async def _write_meta(self, content: str) -> None:
        """Flush pending text, then mount a styled Static line."""
        await self._flush_text()
        self._current_md = None
        self._response_text = ""
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(Static(content, classes="meta"))
        scroll.scroll_end(animate=False)

    async def _write_user(self, content: str) -> None:
        """Flush pending text, then mount a user message with distinct styling."""
        await self._flush_text()
        self._current_md = None
        self._response_text = ""
        scroll = self.query_one("#log", VerticalScroll)
        if not scroll.is_attached:
            return
        await scroll.mount(Static(content, classes="user-msg"))
        scroll.scroll_end(animate=False)

    async def _ensure_tool_status(self) -> _ToolStatusWidget:
        if self._tool_status is None:
            await self._flush_text()
            self._current_md = None
            self._response_text = ""
            self._tool_status = _ToolStatusWidget()
            scroll = self.query_one("#log", VerticalScroll)
            if scroll.is_attached:
                await scroll.mount(self._tool_status)
                scroll.scroll_end(animate=False)
        return self._tool_status

    async def _path_guard_prompt_fn(self, msg: str) -> str:
        async with self._guard_lock:
            try:
                from axio_tui_guards.dialogs import PathGuardDialog

                result: str = await self.push_screen_wait(PathGuardDialog(msg))
                return result
            except ImportError:
                return "y"

    async def _llm_guard_prompt_fn(self, msg: str) -> str:
        async with self._guard_lock:
            try:
                from axio_tui_guards.dialogs import LLMGuardDialog

                result = await self.push_screen_wait(LLMGuardDialog(msg))
                return str(result)
            except ImportError:
                return "y"

    def _instantiate_guard(self, name: str, cls: type[PermissionGuard]) -> PermissionGuard | None:
        match name:
            case "path":
                return cls(prompt_fn=self._path_guard_prompt_fn)  # type: ignore[call-arg]
            case "llm":
                if ModelRole.GUARD not in self._role_bindings:
                    return None
                gb = self._role_bindings[ModelRole.GUARD]
                guard_transport = self._transports.make_transport(gb.transport, gb.model)
                from axio.context import MemoryContextStore as _MemCtx

                guard_agent = Agent(
                    system="You review tool calls for safety. Use the confirm tool to classify.",
                    tools=[Tool(name="confirm", description="Classify tool call", handler=Confirm)],
                    transport=guard_transport,
                )
                guard_context = _MemCtx()
                return cls(  # type: ignore[call-arg]
                    agent=guard_agent,
                    context=guard_context,
                    prompt_fn=self._llm_guard_prompt_fn,
                )
            case _:
                try:
                    return cls()
                except TypeError:
                    return None

    def _rebuild_agent_tools(self) -> None:
        tools: list[Tool] = []
        for t in self._all_tools:
            if t.name in self._disabled_plugins:
                continue
            guards: list[PermissionGuard] = []
            for guard_name, tool_names in self._guard_tool_map.items():
                if guard_name in self._disabled_guards:
                    continue
                if t.name in tool_names and guard_name in self._guard_instances:
                    guards.append(self._guard_instances[guard_name])
            if guards:
                t = replace(t, guards=tuple(guards))
            tools.append(t)
        self._active_tools = tools
        if self._agent is not None:
            self._agent.tools = tools

    def _show_model_select(self) -> None:
        if not self._transports.available:
            return
        self.push_screen(ModelRoleSelectScreen(self._role_bindings), self._on_role_selected)

    def _on_role_selected(self, role: ModelRole | None) -> None:
        if role is None or not self._transports.available:
            return
        match role:
            case ModelRole.GUARD:
                models = self._transports.all_models()
            case ModelRole.VISION:
                models = self._transports.all_models(Capability.vision)
            case ModelRole.EMBEDDING:
                models = self._transports.all_models(Capability.embedding)
            case ModelRole.REASONING:
                models = self._transports.all_models(Capability.reasoning)
            case _:
                models = self._transports.all_models(Capability.tool_use)
        self.push_screen(
            ModelSelectScreen(models),
            lambda result: self._on_model_selected(role, result),
        )

    async def _on_model_selected(self, role: ModelRole, result: tuple[str, ModelSpec] | None) -> None:
        if result is None:
            return
        transport_name, spec = result
        binding = RoleBinding(transport=transport_name, model=spec)
        self._role_bindings[role] = binding
        await self._config.set(f"model.{role}", self._transports.encode(transport_name, spec.id))

        match role:
            case ModelRole.CHAT:
                self._chat_transport = self._transports.make_transport(transport_name, spec)
                if self._agent is not None:
                    transport = self._chat_transport
                    if self._tool_selector is not None:
                        transport = ToolFilteringTransport(transport, self._tool_selector)
                    self._agent.transport = transport
                self._chat_model = spec
                self._model_name = spec.id
                self.sub_title = spec.id
            case ModelRole.VISION:
                VisionAnalyze._transport = self._transports.make_transport(transport_name, spec)
            case ModelRole.EMBEDDING:
                if self._agent is not None:
                    if self._embed_cache is not None:
                        await self._embed_cache.close()
                    raw_transport = self._transports.make_transport(transport_name, spec)
                    try:
                        from axio_tui_rag.embedding_cache import CachedEmbeddingTransport

                        self._embed_cache = CachedEmbeddingTransport(raw_transport, DB_PATH, spec.id)
                        self._tool_selector = EmbeddingToolSelector(
                            transport=self._embed_cache,
                            top_k=6,
                            pinned=frozenset({"status_line"}),
                        )
                        if self._chat_transport is not None:
                            self._agent.transport = ToolFilteringTransport(
                                self._chat_transport,
                                self._tool_selector,
                            )
                    except ImportError:
                        pass
            case ModelRole.GUARD | ModelRole.COMPACT | ModelRole.SUBAGENT | ModelRole.REASONING:
                pass  # stored in self._role_bindings, used on demand
        await self._update_status()

    async def _new_session(self) -> None:
        await self._chat_context.close()
        self._chat_context = SQLiteContextStore(DB_PATH, uuid4().hex)
        self._last_input_tokens = 0
        await self.action_clear_log()
        await self._write_meta("[dim]--- New session ---[/]")

    async def _fork_conversation(self) -> None:
        if self._is_running:
            return
        forked = await self._chat_context.fork()
        await self._chat_context.close()
        self._chat_context = forked
        await self.action_clear_log()
        await self._render_history()
        await self._write_meta("[dim]--- Forked conversation ---[/]")

    async def _show_session_select(self) -> None:
        sessions = await self._chat_context.list_sessions()
        if not sessions:
            await self._write_meta("[dim]No previous sessions found[/]")
            return
        self.push_screen(SessionSelectScreen(sessions), self._on_session_selected)

    async def _on_session_selected(self, info: SessionInfo | None) -> None:
        if info is None:
            return
        await self._chat_context.close()
        self._chat_context = SQLiteContextStore(DB_PATH, info.session_id)
        self._last_input_tokens = 0
        await self._update_status()
        await self.action_clear_log()
        await self._write_meta(f"[dim]--- Restored session ({info.message_count} msgs): {info.preview} ---[/]")
        await self._render_history()

    async def _render_history(self) -> None:
        """Replay stored messages into the chat log."""
        self._nav_index = None
        from axio.blocks import TextBlock, ToolResultBlock, ToolUseBlock

        history = await self._chat_context.get_history()
        scroll = self.query_one("#log", VerticalScroll)
        # Build result map: tool_use_id -> ToolResultBlock
        result_map: dict[str, ToolResultBlock] = {}
        for msg in history:
            if msg.role == "user":
                for b in msg.content:
                    if isinstance(b, ToolResultBlock):
                        result_map[b.tool_use_id] = b

        for msg in history:
            if msg.role == "user":
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    await scroll.mount(Static(f"[bold]You:[/] {texts[0]}", classes="user-msg"))
                # Tool results rendered as part of assistant tool summary — skip here
            else:
                texts = [b.text for b in msg.content if isinstance(b, TextBlock)]
                if texts:
                    md = _Markdown("\n\n".join(texts))
                    await scroll.mount(md)
                tool_blocks = [b for b in msg.content if isinstance(b, ToolUseBlock)]
                if tool_blocks:
                    w = _ToolStatusWidget()
                    for b in tool_blocks:
                        result_block = result_map.get(b.id)
                        content_str = ""
                        is_error = False
                        if result_block is not None:
                            is_error = result_block.is_error
                            content_str = (
                                result_block.content
                                if isinstance(result_block.content, str)
                                else "\n".join(tb.text for tb in result_block.content if isinstance(tb, TextBlock))
                            )
                        w.track(b.id, b.name)
                        w.complete(b.id, is_error=is_error, content=content_str, tool_input=b.input)
                    await scroll.mount(w)
        scroll.scroll_end(animate=False)

    async def _replace_context(self, messages: list[Message]) -> None:
        """Close current context, create a fresh SQLite store, populate it."""
        total_in, total_out = await self._chat_context.get_context_tokens()
        await self._chat_context.close()
        self._chat_context = SQLiteContextStore(DB_PATH, uuid4().hex)
        for msg in messages:
            await self._chat_context.append(msg)
        if total_in or total_out:
            await self._chat_context.set_context_tokens(total_in, total_out)

    def _should_compact(self) -> bool:
        if self._chat_model is None or self._last_input_tokens == 0:
            return False
        return self._last_input_tokens / self._chat_model.context_window > 0.7

    async def _compact_with_progress(self, max_messages: int = 20) -> list[Message] | None:
        """Run compaction while showing a live elapsed-time counter in the chat."""
        if self._agent is None:
            return None
        elapsed = 0
        widget = Static("[dim]Compacting in progress... 0s[/]", classes="meta")
        scroll = self.query_one("#log", VerticalScroll)
        await scroll.mount(widget)
        scroll.scroll_end(animate=False)

        def _tick() -> None:
            nonlocal elapsed
            elapsed += 1
            widget.update(f"[dim]Compacting in progress... {elapsed}s[/]")

        timer = self.set_interval(1, _tick)
        try:
            if ModelRole.COMPACT in self._role_bindings:
                cb = self._role_bindings[ModelRole.COMPACT]
                transport = self._transports.make_transport(cb.transport, cb.model)
            else:
                transport = self._agent.transport
            return await compact_context(self._chat_context, transport, max_messages=max_messages)
        finally:
            timer.stop()
            await widget.remove()

    async def _compact_conversation(self) -> None:
        if self._agent is None or self._is_running:
            return
        self.run_worker(self._do_compact(), exclusive=True)

    async def _do_compact(self) -> None:
        messages = await self._compact_with_progress()
        if messages is not None:
            await self._replace_context(messages)
            await self._write_meta("[dim]--- Context compacted ---[/]")
        else:
            await self._write_meta("[dim]--- Nothing to compact ---[/]")

    def action_command_palette(self) -> None:
        self._nav_select(None)
        super().action_command_palette()

    def get_system_commands(self, screen: object) -> Generator[SystemCommand, None, None]:
        yield from super().get_system_commands(screen)  # type: ignore[arg-type]
        yield SystemCommand("Change Model", "Switch the active LLM model", self._show_model_select)
        yield SystemCommand("Clear Log", "Clear the chat log", self.action_clear_log)
        yield SystemCommand("New Session", "Reset conversation context", self._new_session)
        yield SystemCommand("Fork Conversation", "Branch from current conversation", self._fork_conversation)
        yield SystemCommand("Restore Session", "Load a previous conversation", self._show_session_select)
        yield SystemCommand(
            "Compact Conversation", "Summarize old messages to save context", self._compact_conversation
        )
        yield SystemCommand(
            "Toggle Dev Console", "Show or hide the developer log console", self.action_toggle_dev_console
        )
        yield SystemCommand("Manage Plugins", "Configure tools and guards", self._show_plugin_hub)
        for pname, plugin in self._tools_plugins.items():
            yield SystemCommand(
                f"Manage {plugin.label}",
                f"Configure {plugin.label}",
                lambda p=plugin: self._show_tools_plugin(p),
            )
        screens = self._transports.settings_screens()
        for name in self._transports.discovered:
            if name not in screens:
                continue
            meta = self._transports.get_meta(name)
            screen_cls = screens[name]
            yield SystemCommand(
                f"Configure {meta.label}",
                f"Edit {meta.label} transport settings",
                lambda n=name, s=screen_cls: self._show_transport_settings(n, s),
            )

    async def _show_transport_settings(self, name: str, screen_cls: type) -> None:
        settings = await self._transports.get_settings(name)
        self.push_screen(
            screen_cls(settings),
            lambda result: self._on_transport_configured(name, result),
        )

    async def _on_transport_configured(self, name: str, result: dict[str, str] | None) -> None:
        if result is None:
            return
        was_available = name in self._transports.available
        await self._transports.save_settings(name, result)
        meta = self._transports.get_meta(name)

        if name not in self._transports.available:
            await self._write_meta(f"[yellow]{meta.label}: API key required to activate[/]")
            return

        if not was_available:
            await self._activate_transport(name)
        else:
            await self._reconfigure_transport(name)
        await self._update_status()

    async def _activate_transport(self, name: str) -> None:
        """Resolve role defaults and set up agent for a newly activated transport."""
        meta = self._transports.get_meta(name)

        # Fill unset roles from this transport's defaults
        for role in ModelRole:
            if role in self._role_bindings:
                continue
            binding = self._transports.resolve_default(role.value)
            if binding is not None and binding.transport == name:
                self._role_bindings[role] = binding
                await self._config.set(f"model.{role}", self._transports.encode(name, binding.model.id))

        # Set up chat if this transport claimed the chat role
        if ModelRole.CHAT in self._role_bindings and self._role_bindings[ModelRole.CHAT].transport == name:
            cb = self._role_bindings[ModelRole.CHAT]
            self._chat_transport = self._transports.make_transport(name, cb.model)
            self._chat_model = cb.model
            self._model_name = cb.model.id
            self.sub_title = self._model_name
            transport = self._chat_transport
            if self._tool_selector is not None:
                transport = ToolFilteringTransport(transport, self._tool_selector)
            if self._agent is None:
                self._rebuild_agent_tools()
                self._agent = Agent(system=SYSTEM_PROMPT, tools=self._active_tools, transport=transport)
            else:
                self._agent.transport = transport

        if ModelRole.VISION in self._role_bindings and self._role_bindings[ModelRole.VISION].transport == name:
            vb = self._role_bindings[ModelRole.VISION]
            VisionAnalyze._transport = self._transports.make_transport(name, vb.model)

        await self._write_meta(f"[green]{meta.label} activated[/]")

    async def _reconfigure_transport(self, name: str) -> None:
        """Rebuild cached transport instances after settings change."""
        meta = self._transports.get_meta(name)
        for role, binding in self._role_bindings.items():
            if binding.transport != name:
                continue
            match role:
                case ModelRole.CHAT:
                    self._chat_transport = self._transports.make_transport(name, binding.model)
                    if self._agent is not None:
                        transport = self._chat_transport
                        if self._tool_selector is not None:
                            transport = ToolFilteringTransport(transport, self._tool_selector)
                        self._agent.transport = transport
                case ModelRole.VISION:
                    VisionAnalyze._transport = self._transports.make_transport(name, binding.model)
                case ModelRole.EMBEDDING:
                    if self._agent is not None and self._embed_cache is not None:
                        await self._embed_cache.close()
                        raw_transport = self._transports.make_transport(name, binding.model)
                        try:
                            from axio_tui_rag.embedding_cache import CachedEmbeddingTransport

                            self._embed_cache = CachedEmbeddingTransport(raw_transport, DB_PATH, binding.model.id)
                            self._tool_selector = EmbeddingToolSelector(
                                transport=self._embed_cache,
                                top_k=6,
                                pinned=frozenset({"status_line"}),
                            )
                            if self._chat_transport is not None:
                                self._agent.transport = ToolFilteringTransport(
                                    self._chat_transport,
                                    self._tool_selector,
                                )
                        except ImportError:
                            pass
        await self._write_meta(f"[dim]{meta.label} settings updated[/]")

    def _guard_descriptions(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, cls in self._guard_classes.items():
            doc = cls.__doc__
            result[name] = doc.strip().split("\n")[0] if doc else name
        return result

    def _show_plugin_hub(self) -> None:
        self.push_screen(
            PluginHubScreen(
                tools=self._all_tools,
                disabled_plugins=self._disabled_plugins,
                guard_names=self._guard_descriptions(),
                disabled_guards=self._disabled_guards,
                guard_tool_map=self._guard_tool_map,
                on_plugins_changed=self._on_plugins_changed,
                on_guards_changed=self._on_guards_changed,
            ),
        )

    def _show_tools_plugin(self, plugin: ToolsPlugin) -> None:
        screen = plugin.settings_screen()
        self.push_screen(screen, lambda _result: self._on_tools_plugin_dismissed())

    def _on_tools_plugin_dismissed(self) -> None:
        # Rebuild tools list with all plugin tools
        tools: list[Tool] = discover_tools()
        for plugin in self._tools_plugins.values():
            tools.extend(plugin.all_tools)
        self._all_tools = tools
        self._rebuild_agent_tools()

    async def _on_plugins_changed(self, disabled: set[str]) -> None:
        if self._agent is None:
            return
        self._disabled_plugins = disabled
        if disabled:
            await self._config.set("plugins.disabled", ",".join(sorted(disabled)))
        else:
            await self._config.delete("plugins.disabled")
        self._rebuild_agent_tools()

    async def _on_guards_changed(self, disabled_guards: set[str], guard_tool_map: dict[str, set[str]]) -> None:
        if self._agent is None:
            return
        # Persist disabled guards
        self._disabled_guards = disabled_guards
        if disabled_guards:
            await self._config.set("guards.disabled", ",".join(sorted(disabled_guards)))
        else:
            await self._config.delete("guards.disabled")

        # Persist guard-tool assignments
        self._guard_tool_map = guard_tool_map
        for guard_name, tool_names in guard_tool_map.items():
            if tool_names:
                await self._config.set(f"guards.{guard_name}.tools", ",".join(sorted(tool_names)))
            else:
                await self._config.delete(f"guards.{guard_name}.tools")

        # Re-instantiate guards that were re-enabled
        for guard_name, cls in self._guard_classes.items():
            if guard_name in disabled_guards:
                self._guard_instances.pop(guard_name, None)
            elif guard_name not in self._guard_instances:
                instance = self._instantiate_guard(guard_name, cls)
                if instance is not None:
                    self._guard_instances[guard_name] = instance

        self._rebuild_agent_tools()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        self._nav_select(None)
        value = message.value.strip()
        if not value:
            return
        message.input.value = ""

        if value.lower() == "exit":
            self.exit()
            return

        if self._is_running:
            widget = Static(f"[dim]You (queued):[/] {value}", classes="user-msg")
            scroll = self.query_one("#log", VerticalScroll)
            await scroll.mount(widget)
            scroll.scroll_end(animate=False)
            self._pending_input.put_nowait((value, widget))
            return

        await self._write_user(f"[bold]You:[/] {value}")
        self._worker = self.run_worker(self._run_agent_work(value), exclusive=True)

    def action_stop_agent(self) -> None:
        if self._worker is not None and self._worker.is_running:
            self._worker.cancel()

    async def _run_agent_work(self, task: str) -> None:
        if self._agent is None:
            return

        self._is_running = True
        if self._should_compact():
            messages = await self._compact_with_progress(max_messages=0)
            if messages is not None:
                await self._replace_context(messages)
                self._last_input_tokens = 0
                await self._write_meta("[dim]--- Context compacted ---[/]")

        cancelled = False
        try:
            while True:
                self._agent_emoji = "💬"
                await self._update_status()
                stream = self._agent.run_stream(task, self._chat_context)
                try:
                    async for event in stream:
                        match event:
                            case ReasoningDelta(delta=delta):
                                self._tool_status = None
                                self._agent_emoji = "🤔"
                                await self._ensure_md()
                                self._response_text += delta
                                self._text_dirty = True
                            case TextDelta(delta=delta):
                                self._tool_status = None
                                self._agent_emoji = "💬"
                                await self._ensure_md()
                                self._response_text += delta
                                self._text_dirty = True
                            case ToolUseStart(tool_use_id=tid, name=name) if name != "status_line":
                                self._agent_emoji = "🔧"
                                await self._update_status()
                                w = await self._ensure_tool_status()
                                w.track(tid, name)
                            case ToolResult(
                                tool_use_id=tid,
                                name=name,
                                is_error=is_error,
                                content=content,
                                input=inp,
                            ) if name != "status_line":
                                w = await self._ensure_tool_status()
                                w.complete(tid, is_error=is_error, content=content, tool_input=inp)
                            case IterationEnd(usage=usage):
                                self._agent_emoji = "✅"
                                self._last_input_tokens = usage.input_tokens
                                await self._update_status()
                            case SessionEndEvent():
                                await self._flush_text()
                                self._current_md = None
                                self._response_text = ""
                                self._tool_status = None
                            case Error(exception=exc):
                                await self._write_meta(f"[red]Error: {exc}[/]")
                finally:
                    await stream.aclose()

                if not self._pending_input.empty():
                    task, widget = self._pending_input.get_nowait()
                    widget.update(f"[bold]You:[/] {task}")
                    continue
                break
        except asyncio.CancelledError:
            cancelled = True
        except Exception as exc:
            await self._write_meta(f"[red]Error: {exc}[/]")
        finally:
            self._is_running = False
            self._agent_emoji = ""
            self._hide_agent_status()
            # Drain stale queued messages on cancellation or error
            while not self._pending_input.empty():
                self._pending_input.get_nowait()
            self._tool_status = None
            await self._update_status()
            if cancelled:
                await self._flush_text()
                self._current_md = None
                self._response_text = ""
                await self._write_meta("[yellow]--- Cancelled ---[/]")
            self.query_one("#input", Input).focus()

    def action_toggle_dev_console(self) -> None:
        self.query_one("#dev-console", RichLog).toggle_class("visible")

    # -- Chat message navigation (Alt+Up / Alt+Down) --

    def _nav_children(self) -> list[Widget]:
        return list(self.query_one("#log").children)

    def _nav_select(self, index: int | None) -> None:
        children = self._nav_children()
        if self._nav_index is not None and self._nav_index < len(children):
            children[self._nav_index].remove_class("nav-selected")
        self._nav_index = index
        if index is not None and index < len(children):
            children[index].add_class("nav-selected")
            children[index].scroll_visible()

    def _nav_exit(self) -> None:
        self._nav_select(None)
        self.query_one("#input", Input).focus()

    def _nav_activate(self) -> None:
        if self._nav_index is None:
            return
        children = self._nav_children()
        if not children or self._nav_index >= len(children):
            return
        widget = children[self._nav_index]
        if isinstance(widget, _ToolStatusWidget):
            for tid, info in widget._tools.items():
                if info.status is not None:
                    widget.action_show_detail(tid)
                    return

    def action_nav_up(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(len(children) - 1)
        elif self._nav_index > 0:
            self._nav_select(self._nav_index - 1)

    def action_nav_down(self) -> None:
        children = self._nav_children()
        if not children:
            return
        if self._nav_index is None:
            self._nav_select(0)
        elif self._nav_index < len(children) - 1:
            self._nav_select(self._nav_index + 1)
        else:
            self._nav_exit()

    def on_key(self, event: Key) -> None:
        if self._nav_index is None:
            return
        if isinstance(self.screen, ModalScreen):
            return
        if isinstance(self.focused, Input):
            return
        if event.key == "escape":
            self._nav_exit()
            event.prevent_default()
            event.stop()
        elif event.key == "enter":
            self._nav_activate()
            event.prevent_default()
            event.stop()

    async def action_clear_log(self) -> None:
        await self.query_one("#log", VerticalScroll).remove_children()
        self._current_md = None
        self._response_text = ""
        self._tool_status = None
        self._nav_index = None

    def action_quit(self) -> None:  # type: ignore[override]
        self.push_screen(QuitDialog(), callback=self._on_quit_confirmed)

    def _on_quit_confirmed(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()
