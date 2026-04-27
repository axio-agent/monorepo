"""Agent Swarm — a team of specialist AI agents tackling a task from idea to delivery.

Usage:
    uv run python main.py "Build a Python rate limiter library"
    uv run python main.py --workspace /tmp/my_project "Design a REST API for a blog"

Requires NEBIUS_API_KEY in the environment (Nebius AI Studio — TokenFactory).
To adjust which model each role uses, edit the role_models dict in main().
"""

import argparse
import asyncio
import sys
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import aiohttp
from axio.events import (
    Error,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolResult,
    ToolUseStart,
)
from axio.models import ModelSpec
from axio.permission import PermissionGuard
from axio.tool import ToolHandler
from axio_transport_openai.nebius import NebiusTransport
from rich.console import Console, ConsoleRenderable
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .swarm import run_swarm

# ---------------------------------------------------------------------------
# Role colours
# ---------------------------------------------------------------------------

ROLE_STYLES: dict[str, str] = {
    "orchestrator": "bold white",
    "architect": "bold cyan",
    "backend_dev": "bold green",
    "frontend_dev": "bold blue",
    "project_manager": "bold yellow",
    "qa": "bold magenta",
    "designer": "bold red",
    "ux_engineer": "bold bright_blue",
    "etl_engineer": "bold bright_cyan",
    "security_engineer": "bold bright_red",
    "challenger": "bold orange3",
    "analyst": "dim white",
}

ROLE_TITLES: dict[str, str] = {
    "orchestrator": "Orchestrator",
    "architect": "Software Architect",
    "backend_dev": "Backend Developer",
    "frontend_dev": "Frontend Developer",
    "project_manager": "Project Manager",
    "qa": "QA Engineer",
    "designer": "Visual Designer",
    "ux_engineer": "UX Engineer",
    "etl_engineer": "ETL Engineer",
    "security_engineer": "Security Engineer",
    "challenger": "Challenger",
    "analyst": "Analyst",
}

# Tools whose output is file content — wrap in a fenced code block for display
FILE_CONTENT_TOOLS = {"read_file", "run_python", "shell"}

EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
    ".sql": "sql",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".txt": "",
}


def fence(content: str, tool_name: str, tool_input: dict[str, Any]) -> str:
    """Wrap *content* in a fenced code block if the tool returns file/shell output."""
    if tool_name not in FILE_CONTENT_TOOLS:
        return content
    filename = str(tool_input.get("filename") or tool_input.get("file_path") or "")
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ""
    lang = EXT_LANG.get(suffix, "")
    return f"```{lang}\n{content}\n```"


# ---------------------------------------------------------------------------
# Guard: logs every tool call before execution
# ---------------------------------------------------------------------------


class RoleGuard(PermissionGuard):
    """PermissionGuard that logs tool inputs to the SwarmRenderer before execution.

    Acquires the renderer's lock so concurrent delegate outputs don't interleave.
    This replaces manual ToolFieldStart / ToolFieldDelta / ToolFieldEnd handling
    in the event loop — the guard receives the fully-parsed ToolHandler instance
    and can render all fields at once.
    """

    def __init__(self, role: str, tool_name: str, renderer: "SwarmRenderer") -> None:
        self._role = role
        self._tool_name = tool_name
        self._renderer = renderer

    async def check(self, handler: ToolHandler) -> ToolHandler:
        async with self._renderer._lock:
            self._renderer._print_tool_call(self._role, self._tool_name, handler)
        return handler


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------


class StatusBar(ConsoleRenderable):
    """Rich renderable that reads SwarmRenderer state on every refresh."""

    def __init__(self, renderer: "SwarmRenderer") -> None:
        self._r = renderer

    def __rich_console__(self, console: Console, options: object) -> object:  # type: ignore[override]
        r = self._r
        grid = Table.grid(padding=(0, 2))
        for role in sorted(r._agent_status):
            style = SwarmRenderer._role_style(role)
            title = SwarmRenderer._role_title(role)
            grid.add_row(
                Spinner("dots", style=style),
                Text(title, style=style),
                Text(r._agent_status[role], style="dim"),
            )
        elapsed = int(time.monotonic() - r._start_time)
        m, s = divmod(elapsed, 60)
        summary = f"{r._event_count} events · ↑{r._total_in:,} ↓{r._total_out:,} tokens · {m:02d}:{s:02d}"
        grid.add_row(Text(""), Text(""), Text(summary, style="dim"))
        yield grid


class SwarmRenderer:
    def __init__(self, console: Console) -> None:
        self._lock = asyncio.Lock()
        self._active_text: str | None = None
        self._text_buf: dict[str, list[str]] = {}
        self._header_printed: set[str] = set()
        # Status bar state
        self._agent_status: dict[str, str] = {}
        self._total_in: int = 0
        self._total_out: int = 0
        self._event_count: int = 0
        self._start_time: float = time.monotonic()
        self._live = Live(StatusBar(self), console=console, refresh_per_second=4)

    def __enter__(self) -> "SwarmRenderer":
        self._start_time = time.monotonic()
        self._live.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any | None
    ) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    @cache
    def make_prompt_fn(self) -> Callable[[], str]:
        """Return a prompt function that pauses this renderer's Live during input."""
        live = self._live

        def _prompt() -> str:
            live.stop()
            try:
                return input("> ").strip() or "(no answer)"
            finally:
                live.start()

        return _prompt

    def _print(self, *args: object, **kwargs: object) -> None:
        self._live.console.print(*args, **kwargs)  # type: ignore[arg-type]

    def make_guard(self, role: str, tool_name: str) -> RoleGuard:
        """Factory passed to run_swarm so every tool gets a logging guard."""
        return RoleGuard(role=role, tool_name=tool_name, renderer=self)

    # ------------------------------------------------------------------
    # Public callback — handles stream events from the agent loop
    # ------------------------------------------------------------------

    async def on_event(self, role: str, event: StreamEvent) -> None:  # noqa: C901
        async with self._lock:
            await self._handle(role, event)

    async def _handle(self, role: str, event: StreamEvent) -> None:
        self._event_count += 1
        style = ROLE_STYLES.get(role, "white")

        match event:
            case ReasoningDelta():
                self._flush_text(role)
                if role not in self._header_printed:
                    self._print_header(role)
                self._agent_status[role] = "reasoning…"
                self._print(f"[dim italic]{event.delta}[/dim italic]", end="", highlight=False)

            case TextDelta():
                if role not in self._header_printed:
                    self._print_header(role)
                self._text_buf.setdefault(role, []).append(event.delta)
                self._active_text = role
                self._agent_status[role] = "writing…"

            case ToolUseStart():
                self._flush_text(role)
                if role not in self._header_printed:
                    self._print_header(role)
                self._agent_status[role] = f"▶ {event.name}"

            case ToolResult():
                result_status = "[red]✗ error[/red]" if event.is_error else "[green]✓[/green]"
                content = (event.content or "").strip()
                self._print(
                    f"[{style}]{event.name}[/{style}] {result_status}",
                    highlight=False,
                )
                if content:
                    self._print(Markdown(fence(content, event.name, event.input)))
                self._agent_status[role] = "thinking…"

            case IterationEnd():
                u = event.usage
                self._total_in += u.input_tokens
                self._total_out += u.output_tokens
                self._agent_status[role] = "thinking…"
                msg = (
                    f"[dim]  iter {event.iteration} · {event.stop_reason} · ↑{u.input_tokens} ↓{u.output_tokens}[/dim]"
                )
                self._print(msg, highlight=False)

            case Error():
                self._flush_text(role)
                self._agent_status.pop(role, None)
                self._print(f"[bold red]ERROR ({role}): {event.exception}[/bold red]")

            case SessionEndEvent():
                self._flush_text(role)
                self._agent_status.pop(role, None)
                u = event.total_usage
                self._print(
                    f"[dim][{style}]{self._role_title(role)}[/{style}] "
                    f"done — ↑{u.input_tokens} ↓{u.output_tokens} tokens total[/dim]",
                    highlight=False,
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_text(self, role: str) -> None:
        buf = self._text_buf.pop(role, None)
        if not buf:
            return
        self._print(Markdown("".join(buf)))
        if self._active_text == role:
            self._active_text = None

    @staticmethod
    def _parse_role(agent_id: str) -> tuple[str, str | None, str | None]:
        """Split 'backend_dev#3:auth middleware' → (role, number, topic)."""
        role_part, _, topic = agent_id.partition(":")
        base, _, num = role_part.partition("#")
        return base, (num or None), (topic or None)

    @staticmethod
    def _role_style(agent_id: str) -> str:
        base, _, _ = SwarmRenderer._parse_role(agent_id)
        return ROLE_STYLES.get(base, "white")

    @staticmethod
    def _role_title(agent_id: str) -> str:
        base, num, topic = SwarmRenderer._parse_role(agent_id)
        title = ROLE_TITLES.get(base, base)
        if num:
            title = f"{title} #{num}"
        if topic:
            title = f"{title} [{topic}]"
        return title

    def _print_header(self, role: str) -> None:
        style = self._role_style(role)
        title = self._role_title(role)
        self._agent_status.setdefault(role, "starting…")
        self._print(Rule(Text(title, style=style)))
        self._header_printed.add(role)

    def _print_tool_call(self, role: str, tool_name: str, handler: ToolHandler) -> None:
        """Called by RoleGuard before each tool execution."""
        style = self._role_style(role)
        self._print(f"  [{style}]▶ {tool_name}[/{style}]", highlight=False)
        for key, value in handler.model_dump().items():
            v_str = str(value)
            if len(v_str) > 200:
                v_str = v_str[:197] + "…"
            v_str = v_str.replace("\n", "↵")
            self._print(f"    [dim]{key}:[/dim] {v_str}", highlight=False)

    def print_workspace_tree(self, workspace: Path) -> None:
        self._print()
        self._print(Rule("[dim]Workspace[/dim]"))
        if not workspace.exists():
            self._print("[dim](empty)[/dim]")
            return
        tree = Tree(f"[bold]{workspace}[/bold]")
        for path in sorted(workspace.rglob("*")):
            if path.is_file() and not any(part.startswith(".") for part in path.parts):
                rel = path.relative_to(workspace)
                parts = rel.parts
                node = tree
                for part in parts[:-1]:
                    for child in node.children:
                        if child.label == part:  # type: ignore[arg-type]
                            node = child
                            break
                    else:
                        node = node.add(part)
                size = path.stat().st_size
                node.add(f"[green]{parts[-1]}[/green] [dim]({size:,} bytes)[/dim]")
        self._print(tree)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an agent swarm to tackle a software task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task", help="The task for the team to solve")
    parser.add_argument(
        "--workspace",
        required=True,
        help="Directory where agents read and write files",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()

    console = Console()
    renderer = SwarmRenderer(console)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(sock_read=120, sock_connect=2)) as session:
        transport = NebiusTransport(session=session)
        await transport.fetch_models()

        # ------------------------------------------------------------------
        # Model selection — edit here to change which model each role uses.
        #
        # transport.models is a ModelRegistry populated from the Nebius API.
        # Use .search("substring") to find models, then .first() to pick one.
        #
        # Examples:
        #   transport.models.search("DeepSeek-R1").first()      # reasoning
        #   transport.models.search("Llama-3.3-70B").first()    # open-weights
        #   transport.models.search("Qwen2.5-72B").first()      # multilingual
        # ------------------------------------------------------------------
        role_models: dict[str, ModelSpec] = {
            "default": transport.models["MiniMaxAI/MiniMax-M2.5"],
            "architect": transport.models["Qwen/Qwen3-235B-A22B-Instruct-2507"],
            "security_engineer": transport.models["openai/gpt-oss-120b"],
            "project_manager": transport.models["openai/gpt-oss-120b"],
            "challenger": transport.models["zai-org/GLM-5"],
            # analyst runs many instances in parallel
            "analyst": transport.models["deepseek-ai/DeepSeek-V3.2"],
        }

        console.print()
        console.print(Rule("[bold]Agent Swarm[/bold]"))
        console.print(f"[dim]Task:[/dim]          {args.task}")
        console.print(f"[dim]Workspace:[/dim]     {workspace}")
        console.print(f"[dim]Default model:[/dim] {role_models['default'].id}")
        console.print()

        try:
            with renderer:
                await run_swarm(
                    task=args.task,
                    workspace=workspace,
                    on_event=renderer.on_event,
                    transport=transport,
                    role_models=role_models,
                    guard_factory=renderer.make_guard,
                    prompt_fn=renderer.make_prompt_fn(),
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(1)

    renderer.print_workspace_tree(workspace)


def cli() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    cli()
