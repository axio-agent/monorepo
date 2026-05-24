"""Interactive REPL coding assistant powered by axio agent framework.

Auto-detects transport from available API keys (OPENAI_API_KEY, NEBIUS_API_KEY,
OPENROUTER_API_KEY), or use --transport to pick explicitly.

Run:
    axio-repl
    axio-repl "your prompt here"
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import dataclasses
import os
import re
import signal
import sys
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, NamedTuple, cast

import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import (
    AudioOutput,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    SessionEndEvent,
    StreamEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolInputDelta,
    ToolOutputDelta,
    ToolResult,
    ToolUseStart,
    VideoOutput,
)
from axio.field import StrictStr
from axio.models import Capability, ModelSpec
from axio.tool import Tool
from axio.tool_args import ToolArgStream
from axio_tools_agents.peers import (
    PeerMessage,
    PeerServer,
    enqueue_local_agent_prompt,
    format_message_for_dialog,
    interrupt_agent,
    is_local_background_agent,
    list_peers,
    local_background_agent_records,
    send_message,
    set_agent_event_handler,
    set_spawn_agent_factory,
    spawn_agent,
    stop_agent,
    stop_local_background_agents,
    wait_local_background_agents_idle,
)
from axio_tools_local.list_files import list_files
from axio_tools_local.patch_file import patch_file
from axio_tools_local.read_file import read_file
from axio_tools_local.shell import shell
from axio_tools_local.write_file import write_file

_readline: Any
try:
    import readline as _readline
except ImportError:
    _readline = None

readline: Any = _readline

AGENT_NAME = "axio-repl"
AGENT_VERSION = "0.2.3"

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


# ── Custom search tool ───────────────────────────────────────────────


async def search_files(
    query: StrictStr,
    path: StrictStr = ".",
    regex: bool = False,
    max_results: int = 100,
) -> str:
    """Search for text or regex patterns in files under a directory.
    Returns matching lines with file paths and line numbers."""

    def _search() -> str:
        base = Path(path).resolve()
        if not base.exists():
            return f"error: path not found: {path}"

        try:
            pattern = re.compile(query) if regex else None
        except re.error as exc:
            return f"error: invalid regex: {exc}"

        skip = {".git", ".venv", "__pycache__", "node_modules"}
        matches: list[str] = []

        files = [base] if base.is_file() else list(_iter_files(base, skip))
        for file_path in files:
            if len(matches) >= max_results:
                break
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                found = pattern.search(line) if pattern else (query in line)
                if found:
                    matches.append(f"{file_path}:{idx}: {line}")
                    if len(matches) >= max_results:
                        break

        if not matches:
            return f"No matches for {query!r}"
        return "\n".join(matches)

    return await asyncio.to_thread(_search)


def _iter_files(base: Path, skip: set[str]) -> Iterator[Path]:
    for current_dir, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for name in sorted(files):
            if not name.startswith("."):
                yield Path(current_dir) / name


# ── Tools ────────────────────────────────────────────────────────────

TOOLS: list[Tool[Any]] = [
    Tool(name="read_file", handler=read_file),
    Tool(name="write_file", handler=write_file),
    Tool(name="patch_file", handler=patch_file),
    Tool(name="list_files", handler=list_files),
    Tool(name="search_files", handler=search_files),
    Tool(name="shell", handler=shell),
    Tool(name="interrupt_agent", handler=interrupt_agent),
    Tool(name="list_peers", handler=list_peers),
    Tool(name="send_message", handler=send_message),
    Tool(name="spawn_agent", handler=spawn_agent, concurrency=3),
    Tool(name="stop_agent", handler=stop_agent),
]


# ── Transport auto-detection ─────────────────────────────────────────


def _discover_transports() -> dict[str, Callable[..., Any]]:
    result: dict[str, Callable[..., Any]] = {}
    for ep in entry_points(group="axio.transport"):
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


_TRANSPORT_ENV_VARS: dict[str, list[str]] = {
    "google": ["GEMINI_API_KEY"],
    "google-vertex": ["GOOGLE_GENAI_USE_VERTEXAI"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "nebius": ["NEBIUS_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _transport_has_credentials(name: str) -> bool:
    env_vars = _TRANSPORT_ENV_VARS.get(name, [])
    return any(os.environ.get(v, "") for v in env_vars)


def _select_transport(name: str | None) -> tuple[Callable[..., Any], str]:
    available = _discover_transports()
    if name:
        if name not in available:
            print(
                f"Unknown transport {name!r}. Available: {', '.join(sorted(available))}",
                file=sys.stderr,
            )
            sys.exit(1)
        return available[name], ""

    for transport_name, cls in available.items():
        if _transport_has_credentials(transport_name):
            return cls, ""

    print("No API key found. Set one of:", file=sys.stderr)
    for transport_name in available:
        env_vars = _TRANSPORT_ENV_VARS.get(transport_name, [])
        if env_vars:
            print(f"  {', '.join(env_vars)}  ({transport_name})", file=sys.stderr)
    sys.exit(1)


# ── AGENTS.md & system prompt ────────────────────────────────────────


def load_agents_instructions(root: Path) -> str:
    agents_file = root / "AGENTS.md"
    if not agents_file.exists():
        return ""
    try:
        return agents_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def build_system_prompt(
    root: Path,
    model: ModelSpec,
    tools: list[Tool[Any]],
    agents_text: str = "",
) -> str:
    caps = model.capabilities
    ctx_k = model.context_window // 1000
    out_k = model.max_output_tokens // 1000
    tool_names = ", ".join(t.name for t in tools)

    has_tools = Capability.tool_use in caps

    lines = [
        f"You are {AGENT_NAME} (v{AGENT_VERSION}) — a terminal coding assistant.",
        f"Model: {model.id} ({ctx_k}K context, {out_k}K max output)",
        f"Current directory: {root} (perform actions here unless specified otherwise)",
    ]
    if has_tools:
        lines.append(f"Tools: {tool_names}")
    lines.append("")

    # Capability-aware guidance
    cap_notes: list[str] = []
    if Capability.vision in caps:
        cap_notes.append("You can see images via read_file (screenshots, diagrams, photos).")
    if Capability.audio in caps:
        cap_notes.append("You can listen to audio files via read_file (speech, music, podcasts).")
    if Capability.video in caps:
        cap_notes.append("You can see video files via read_file.")
    if Capability.image_generation in caps:
        cap_notes.append("You can generate images inline — describe what to draw in your response.")
    if Capability.reasoning in caps:
        cap_notes.append("Extended thinking is available for complex reasoning.")
    if cap_notes:
        lines += cap_notes + [""]

    lines.append("Rules:")
    if has_tools:
        lines += [
            "- Start every task by listing the current directory to understand the project.",
            "- Read files before editing. Use line_numbers=True before patch_file.",
            "- Keep edits minimal and targeted — don't reformat surrounding code.",
            "- Ground answers on project context gathered through tools.",
        ]
    lines += [
        "- Write idiomatic code — follow the conventions and best practices of the "
        "languages and frameworks used in the project.",
        "- When the user asks about a file they provided or you read, base your answer "
        "strictly on the actual file contents. Do not guess, assume, or fill in details "
        "from general knowledge — only state what the file actually contains.",
        f"- Your max output is {out_k}K tokens. Use as many as the task requires — "
        "do not stop early. If the user asks for a full transcript, detailed analysis, "
        f"or comprehensive review, produce the complete output up to the {out_k}K limit.",
        "- Never refuse safe requests or claim inability.",
    ]
    if has_tools:
        lines += [
            "- If a tool call fails, analyze the error and try a different approach. "
            "If stuck after 3 attempts at the same sub-problem, "
            "explain what you tried and ask for guidance.",
            "- Do not return a final answer until all necessary work is done or you are stuck.",
            "- For compound requests, build a checklist of all items and verify each is addressed before finishing.",
            "- Don't narrate your tool calls — the user sees their full output.",
            "- After completing work, summarize what changed briefly.",
            "- Not tested — not done. Always run tests or builds to verify your changes. "
            "Re-read edited files, observe actual results — don't assume success "
            "from exit codes alone.",
            "- After any test or build that produces images or video, you MUST read_file "
            "every output file to actually see the results. Never describe visual output "
            "you haven't viewed. 'Tests passed' is not the same as 'I looked at the "
            "screenshots and they look correct'.",
            "- To verify UI, use browser automation (Playwright, Puppeteer) to capture "
            "real screenshots at multiple viewport sizes (desktop 1280×800, tablet 768×1024, "
            "mobile 375×667), then read_file every screenshot.",
            "- When you read a screenshot, you MUST critically analyze it. List every "
            "visual defect you notice: broken layout, text overflow, misaligned elements, "
            "poor contrast, missing images, clipped content, wrong spacing, responsive "
            "issues. Do NOT say 'looks good' unless you can specifically confirm each "
            "aspect is correct.",
            "- UI review is iterative: screenshot → list issues → fix code → re-screenshot "
            "→ verify fixes. Repeat until zero defects. Never declare UI done after a "
            "single screenshot pass.",
            "- Never use generate_image as a substitute for real UI testing.",
            "- Never run destructive shell commands (rm -rf, git reset --hard) without user confirmation.",
            "- For large files, read specific line ranges instead of the entire file.",
        ]
        if any(t.name == "spawn_agent" for t in tools):
            lines += [
                "- Use spawn_agent for independent parallel work that can run in the background. It returns "
                "immediately with a global agent_id; it does not return the child agent's final answer.",
                "- Spawned agents start with empty context by default. Set inherit_context=true only when the child "
                "must see the current conversation; otherwise include all required instructions in task.",
                "- When spawning a child that must report back, tell it exactly how to deliver results: either "
                "send_message(agent_id=<parent_id>, message=<report>) to the parent when done, or wait for the "
                "REPL to focus/follow the child and print the child's final response. Do not tell the user that "
                "results will appear later unless you have arranged one of those delivery paths.",
                "- Use list_peers() to discover running agents in this project, or list_peers(all_projects=true) "
                "to inspect all local agent ids. Use send_message(agent_id=..., message=...) for IPC by global id.",
                "- Use interrupt_agent(agent_id=...) to cancel a spawned agent's current response while keeping it "
                "alive. Use stop_agent(agent_id=...) only when the child should exit. A parent may interrupt or "
                "stop its own children by id.",
                "- In axio-repl, the user can switch the active local agent with /agent-focus, list them with "
                "/agents, interrupt with /agent-interrupt, and stop with /agent-stop. Only the focused agent "
                "streams fully; background agents are summarized between focused streams.",
            ]
    lines.append("")

    if agents_text:
        lines += ["AGENTS.md instructions:", agents_text, ""]

    return "\n".join(lines)


# ── Readline history ─────────────────────────────────────────────────


def setup_history() -> None:
    if readline is None:
        return
    history_path = Path.home() / ".axio_repl_history"
    if history_path.exists():
        try:
            readline.read_history_file(str(history_path))
        except (OSError, RuntimeError):
            pass
    try:
        readline.set_history_length(5000)
    except (AttributeError, ValueError):
        pass

    def _save() -> None:
        try:
            readline.write_history_file(str(history_path))
        except (OSError, RuntimeError):
            pass

    atexit.register(_save)


# ── Event rendering ──────────────────────────────────────────────────


class _AgentRenderState:
    def __init__(self) -> None:
        self.in_text = False
        self.arg_streams: dict[str, ToolArgStream] = {}
        self.streamed_tool_ids: set[str] = set()
        self.field_first_delta = True
        self.field_key: str | None = None
        self.background_text_chars = 0
        self.background_tools: list[str] = []
        self.background_errors: list[str] = []
        self.background_events: list[StreamEvent] = []


class ReplRenderer:
    def __init__(self, *, buffer_background_events: bool = False) -> None:
        self._lock = asyncio.Lock()
        self._buffer_background_events = buffer_background_events
        self._states: dict[str, _AgentRenderState] = {}
        self._active_agent: str | None = None
        self._focused_agent = "main"
        self._foreground_streaming = False
        self._background_pending: set[str] = set()
        self._input_active = False
        self._input_interrupted = False
        self._redraw_input = False

    @property
    def focused_agent(self) -> str:
        return self._focused_agent

    def set_focus(self, agent_id: str) -> None:
        self._focused_agent = agent_id
        state = self._state(agent_id)
        buffered = state.background_events
        state.background_events = []
        state.background_text_chars = 0
        state.background_tools.clear()
        state.background_errors.clear()
        self._background_pending.discard(agent_id)
        for event in buffered:
            self._render_locked(agent_id, event)
            if isinstance(event, Error | SessionEndEvent):
                self._foreground_streaming = False
            elif not isinstance(event, IterationEnd):
                self._foreground_streaming = True

    def set_input_active(self, active: bool) -> None:
        self._input_active = active
        if not active:
            self._input_interrupted = False
            self._redraw_input = False

    async def render(self, agent_id: str, event: StreamEvent) -> None:
        async with self._lock:
            if agent_id == self._focused_agent:
                self._render_locked(agent_id, event)
                if isinstance(event, Error | SessionEndEvent):
                    self._foreground_streaming = False
                    self._flush_background_summaries_locked()
                elif not isinstance(event, IterationEnd):
                    self._foreground_streaming = True
            else:
                self._record_background_event_locked(agent_id, event)

    async def mark_idle(self) -> None:
        async with self._lock:
            self._foreground_streaming = False
            self._flush_background_summaries_locked()

    async def notice(self, text: str) -> None:
        async with self._lock:
            self._prepare_input_output()
            if self._active_agent is not None and self._state(self._active_agent).in_text:
                print()
                self._state(self._active_agent).in_text = False
            print(f"{DIM}{text}{RESET}")

    def _state(self, agent_id: str) -> _AgentRenderState:
        return self._states.setdefault(agent_id, _AgentRenderState())

    def _prepare_input_output(self) -> None:
        if not self._input_active:
            return
        if not self._input_interrupted:
            print()
            self._input_interrupted = True
        self._redraw_input = True

    def _switch_agent(self, agent_id: str) -> _AgentRenderState:
        switched = self._active_agent != agent_id
        if switched:
            if self._active_agent is not None and self._state(self._active_agent).in_text:
                print()
                self._state(self._active_agent).in_text = False
            if self._active_agent is not None or agent_id != "main":
                print(f"\n{DIM}── {agent_id} ──{RESET}")
            self._active_agent = agent_id
        state = self._state(agent_id)
        if switched and state.field_key is not None:
            sys.stdout.write(f"\n  {YELLOW}{state.field_key} (continued){RESET}: {DIM}")
            sys.stdout.flush()
            state.field_first_delta = True
        return state

    def _render_locked(self, agent_id: str, event: StreamEvent) -> None:  # noqa: C901
        if not isinstance(event, IterationEnd):
            self._prepare_input_output()
        state = self._switch_agent(agent_id)
        match event:
            case ReasoningDelta(delta=delta):
                if state.in_text:
                    print()
                    state.in_text = False
                sys.stdout.write(f"{DIM}> {delta}{RESET}")
                sys.stdout.flush()

            case TextDelta(delta=delta):
                if not state.in_text:
                    state.in_text = True
                if "[Output truncated:" in delta:
                    sys.stdout.write(f"\n{RED}{delta.strip()}{RESET}\n")
                    state.in_text = False
                else:
                    sys.stdout.write(delta)
                sys.stdout.flush()

            case ImageOutput(data=data, media_type=mt):
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[image saved: {path}]{RESET}")

            case AudioOutput(data=data, media_type=mt):
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[audio saved: {path}]{RESET}")

            case VideoOutput(data=data, media_type=mt):
                if state.in_text:
                    print()
                    state.in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[video saved: {path}]{RESET}")

            case ToolUseStart(index=index, tool_use_id=tid, name=name):
                if state.in_text:
                    print()
                    state.in_text = False
                sys.stdout.write(f"\n{BOLD}{CYAN}\u25b6 {name}{RESET}")
                sys.stdout.flush()
                state.arg_streams[tid] = ToolArgStream(tid, index)

            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                stream = state.arg_streams.get(tid)
                if stream:
                    for fe in stream.feed(pj):
                        self._render_field_event(state, fe)
                    if stream.done:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        del state.arg_streams[tid]

            case ToolOutputDelta(tool_use_id=tid, key=key, delta=delta):
                if tid not in state.streamed_tool_ids:
                    sys.stdout.write("\n")
                state.streamed_tool_ids.add(tid)
                color = RED if key == "stderr" else DIM
                sys.stdout.write(f"{color}{delta}{RESET}")
                sys.stdout.flush()

            case ToolResult(tool_use_id=tid, name=name, is_error=is_error, content=content):
                if is_error:
                    sys.stdout.write(f"{RESET}\n{RED}{content}{RESET}\n")
                elif name == "spawn_agent":
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                elif tid in state.streamed_tool_ids:
                    sys.stdout.write(f"{RESET}\n")
                else:
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                sys.stdout.flush()

            case IterationEnd():
                pass

            case Error(exception=exc):
                print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)

            case SessionEndEvent(total_usage=usage):
                if state.in_text:
                    print()
                    state.in_text = False
                print(f"{DIM}[{usage.input_tokens}in/{usage.output_tokens}out tokens]{RESET}")
                if self._redraw_input and self._input_active and readline is not None:
                    readline.redisplay()
                self._redraw_input = False

    def _record_background_event_locked(self, agent_id: str, event: StreamEvent) -> None:
        state = self._state(agent_id)
        if self._buffer_background_events:
            state.background_events.append(event)
        match event:
            case TextDelta(delta=delta):
                state.background_text_chars += len(delta)
            case ToolUseStart(name=name):
                if name not in state.background_tools:
                    state.background_tools.append(name)
            case Error(exception=exc):
                state.background_errors.append(str(exc))
                if not self._buffer_background_events:
                    self._background_pending.add(agent_id)
                    if not self._foreground_streaming:
                        self._flush_background_summaries_locked()
            case SessionEndEvent():
                if not self._buffer_background_events:
                    self._background_pending.add(agent_id)
                    if not self._foreground_streaming:
                        self._flush_background_summaries_locked()
            case _:
                pass

    def _flush_background_summaries_locked(self) -> None:
        if not self._background_pending:
            return
        self._prepare_input_output()
        for agent_id in sorted(self._background_pending):
            state = self._state(agent_id)
            parts = [f"{agent_id} completed"]
            if state.background_tools:
                parts.append(f"tools={','.join(state.background_tools)}")
            if state.background_text_chars:
                parts.append(f"text={state.background_text_chars} chars")
            if state.background_errors:
                parts.append(f"errors={len(state.background_errors)}")
            print(f"{DIM}[background {'; '.join(parts)}]{RESET}")
            state.background_text_chars = 0
            state.background_tools.clear()
            state.background_errors.clear()
        self._background_pending.clear()
        if self._redraw_input and self._input_active and readline is not None:
            readline.redisplay()
        self._redraw_input = False

    def _render_field_event(
        self,
        state: _AgentRenderState,
        event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd,
    ) -> None:
        match event:
            case ToolFieldStart(key=key):
                sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
                sys.stdout.flush()
                state.field_key = key
                state.field_first_delta = True
            case ToolFieldDelta(text=text):
                if state.field_first_delta and "\n" in text:
                    sys.stdout.write("\n")
                state.field_first_delta = False
                sys.stdout.write(text)
                sys.stdout.flush()
            case ToolFieldEnd():
                sys.stdout.write(RESET)
                sys.stdout.flush()
                state.field_key = None


async def run_prompt(agent: Agent, ctx: MemoryContextStore, prompt: str, renderer: ReplRenderer) -> None:
    async for event in agent.run_stream(prompt, ctx):
        await renderer.render("main", event)


def _clone_transport_for_spawn(transport: Any) -> Any:
    clone = copy.copy(transport)
    if dataclasses.is_dataclass(clone):
        for field_info in dataclasses.fields(clone):
            value = getattr(clone, field_info.name)
            if isinstance(value, dict):
                setattr(clone, field_info.name, dict(value))
            elif isinstance(value, list):
                setattr(clone, field_info.name, list(value))
            elif isinstance(value, set):
                setattr(clone, field_info.name, set(value))
    if hasattr(clone, "last_usage"):
        setattr(clone, "last_usage", None)
    if hasattr(clone, "_thought_signatures"):
        setattr(clone, "_thought_signatures", {})
    return clone


_media_counter = 0


def _save_media(data: bytes, media_type: str) -> str:
    """Save media bytes to a temp file, return the path."""
    import tempfile

    global _media_counter
    _media_counter += 1
    ext = media_type.split("/")[-1].split(";")[0]
    fd, path = tempfile.mkstemp(suffix=f".{ext}", prefix=f"axio_{_media_counter:03d}_")
    os.write(fd, data)
    os.close(fd)
    return path


# ── Input handling ───────────────────────────────────────────────────


def _read_input() -> str:
    """Read user input, collecting extra lines from a multiline paste."""
    import select

    first = input("repl> ")
    lines = [first]
    fd = sys.stdin.fileno()
    while select.select([fd], [], [], 0.05)[0]:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        extra = chunk.decode(errors="replace").splitlines()
        for line in extra:
            print(f"  ... {line}")
        lines.extend(extra)
    return "\n".join(lines).strip()


async def _read_input_async(loop: asyncio.AbstractEventLoop, renderer: ReplRenderer) -> str:
    renderer.set_input_active(True)
    try:
        return await loop.run_in_executor(None, _read_input)
    finally:
        renderer.set_input_active(False)


def _peer_name(root: Path) -> str:
    return f"axio-repl:{root.name}:{os.getpid()}"


def _resolve_local_agent_id(value: str) -> str | None:
    if value == "main":
        return "main"
    records = local_background_agent_records()
    exact = [record.id for record in records if record.id == value or record.name == value]
    if len(exact) == 1:
        return exact[0]
    prefixed = [record.id for record in records if record.id.startswith(value)]
    if len(prefixed) == 1:
        return prefixed[0]
    return None


def _resolve_command_agent_id(value: str, renderer: ReplRenderer) -> str | None:
    if value == "current":
        return renderer.focused_agent
    return _resolve_local_agent_id(value)


def _show_agents(renderer: ReplRenderer) -> None:
    print(f"Focused agent: {BOLD}{renderer.focused_agent}{RESET}")
    records = local_background_agent_records()
    if not records:
        print("No local background agents.")
        return
    for record in records:
        marker = "*" if record.id == renderer.focused_agent else " "
        print(f"{marker} {record.id} name={record.name!r} kind={record.kind} pid={record.pid}")


# ── REPL commands ────────────────────────────────────────────────────


class Command(NamedTuple):
    """A REPL command with separate show (no arg) and apply (with arg) modes."""

    show: Callable[[], None]
    apply: Callable[[str], None]


# CLI arg attr → slash command name (for unified init).
_CLI_TO_SLASH: dict[str, str] = {
    "thinking": "/thinking",
    "temperature": "/temperature",
    "max_tokens": "/max-tokens",
    "debug": "/debug",
}


def _apply_cli_args(args: object, commands: dict[str, Command]) -> None:
    """Apply CLI arguments through the same command handlers as slash commands."""
    for attr, cmd_name in _CLI_TO_SLASH.items():
        val: Any = getattr(args, attr, None)
        if val is None or val is False:
            continue
        arg = "on" if isinstance(val, bool) else val if isinstance(val, str) else str(val)
        commands[cmd_name].apply(arg)


# ── model ──


def _show_model(transport: Any) -> None:
    model = transport.model
    caps = ", ".join(sorted(c.value for c in model.capabilities))
    print(f"Current model: {BOLD}{model.id}{RESET}")
    print(f"Capabilities: {caps}")
    print(f"Available: {', '.join(transport.models.keys())}")


def _resolve_model_arg(transport: Any, model_id: str) -> ModelSpec:
    resolver = getattr(transport, "resolve_model", None)
    if callable(resolver):
        return cast(ModelSpec, resolver(model_id))
    return cast(ModelSpec, transport.models[model_id])


def _apply_model(
    transport: Any,
    agent: Agent,
    tools: list[Tool[Any]],
    root: Path,
    agents_text: str,
    arg: str,
) -> None:
    try:
        transport.model = _resolve_model_arg(transport, arg)
    except KeyError:
        matches = transport.models.search(arg)
    else:
        agent.system = build_system_prompt(root, transport.model, tools, agents_text)
        print(f"Switched to {BOLD}{transport.model.id}{RESET}")
        return

    if len(matches) == 1:
        transport.model = next(iter(matches.values()))
        agent.system = build_system_prompt(root, transport.model, tools, agents_text)
        print(f"Switched to {BOLD}{transport.model.id}{RESET}")
    elif len(matches) == 0:
        print(f"No model matching {arg!r}. Available: {', '.join(transport.models.keys())}")
    else:
        print(f"Ambiguous — matches: {', '.join(matches.keys())}")


# ── thinking ──


def _show_thinking(transport: Any) -> None:
    level = getattr(transport, "thinking_level", None)
    budget = getattr(transport, "thinking_budget", None)
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if level:
        print(f"Thinking level: {BOLD}{level}{RESET}")
    elif budget is not None:
        print(f"Thinking budget: {BOLD}{budget}{RESET} tokens")
    else:
        print("Thinking: default")
    if valid_levels is not None:
        print(f"Valid levels: {', '.join(valid_levels)}")
    elif get_opts is not None:
        print("Usage: /thinking <budget_tokens>")


def _apply_thinking(transport: Any, arg: str) -> None:
    get_opts = getattr(transport, "get_thinking_options", None)
    valid_levels = get_opts() if get_opts else None
    if arg.isdigit():
        if valid_levels is not None:
            model_id = getattr(getattr(transport, "model", None), "id", "?")
            print(f"{model_id} uses thinking levels, not token budgets.")
            print(f"Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_budget = int(arg)
        transport.thinking_level = None
        print(f"Thinking budget: {BOLD}{arg}{RESET} tokens")
    else:
        name = arg.upper()
        if valid_levels is not None and name not in valid_levels:
            print(f"{name} is not valid. Valid levels: {', '.join(valid_levels)}")
            return
        transport.thinking_level = name
        transport.thinking_budget = None
        print(f"Thinking level: {BOLD}{name}{RESET}")


# ── temperature ──


def _show_temperature(transport: Any) -> None:
    temp = getattr(transport, "temperature", None)
    print(f"Temperature: {BOLD}{temp if temp is not None else 'default'}{RESET}")


def _apply_temperature(transport: Any, arg: str) -> None:
    try:
        val = float(arg)
    except ValueError:
        print(f"Invalid temperature: {arg!r}")
        return
    if hasattr(transport, "temperature"):
        transport.temperature = val
        print(f"Temperature: {BOLD}{val}{RESET}")
    else:
        print("Transport does not support temperature")


# ── iterations ──


def _show_iterations(agent: Agent) -> None:
    print(f"Max iterations: {BOLD}{agent.max_iterations}{RESET}")


def _apply_iterations(agent: Agent, arg: str) -> None:
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    agent.max_iterations = val
    print(f"Max iterations: {BOLD}{val}{RESET}")


# ── max-tokens ──


def _show_max_tokens(transport: Any) -> None:
    cur = getattr(transport, "max_output_tokens", None)
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if cur:
        print(f"Max output tokens: {BOLD}{cur}{RESET} (model default: {model_default})")
    else:
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")


def _apply_max_tokens(transport: Any, arg: str) -> None:
    model_default = getattr(getattr(transport, "model", None), "max_output_tokens", None)
    if arg == "default":
        transport.max_output_tokens = None
        print(f"Max output tokens: {BOLD}{model_default}{RESET} (model default)")
        return
    try:
        val = int(arg)
    except ValueError:
        print(f"Invalid value: {arg!r}")
        return
    transport.max_output_tokens = val
    print(f"Max output tokens: {BOLD}{val}{RESET}")


# ── debug ──


def _show_debug(transport: Any) -> None:
    cur = getattr(transport, "debug", False)
    print(f"Debug: {BOLD}{'on' if cur else 'off'}{RESET}")


def _apply_debug(transport: Any, arg: str) -> None:
    val = arg.lower()
    if val == "on":
        transport.debug = True
        print(f"Debug: {BOLD}on{RESET} (request/response bodies logged to stderr)")
    elif val == "off":
        transport.debug = False
        print(f"Debug: {BOLD}off{RESET}")
    else:
        print("Usage: /debug on|off")


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="REPL coding assistant (axio)")
    parser.add_argument("prompt", nargs="?", default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--transport", default=None, help="Transport name (auto-detected if omitted)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--thinking", default=None, help="Thinking level or token budget (integer)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max output tokens")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--debug", action="store_true", help="Log request/response bodies to stderr")
    args = parser.parse_args()

    transport_cls, _ = _select_transport(args.transport)
    root = Path.cwd().resolve()
    agents_text = load_agents_instructions(root)
    setup_history()

    async with aiohttp.ClientSession() as session:
        transport = transport_cls(session=session)
        await transport.fetch_models()

        if args.model:
            try:
                transport.model = _resolve_model_arg(transport, args.model)
            except KeyError:
                available_models = ", ".join(transport.models.keys())
                print(f"No model matching {args.model!r}. Available: {available_models}", file=sys.stderr)
                sys.exit(1)

        # Transport-level commands (available before agent creation).
        commands: dict[str, Command] = {
            "/thinking": Command(lambda: _show_thinking(transport), lambda a: _apply_thinking(transport, a)),
            "/temperature": Command(lambda: _show_temperature(transport), lambda a: _apply_temperature(transport, a)),
            "/max-tokens": Command(lambda: _show_max_tokens(transport), lambda a: _apply_max_tokens(transport, a)),
            "/debug": Command(lambda: _show_debug(transport), lambda a: _apply_debug(transport, a)),
        }
        _apply_cli_args(args, commands)

        tools = list(TOOLS)
        system = build_system_prompt(root, transport.model, tools, agents_text)
        agent = Agent(
            system=system,
            tools=tools,
            transport=transport,
            max_iterations=args.max_iterations,
        )
        ctx = MemoryContextStore()

        async def _make_spawn_agent(inherit_context: bool) -> tuple[Agent, MemoryContextStore]:
            child_ctx = await MemoryContextStore.from_context(ctx) if inherit_context else MemoryContextStore()
            child_transport = _clone_transport_for_spawn(agent.transport)
            child_tools = [
                Tool(
                    name=t.name,
                    description=t.description,
                    handler=t.handler,
                    guards=t.guards,
                    context=t.context,
                    concurrency=t.concurrency,
                )
                for t in agent.tools
                if t.name != "spawn_agent"
            ]
            child_system = build_system_prompt(root, child_transport.model, child_tools, agents_text)
            return agent.copy(
                transport=child_transport,
                system=child_system,
                tools=child_tools,
                max_iterations=min(agent.max_iterations, 25),
            ), child_ctx

        set_spawn_agent_factory(_make_spawn_agent)

        # Agent-dependent commands.
        commands["/model"] = Command(
            lambda: _show_model(transport),
            lambda a: _apply_model(transport, agent, tools, root, agents_text, a),
        )
        commands["/iterations"] = Command(
            lambda: _show_iterations(agent),
            lambda a: _apply_iterations(agent, a),
        )

        loop = asyncio.get_event_loop()
        renderer = ReplRenderer(buffer_background_events=args.prompt is not None)
        set_agent_event_handler(renderer.render)
        prompt_task: asyncio.Task[None] | None = None
        input_task: asyncio.Task[str] | None = None
        inbox_task: asyncio.Task[str] | None = None
        peer_queue: asyncio.Queue[str] = asyncio.Queue()
        peer_server: PeerServer | None = None

        async def _on_peer_message(message: PeerMessage) -> None:
            await peer_queue.put(format_message_for_dialog(message))

        async def _run_turn(prompt: str) -> None:
            nonlocal prompt_task
            prompt_task = asyncio.create_task(run_prompt(agent, ctx, prompt, renderer))
            try:
                await prompt_task
            except asyncio.CancelledError:
                print(f"\n{DIM}[interrupted]{RESET}")
            finally:
                prompt_task = None
                await renderer.mark_idle()

        async def _interrupt_focused_agent(agent_id: str) -> None:
            await interrupt_agent(agent_id, reason="SIGINT")
            await renderer.notice("[interrupted]")
            await renderer.mark_idle()

        async def _run_one_shot_background_agents() -> None:
            records = local_background_agent_records()
            if not records:
                return
            if len(records) == 1:
                renderer.set_focus(records[0].id)
                await renderer.notice(f"[following background agent {records[0].id}]")
            else:
                await renderer.notice(f"[waiting for {len(records)} background agents]")
                renderer.set_focus(records[0].id)
            await wait_local_background_agents_idle([record.id for record in records])
            for record in records[1:]:
                renderer.set_focus(record.id)
            await renderer.mark_idle()

        def _on_sigint() -> None:
            nonlocal prompt_task
            if renderer.focused_agent != "main":
                asyncio.create_task(_interrupt_focused_agent(renderer.focused_agent))
            elif prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        try:
            if args.prompt:
                await _run_turn(args.prompt)
                await _run_one_shot_background_agents()
                return

            try:
                peer_server = await PeerServer(
                    _peer_name(root),
                    kind="axio-repl",
                    handler=_on_peer_message,
                    cwd=str(root),
                ).start()
            except OSError as exc:
                print(f"{DIM}[peer messaging disabled: {exc}]{RESET}")

            agent_commands = ["/agents", "/agent-focus", "/agent-interrupt", "/agent-stop"]
            commands_list = ", ".join(["/help", *commands, *agent_commands, "/quit"])
            label = getattr(transport, "name", "unknown")
            print(f"REPL ready ({label}). Commands: {commands_list}")

            while True:
                if input_task is None:
                    input_task = asyncio.create_task(_read_input_async(loop, renderer))
                if inbox_task is None:
                    inbox_task = asyncio.create_task(peer_queue.get())

                done, _ = await asyncio.wait(
                    {input_task, inbox_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if inbox_task in done:
                    peer_prompt = inbox_task.result()
                    inbox_task = None
                    await renderer.notice("[peer message queued]")
                    await _run_turn(peer_prompt)
                    continue

                if input_task not in done:
                    continue

                try:
                    user_input = input_task.result()
                except EOFError:
                    print()
                    break
                finally:
                    input_task = None

                if not user_input:
                    continue
                lowered = user_input.lower()
                if lowered in {"/quit", "/exit", "/q"}:
                    break
                if lowered == "/help":
                    tool_list = ", ".join(t.name for t in tools)
                    print(f"Type your request. Tools: {tool_list}")
                    print(f"Commands: {commands_list}")
                    continue
                if lowered == "/agents":
                    _show_agents(renderer)
                    continue
                if lowered == "/agent-focus" or lowered.startswith("/agent-focus "):
                    arg = user_input[len("/agent-focus") :].strip()
                    if not arg:
                        _show_agents(renderer)
                        continue
                    agent_id = _resolve_local_agent_id(arg)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    renderer.set_focus(agent_id)
                    print(f"Focused agent: {BOLD}{agent_id}{RESET}")
                    continue
                if lowered == "/agent-stop" or lowered.startswith("/agent-stop "):
                    arg = user_input[len("/agent-stop") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    if agent_id == "main":
                        print("Use /quit to exit the main REPL agent.")
                        continue
                    print(await stop_agent(agent_id, reason="user requested stop"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                        renderer.set_focus("main")
                        print(f"Focused agent: {BOLD}main{RESET}")
                    continue
                if lowered == "/agent-interrupt" or lowered.startswith("/agent-interrupt "):
                    arg = user_input[len("/agent-interrupt") :].strip() or renderer.focused_agent
                    agent_id = _resolve_command_agent_id(arg, renderer)
                    if agent_id is None:
                        print(f"No local agent matching {arg!r}")
                        continue
                    if agent_id == "main":
                        print("Press Ctrl-C while the main agent is streaming to interrupt it.")
                        continue
                    print(await interrupt_agent(agent_id, reason="user requested interrupt"))
                    if renderer.focused_agent == agent_id:
                        await renderer.mark_idle()
                    continue

                matched = False
                for prefix, cmd in commands.items():
                    if lowered == prefix or lowered.startswith(prefix + " "):
                        cmd_arg = user_input[len(prefix) :].strip() or None
                        if cmd_arg is None:
                            cmd.show()
                        else:
                            cmd.apply(cmd_arg)
                        matched = True
                        break
                if matched:
                    continue

                if renderer.focused_agent == "main":
                    await _run_turn(user_input)
                elif is_local_background_agent(renderer.focused_agent):
                    delivered = await enqueue_local_agent_prompt(renderer.focused_agent, user_input, wait=True)
                    await renderer.mark_idle()
                    if not delivered:
                        print(f"Agent {renderer.focused_agent!r} is no longer running.")
                        renderer.set_focus("main")
                else:
                    print(f"Agent {renderer.focused_agent!r} is no longer local; focusing main.")
                    renderer.set_focus("main")
        finally:
            for task in (input_task, inbox_task):
                if task is not None and not task.done():
                    task.cancel()
            await stop_local_background_agents()
            if peer_server is not None:
                await peer_server.close()
            set_agent_event_handler(None)
            set_spawn_agent_factory(None)
            loop.remove_signal_handler(signal.SIGINT)


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()
