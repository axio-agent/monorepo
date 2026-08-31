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
    Citation,
    Error,
    ImageOutput,
    IterationEnd,
    ReasoningDelta,
    Refusal,
    SessionEndEvent,
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
from axio.types import INCOMPLETE
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


async def run_prompt(agent: Agent, ctx: MemoryContextStore, prompt: str) -> None:
    in_text = False
    declined = False
    arg_streams: dict[str, ToolArgStream] = {}
    streamed_tool_ids: set[str] = set()

    async for event in agent.run_stream(prompt, ctx):
        match event:
            case ReasoningDelta(delta=delta):
                if in_text:
                    print()
                    in_text = False
                sys.stdout.write(f"{DIM}> {delta}{RESET}")
                sys.stdout.flush()

            case TextDelta(delta=delta):
                if not in_text:
                    in_text = True
                if "[Output truncated:" in delta:
                    sys.stdout.write(f"\n{RED}{delta.strip()}{RESET}\n")
                    in_text = False
                else:
                    sys.stdout.write(delta)
                sys.stdout.flush()

            case Refusal(text=text, category=category, blocked_input=blocked):
                # Rendered, and not as ordinary text: unrendered, a declined turn looked to the
                # user like the model answering with nothing. The banner goes once per turn: a
                # refusal arrives in fragments, and one banner each read as several refusals.
                if in_text:
                    print()
                    in_text = False
                if not declined:
                    declined = True
                    what = "prompt blocked" if blocked else "declined"
                    tail = f" ({category})" if category else ""
                    sys.stdout.write(f"{RED}[{what}{tail}]{RESET} ")
                sys.stdout.write(text)
                sys.stdout.flush()

            case Citation(cited_text=cited, title=title, url=url):
                if in_text:
                    print()
                    in_text = False
                print(f"{DIM}[cited: {url or title or cited or 'source'}]{RESET}")

            case ImageOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[image saved: {path}]{RESET}")

            case AudioOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[audio saved: {path}]{RESET}")

            case VideoOutput(data=data, media_type=mt):
                if in_text:
                    print()
                    in_text = False
                path = _save_media(data, mt)
                print(f"{GREEN}[video saved: {path}]{RESET}")

            case ToolUseStart(index=index, tool_use_id=tid, name=name):
                if in_text:
                    print()
                    in_text = False
                sys.stdout.write(f"\n{BOLD}{CYAN}\u25b6 {name}{RESET}")
                sys.stdout.flush()
                arg_streams[tid] = ToolArgStream(tid, index)

            case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                stream = arg_streams.get(tid)
                if stream:
                    for fe in stream.feed(pj):
                        _render_field_event(fe)
                    if stream.done:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        del arg_streams[tid]

            case ToolOutputDelta(tool_use_id=tid, key=key, delta=delta):
                if tid not in streamed_tool_ids:
                    sys.stdout.write("\n")
                streamed_tool_ids.add(tid)
                color = RED if key == "stderr" else DIM
                sys.stdout.write(f"{color}{delta}{RESET}")
                sys.stdout.flush()

            case ToolResult(tool_use_id=tid, is_error=is_error, content=content):
                if is_error:
                    sys.stdout.write(f"{RESET}\n{RED}{content}{RESET}\n")
                elif tid in streamed_tool_ids:
                    sys.stdout.write(f"{RESET}\n")
                else:
                    sys.stdout.write(f"{RESET}\n{GREEN}{content}{RESET}\n")
                sys.stdout.flush()

            case IterationEnd():
                # The banner goes once per turn, and a prompt that runs tools has several. Set
                # once for the whole prompt, a refusal in a later turn printed its text with
                # nothing marking it as a refusal.
                declined = False

            case Error(exception=exc):
                print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)

            case SessionEndEvent(stop_reason=reason, total_usage=usage):
                if in_text:
                    print()
                if reason in INCOMPLETE:
                    # Nothing else says so. The answer stops mid-sentence and reads exactly like
                    # one the model finished.
                    print(f"{RED}[incomplete: {reason}]{RESET}")
                print(f"{DIM}[{usage.input_tokens}in/{usage.output_tokens}out tokens]{RESET}")


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


_field_first_delta = True


def _render_field_event(event: ToolFieldStart | ToolFieldDelta | ToolFieldEnd) -> None:
    global _field_first_delta
    match event:
        case ToolFieldStart(key=key):
            sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
            sys.stdout.flush()
            _field_first_delta = True
        case ToolFieldDelta(text=text):
            if _field_first_delta and "\n" in text:
                sys.stdout.write("\n")
            _field_first_delta = False
            sys.stdout.write(text)
            sys.stdout.flush()
        case ToolFieldEnd():
            sys.stdout.write(RESET)
            sys.stdout.flush()


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
        prompt_task: asyncio.Task[None] | None = None

        def _on_sigint() -> None:
            nonlocal prompt_task
            if prompt_task is not None and not prompt_task.done():
                prompt_task.cancel()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)

        try:
            if args.prompt:
                prompt_task = asyncio.create_task(run_prompt(agent, ctx, args.prompt))
                try:
                    await prompt_task
                except asyncio.CancelledError:
                    print(f"\n{DIM}[interrupted]{RESET}")
                finally:
                    prompt_task = None
                return

            commands_list = ", ".join(["/help", *commands, "/quit"])
            label = getattr(transport, "name", "unknown")
            print(f"REPL ready ({label}). Commands: {commands_list}")

            while True:
                try:
                    user_input = await loop.run_in_executor(None, _read_input)
                except EOFError:
                    print()
                    break

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

                matched = False
                for prefix, cmd in commands.items():
                    if lowered == prefix or lowered.startswith(prefix + " "):
                        arg = user_input[len(prefix) :].strip() or None
                        if arg is None:
                            cmd.show()
                        else:
                            cmd.apply(arg)
                        matched = True
                        break
                if matched:
                    continue

                prompt_task = asyncio.create_task(run_prompt(agent, ctx, user_input))
                try:
                    await prompt_task
                except asyncio.CancelledError:
                    print(f"\n{DIM}[interrupted]{RESET}")
                finally:
                    prompt_task = None
        finally:
            loop.remove_signal_handler(signal.SIGINT)


def main_sync() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main_sync()
