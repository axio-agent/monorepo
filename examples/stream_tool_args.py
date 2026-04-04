"""Minimal CLI demonstrating incremental streaming of tool call arguments
with partial JSON decoding — field values appear as they stream in.

Auto-detects transport from available API keys (OPENAI_API_KEY, NEBIUS_API_KEY,
OPENROUTER_API_KEY), or use --transport to pick explicitly.

Run:
    uv run examples/stream_tool_args.py "your prompt here"
"""

from __future__ import annotations

import asyncio
import os
from importlib.metadata import entry_points
import sys

import aiohttp
from axio.agent import Agent
from axio.context import MemoryContextStore
from axio.events import (
    Error,
    IterationEnd,
    SessionEndEvent,
    TextDelta,
    ToolFieldDelta,
    ToolFieldEnd,
    ToolFieldStart,
    ToolResult,
    ToolUseStart,
)
from axio.tool import Tool
from axio_tools_local.patch_file import PatchFile
from axio_tools_local.read_file import ReadFile
from axio_tools_local.write_file import WriteFile

TOOLS = [
    Tool(name="read_file", description=ReadFile.__doc__ or "", handler=ReadFile),
    Tool(name="write_file", description=WriteFile.__doc__ or "", handler=WriteFile),
    Tool(name="patch_file", description=PatchFile.__doc__ or "", handler=PatchFile),
]

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ── Transport auto-detection ─────────────────────────────────────────


def _discover_transports() -> dict[str, type]:
    """Load transport classes from axio.transport entry points."""
    result = {}
    for ep in entry_points(group="axio.transport"):
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


def _select_transport(name: str | None) -> tuple[type, str]:
    """Return (transport_class, api_key) based on --transport or env auto-detection."""
    available = _discover_transports()
    if name:
        if name not in available:
            print(f"Unknown transport {name!r}. Available: {', '.join(sorted(available))}", file=sys.stderr)
            sys.exit(1)
        cls = available[name]
        meta = getattr(cls, "META", None)
        env_var = meta.api_key_env if meta else ""
        api_key = os.environ.get(env_var, "") if env_var else ""
        if not api_key:
            print(f"Set {env_var} for transport {name!r}", file=sys.stderr)
            sys.exit(1)
        return cls, api_key

    for tname, cls in available.items():
        meta = getattr(cls, "META", None)
        if meta and meta.api_key_env:
            api_key = os.environ.get(meta.api_key_env, "")
            if api_key:
                return cls, api_key

    print("No API key found. Set one of:", file=sys.stderr)
    for tname, cls in available.items():
        meta = getattr(cls, "META", None)
        if meta and meta.api_key_env:
            print(f"  {meta.api_key_env}  ({meta.label})", file=sys.stderr)
    sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stream tool call arguments with partial JSON decoding")
    parser.add_argument("prompt", help="prompt to send to the model")
    parser.add_argument("--transport", default=None, help="transport name (auto-detected from API keys if omitted)")
    parser.add_argument("--model", default=None, help="model name (uses transport default if omitted)")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    args = parser.parse_args()

    transport_cls, api_key = _select_transport(args.transport)

    async with aiohttp.ClientSession() as session:
        transport = transport_cls(api_key=api_key, session=session)

        if args.model:
            transport.model = transport.models[args.model]

        if args.temperature is not None:
            _orig_build = transport.build_payload

            def build_payload_with_temp(messages, tools, system):
                payload = _orig_build(messages, tools, system)
                payload["temperature"] = args.temperature
                return payload

            transport.build_payload = build_payload_with_temp  # type: ignore[method-assign]

        agent = Agent(
            system="You are a coding assistant. Use the provided tools.",
            tools=TOOLS,
            transport=transport,
            parse_tool_args=True,
        )
        ctx = MemoryContextStore()

        print(f"{BOLD}👤 user:{RESET} {args.prompt}")

        in_text = False

        async for event in agent.run_stream(args.prompt, ctx):
            match event:
                case TextDelta(delta=delta):
                    if not in_text:
                        sys.stdout.write(f"\n{BOLD}💬 model:{RESET} {DIM}")
                        in_text = True
                    sys.stdout.write(delta)
                    sys.stdout.flush()

                case ToolUseStart(name=name):
                    in_text = False
                    sys.stdout.write(f"{RESET}\n{BOLD}{CYAN}▶ {name}{RESET}")
                    sys.stdout.flush()

                case ToolFieldStart(key=key):
                    sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
                    sys.stdout.flush()

                case ToolFieldDelta(text=text):
                    sys.stdout.write(text)
                    sys.stdout.flush()

                case ToolFieldEnd():
                    sys.stdout.write(RESET)
                    sys.stdout.flush()

                case ToolResult(is_error=is_error, content=content):
                    color = RED if is_error else GREEN
                    sys.stdout.write(f"{RESET}\n{color}{content}{RESET}\n")
                    sys.stdout.flush()

                case IterationEnd():
                    pass

                case Error(exception=exc):
                    print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)

                case SessionEndEvent():
                    print()


if __name__ == "__main__":
    asyncio.run(main())
