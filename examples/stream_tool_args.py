"""Minimal CLI demonstrating incremental streaming of tool call arguments
with partial JSON decoding - field values appear as they stream in.

Run:
    uv run --extra examples python examples/stream_tool_args.py "your prompt here"

Requires OPENAI_API_KEY in env.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

import aiohttp
from partial_json_parser import loads as partial_json_loads

from axio import (
    Agent,
    IterationEnd,
    MemoryContextStore,
    TextDelta,
    Tool,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
)
from axio.events import Error, SessionEndEvent
from axio_transport_openai import OPENAI_MODELS, OpenAITransport

# ── Tools ────────────────────────────────────────────────────────────


async def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Replace old_string with new_string in a file."""
    p = Path(file_path)
    text = p.read_text()
    if old_string not in text:
        raise ValueError(f"old_string not found in {file_path}")
    p.write_text(text.replace(old_string, new_string, 1))
    return f"Replaced content in {file_path}"


async def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {file_path}"


async def read_file(file_path: str) -> str:
    """Read content of a file."""
    return Path(file_path).read_text()


TOOLS = [
    Tool(name="edit_file", handler=edit_file),
    Tool(name="write_file", handler=write_file),
    Tool(name="read_file", handler=read_file),
]

# ── ANSI helpers ─────────────────────────────────────────────────────

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# ── Partial JSON tracker ────────────────────────────────────────────


class ToolArgTracker:
    """Tracks partial JSON for one tool call, diffs decoded fields on each chunk."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.raw = ""
        self.prev_parsed: dict[str, str] = {}
        self.prev_value_lens: dict[str, int] = {}
        self.seen_newline: set[str] = set()

    def feed(self, partial_json: str) -> None:
        self.raw += partial_json
        try:
            parsed = partial_json_loads(self.raw)
        except Exception:
            return
        if not isinstance(parsed, dict):
            return

        for key, value in parsed.items():
            value_str = str(value)
            prev_len = self.prev_value_lens.get(key, 0)
            new_text = value_str[prev_len:]

            if key not in self.prev_parsed:
                sys.stdout.write(f"\n  {YELLOW}{key}{RESET}: {DIM}")
                if "\n" in new_text:
                    self.seen_newline.add(key)
                    sys.stdout.write("\n" + value_str)
                else:
                    sys.stdout.write(value_str)
            elif new_text:
                if key not in self.seen_newline and "\n" in new_text:
                    self.seen_newline.add(key)
                    sys.stdout.write(
                        f"\r\033[2K  {YELLOW}{key}{RESET}:{DIM}\n{value_str}"
                    )
                else:
                    sys.stdout.write(new_text)

            self.prev_parsed[key] = value_str
            self.prev_value_lens[key] = len(value_str)

        sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream tool call arguments with partial JSON decoding"
    )
    parser.add_argument("prompt", help="prompt to send to the model")
    parser.add_argument(
        "--model", default="gpt-5.4", help="model name (default: gpt-5.4)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="sampling temperature"
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        transport = OpenAITransport(
            api_key=api_key,
            model=OPENAI_MODELS[args.model],
            session=session,
        )

        # Inject temperature into payload if requested
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
        )
        ctx = MemoryContextStore()

        print(f"{BOLD}👤 user:{RESET} {args.prompt}")

        trackers: dict[str, ToolArgTracker] = {}
        in_text = False

        async for event in agent.run_stream(args.prompt, ctx):
            match event:
                case TextDelta(delta=delta):
                    if not in_text:
                        sys.stdout.write(f"\n{BOLD}💬 model:{RESET} {DIM}")
                        in_text = True
                    sys.stdout.write(delta)
                    sys.stdout.flush()

                case ToolUseStart(tool_use_id=tid, name=name):
                    in_text = False
                    trackers[tid] = ToolArgTracker(name)
                    sys.stdout.write(f"{RESET}\n{BOLD}{CYAN}▶ {name}{RESET}")
                    sys.stdout.flush()

                case ToolInputDelta(tool_use_id=tid, partial_json=pj):
                    if tid in trackers:
                        trackers[tid].feed(pj)

                case ToolResult(
                    tool_use_id=tid, name=name, is_error=is_error, content=content
                ):
                    color = RED if is_error else GREEN
                    sys.stdout.write(f"{RESET}\n  {color}→ {content}{RESET}\n")
                    sys.stdout.flush()
                    trackers.pop(tid, None)

                case IterationEnd():
                    pass

                case Error(exception=exc):
                    print(f"\n{RED}Error: {exc}{RESET}", file=sys.stderr)

                case SessionEndEvent():
                    print()


if __name__ == "__main__":
    asyncio.run(main())
