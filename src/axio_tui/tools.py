"""TUI-specific tool handlers."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ClassVar, Literal

from axio.agent import Agent
from axio.blocks import ImageBlock, TextBlock
from axio.context import ContextStore
from axio.events import TextDelta
from axio.messages import Message
from axio.tool import ToolHandler
from axio.transport import CompletionTransport


def _short(value: Any, limit: int = 60) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "..."


class StatusLine(ToolHandler):
    """Set a short status message shown to the user in the TUI status bar.
    Call this at the start of every turn before other tools to indicate
    what you are about to do (e.g. "Reading project files",
    "Running tests"). Keep messages short: 3-8 words."""

    message: str
    _callback: ClassVar[Callable[[str], None] | None] = None

    def __repr__(self) -> str:
        return f"StatusLine(message={self.message!r})"

    async def __call__(self) -> str:
        if StatusLine._callback:
            StatusLine._callback(self.message)
        return "ok"


class Confirm(ToolHandler):
    """Submit a safety verdict for a pending tool call. Used by the
    LLM-based permission guard. Verdict must be SAFE, RISKY, or DENY.
    Provide a clear reason and a category for the action."""

    verdict: Literal["SAFE", "RISKY", "DENY"]
    reason: str
    category: str

    def __repr__(self) -> str:
        return f"Confirm(verdict={self.verdict!r}, category={self.category!r})"

    async def __call__(self) -> str:
        return self.verdict


SUBAGENT_SYSTEM_PROMPT = """\
<subagent_behavior>
  <identity>
    You are a sub-agent — a focused autonomous worker spawned by a parent agent.
    You receive a specific task and execute it to completion. You cannot communicate
    with the user — only the parent agent reads your output. You cannot spawn
    sub-agents yourself.
  </identity>

  <tools>
    Your available tools are listed in the tool definitions. The set of tools varies
    depending on what packages the user has installed — you may or may not have
    filesystem, shell, or other capabilities. Work only with the tools you actually
    have. If the task requires a tool you don't have, report that clearly instead
    of trying workarounds.
  </tools>

  <core_rules>
    Execute ONLY the delegated task. Do not deviate, expand scope, or do extra work.
    Do not ask follow-up questions — work with what you have.
    When done, return a clear, structured result as plain text.
  </core_rules>

  <status_line>
    If you have the `status_line` tool, call it at the start of every turn before
    any other tool call. Use a short (3-8 words) description of what you are about
    to do. Examples: "Reading auth module", "Searching for imports", "Running tests".
  </status_line>

  <parallel_tool_calls>
    ALL tool calls in a single turn run CONCURRENTLY. This is critical for performance.
    You MUST maximize the number of tool calls per turn. Every turn where you call
    only one tool but COULD have called more is wasted time.

    Before submitting each turn, ask: "Are there other calls I need that do NOT depend
    on these results?" If yes — add them NOW.

    Pack as many independent calls as possible into every turn:
    - Need to read 5 files? → 5 read calls in one turn, not five turns.
    - Need to list a directory and read a config? → both in one turn.
    - Need to write 3 independent files? → 3 write calls in one turn.
    - Need to run a search and read a known file? → both in one turn.
    - Need status_line + other tools? → status_line + all tools in one turn.

    The ONLY reason for a separate turn is when call B needs the RESULT of call A.
    Example: read a file (turn 1) → modify it based on contents (turn 2).
    Everything else goes in one turn.
  </parallel_tool_calls>

  <execution_methodology>
    Be systematic:

    1. ORIENT — Understand the task. Read relevant files to get context before acting.

    2. ACT — Make changes, write code, run commands. For file modifications, prefer
       patching over full rewrites to minimize disruption. Use full writes only for
       new files.

    3. VERIFY — Confirm your work. If you modified code, run tests or check syntax.
       If you searched for something, confirm by reading the actual code.
       Never report something works without evidence.

    When modifying code:
    - Read the file first to understand context and style.
    - Match the existing code style (indentation, naming, patterns).
    - Make minimal changes — do not refactor or "improve" unrelated code.
  </execution_methodology>

  <permission_denials>
    If a tool call is denied (error containing "denied"), STOP IMMEDIATELY.
    Do NOT retry, use a different tool for the same effect, or work around it.
    Report the denial clearly: what you tried, that it was denied, and that you
    stopped. The parent agent will decide next steps.
  </permission_denials>

  <output_format>
    Return facts, findings, code, or summaries — not conversational filler.
    Structure your output clearly:
    - For research: key findings first, supporting details after.
    - For code changes: list what you changed (file, function, what/why).
    - For errors: what went wrong, what you observed, raw output.
    - If something was not found, say so explicitly rather than guessing.

    The parent agent will synthesize results from multiple sub-agents.
    Keep output focused and machine-readable.
  </output_format>
</subagent_behavior>"""


class SubAgent(ToolHandler):
    """Delegate a task to an independent sub-agent. The sub-agent receives
    conversation context and has access to the same tools (except subagent).
    Up to 3 sub-agents run concurrently. Use for parallelizing independent
    subtasks: reading multiple files, searching across directories,
    or running separate analyses. Write a specific, self-contained task
    description — the sub-agent cannot ask follow-up questions."""

    task: str
    _factory: ClassVar[Callable[[], Awaitable[tuple[Agent, ContextStore]]] | None] = None
    _tool_concurrency: ClassVar[int | None] = 3

    def __repr__(self) -> str:
        return f"SubAgent(task={_short(self.task)!r})"

    async def __call__(self) -> str:
        if SubAgent._factory is None:
            return "SubAgent is not configured"
        agent, context = await SubAgent._factory()
        return await agent.run(self.task, context)


_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class VisionAnalyze(ToolHandler):
    """Analyze an image file using a vision-capable model. Supports
    JPEG, PNG, GIF, and WebP formats. Provide a path to the image and
    an optional prompt describing what to look for. Returns the model's
    textual analysis of the image content."""

    path: str
    prompt: str = "Describe this image in detail."
    _transport: ClassVar[CompletionTransport | None] = None

    def __repr__(self) -> str:
        return f"VisionAnalyze(path={_short(self.path)!r})"

    async def __call__(self) -> str:
        import os

        if VisionAnalyze._transport is None:
            return "Vision is not configured — no vision model selected."

        image_path = Path(os.getcwd()) / self.path
        if not image_path.is_file():
            return f"File not found: {self.path}"

        ext = image_path.suffix.lower()
        media_type = _IMAGE_MEDIA_TYPES.get(ext)
        if media_type is None:
            return f"Unsupported image format: {ext} (supported: jpg, png, gif, webp)"

        data = await asyncio.to_thread(image_path.read_bytes)

        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(text=self.prompt),
                    ImageBlock(media_type=media_type, data=data),  # type: ignore[arg-type]
                ],
            ),
        ]

        text_parts: list[str] = []
        async for event in VisionAnalyze._transport.stream(messages, [], "You are a helpful vision assistant."):
            if isinstance(event, TextDelta):
                text_parts.append(event.delta)
        return "".join(text_parts) or "(no response)"
