"""Permission guards for the Axio TUI."""

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from axio.agent import Agent
from axio.blocks import TextBlock, ToolUseBlock
from axio.context import ContextStore
from axio.exceptions import GuardError
from axio.messages import Message
from axio.permission import PermissionGuard

if TYPE_CHECKING:
    from axio.tool import Tool

from axio_tui.tools import Confirm

type PromptFn = Callable[[str], Awaitable[str]]


class PathGuard(PermissionGuard):
    """Ask user about path access; allow grants access to the parent directory."""

    PATH_FIELDS = ("file_path", "filename", "directory", "path", "cwd")

    def __init__(self, prompt_fn: PromptFn | None = None) -> None:
        self.allowed: set[str] = set()
        self.denied: set[str] = set()
        self._prompt = prompt_fn or ask_user

    def _extract_path(self, kwargs: dict[str, Any]) -> tuple[str | None, str | None]:
        for name in self.PATH_FIELDS:
            if name in kwargs:
                return str(kwargs[name]), name
        return None, None

    @staticmethod
    def _parent_dir(path: str) -> str:
        p = Path(path)
        return str(p if p.suffix == "" else p.parent)

    def _is_allowed(self, path: str) -> bool:
        d = Path(self._parent_dir(path))
        return any(d == Path(a) or Path(a) in d.parents for a in self.allowed)

    def _is_denied(self, path: str) -> bool:
        return path in self.denied

    async def check(self, tool: "Tool[Any]", **kwargs: Any) -> dict[str, Any]:
        path, field = self._extract_path(kwargs)
        if path is None or self._is_allowed(path):
            return kwargs
        if self._is_denied(path):
            raise GuardError(f"Path denied: {field}={path!r}")

        directory = self._parent_dir(path)
        msg = f"{field}={path!r} (directory: {directory})"
        answer = (await self._prompt(msg)).strip()
        if answer.lower() == "deny":
            self.denied.add(path)
            raise GuardError(f"Path denied: {field}={path!r}")
        if not answer or answer.lower() == "n":
            raise GuardError(f"Path access denied: {field}={path!r}")
        # "y" or anything else - allow this directory for future calls
        self.allowed.add(directory)
        return kwargs


class LLMGuard(PermissionGuard):
    """Agent-based guard. User overrides feed back into the context for learning."""

    def __init__(self, agent: Agent, context: ContextStore, prompt_fn: PromptFn | None = None) -> None:
        self.agent = agent
        self.context = context
        self.allowed: set[str] = set()
        self._prompt = prompt_fn or ask_user

    async def extract_confirm(self, context: ContextStore) -> Confirm:
        """Find last confirm tool call from forked context."""
        history = await context.get_history()
        for msg in reversed(history):
            if msg.role == "assistant":
                for block in msg.content:
                    if isinstance(block, ToolUseBlock) and block.name == "confirm":
                        try:
                            return Confirm(**block.input)
                        except Exception:
                            return Confirm(verdict="RISKY", reason="Unparseable", category="unknown")
        return Confirm(verdict="SAFE", reason="No verdict provided", category="unknown")

    async def check(self, tool: "Tool[Any]", **kwargs: Any) -> dict[str, Any]:
        args_str = json.dumps(kwargs, default=str)
        if len(args_str) > 2000:
            args_str = args_str[:2000] + "..."

        description = f"Tool: {tool.name}\nArguments: {args_str}"
        if self.allowed:
            description += "\n\nAuto-approved categories (classify as SAFE): " + ", ".join(sorted(self.allowed))

        # Fork so tool-call noise is discarded; user answers persist in the original
        forked = await self.context.fork()
        async for _ in self.agent.run_stream(description, forked):
            pass

        confirm = await self.extract_confirm(forked)

        if confirm.verdict == "SAFE":
            return kwargs
        if confirm.verdict == "DENY":
            raise GuardError(f"DENIED: {confirm.reason}")

        # RISKY - ask user
        if confirm.category in self.allowed:
            return kwargs

        args_repr = json.dumps(kwargs, default=repr)
        if len(args_repr) > 2000:
            args_repr = args_repr[:2000] + "..."
        msg = f"{tool.name}({args_repr})\n\n{confirm.reason}"
        answer = (await self._prompt(msg)).strip()

        if not answer or answer.lower() == "n":
            raise GuardError(f"User denied: {confirm.reason}")

        if answer.lower() == "always":
            self.allowed.add(confirm.category)
        elif answer.lower() != "y":
            # Feed user answer back and let it run more turns
            async for _ in self.agent.run_stream(answer, forked):
                pass
            # Persist the user note in original context for future checks
            await self.context.append(Message(role="user", content=[TextBlock(text=answer)]))

        return kwargs


# Helper function for user input
ASK_LOCK = threading.Lock()


async def ask_user(message: str) -> str:
    def _blocking() -> str:
        with ASK_LOCK:
            return input(message)

    return await asyncio.to_thread(_blocking)
