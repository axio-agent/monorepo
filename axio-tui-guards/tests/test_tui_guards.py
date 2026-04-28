"""Tests for axio_tui_guards - PathGuard and LLMGuard."""

from __future__ import annotations

from typing import Any

import pytest
from axio.blocks import TextBlock, ToolUseBlock
from axio.context import MemoryContextStore
from axio.exceptions import GuardError
from axio.messages import Message
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool
from axio_tui.tools import confirm, status_line

from axio_tui_guards.guards import LLMGuard, PathGuard

# ---------------------------------------------------------------------------
# Minimal stub handlers - no dependency on axio-tools-local
# ---------------------------------------------------------------------------


async def fake_read_file(filename: str) -> str:  # pragma: no cover
    return ""


async def fake_write_file(file_path: str, content: str) -> str:  # pragma: no cover
    return ""


async def fake_shell(command: str, cwd: str = ".") -> str:  # pragma: no cover
    return ""


async def fake_list_files(pattern: str = "*") -> str:  # pragma: no cover
    return ""


_read_tool: Tool[Any] = Tool(name="read_file", handler=fake_read_file)
_write_tool: Tool[Any] = Tool(name="write_file", handler=fake_write_file)
_shell_tool: Tool[Any] = Tool(name="shell", handler=fake_shell)
_list_tool: Tool[Any] = Tool(name="list_files", handler=fake_list_files)
_confirm_tool: Tool[Any] = Tool(name="confirm", handler=confirm)


class TestPathGuard:
    @staticmethod
    async def _allow(_msg: str) -> str:
        return "y"

    @staticmethod
    async def _deny(_msg: str) -> str:
        return "n"

    @staticmethod
    async def _always_deny(_msg: str) -> str:
        return "deny"

    async def test_allow_grants_directory(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        result = await guard.check(_read_tool, filename="/tmp/test/file.txt")
        assert result == {"filename": "/tmp/test/file.txt"}
        # Same directory should be auto-allowed now
        result2 = await guard.check(_read_tool, filename="/tmp/test/other.txt")
        assert result2 == {"filename": "/tmp/test/other.txt"}

    async def test_deny_raises_guard_error(self) -> None:
        guard = PathGuard(prompt_fn=self._deny)
        with pytest.raises(GuardError, match="denied"):
            await guard.check(_read_tool, filename="/secret/file.txt")

    async def test_always_deny_persists(self) -> None:
        guard = PathGuard(prompt_fn=self._always_deny)
        with pytest.raises(GuardError):
            await guard.check(_read_tool, filename="/deny/file.txt")
        # Second call should also be denied without prompting
        with pytest.raises(GuardError, match="denied"):
            await guard.check(_read_tool, filename="/deny/file.txt")

    async def test_no_path_field_passes_through(self) -> None:
        guard = PathGuard(prompt_fn=self._deny)
        result = await guard.check(_confirm_tool, verdict="SAFE", reason="ok", category="test")
        assert result == {"verdict": "SAFE", "reason": "ok", "category": "test"}

    async def test_shell_extracts_cwd(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        result = await guard.check(_shell_tool, command="ls", cwd="/tmp/project")
        assert result == {"command": "ls", "cwd": "/tmp/project"}
        assert "/tmp/project" in guard.allowed

    async def test_bypass_via_decoy_kwarg_is_prevented(self) -> None:
        """PathGuard is always prompted about the real handler kwarg, not a decoy extra.

        Without the pre-guard strip, a caller could pass file_path="/tmp/safe" alongside
        filename="/secret/x" - PathGuard checks "file_path" first (it appears earlier in
        PATH_FIELDS than "filename"), approves /tmp/safe, and the handler then runs with
        filename="/secret/x" unchecked. The pre-guard strip closes this gap.
        """
        seen_paths: list[str] = []

        async def capture(msg: str) -> str:
            seen_paths.append(msg)
            return "y"  # allow whatever the guard asks about

        guard = PathGuard(prompt_fn=capture)
        tool_with_guard: Tool[Any] = Tool(
            name="read_file",
            handler=fake_read_file,
            guards=(guard,),
        )
        # file_path is not a parameter of fake_read_file; it must be stripped before
        # the guard runs so PathGuard cannot be fooled by the decoy.
        await tool_with_guard(filename="/secret/file.txt", file_path="/tmp/decoy")

        # Guard must have been prompted about the real kwarg only.
        assert len(seen_paths) == 1
        assert "/secret/file.txt" in seen_paths[0]
        assert "/tmp/decoy" not in seen_paths[0]

    async def test_subdirectory_auto_allowed(self) -> None:
        guard = PathGuard(prompt_fn=self._allow)
        await guard.check(_write_tool, file_path="/home/user/project/src/main.py", content="x")
        # Child path in same tree should be auto-allowed
        result = await guard.check(_write_tool, file_path="/home/user/project/src/lib/util.py", content="y")
        assert result == {"file_path": "/home/user/project/src/lib/util.py", "content": "y"}


class TestLLMGuard:
    async def test_safe_verdict_passes(self) -> None:
        confirm_input = {"verdict": "SAFE", "reason": "harmless", "category": "read"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("ok"),
            ]
        )
        agent_from_transport = _make_guard_agent(transport)
        guard = LLMGuard(agent_from_transport, MemoryContextStore())
        result = await guard.check(_read_tool, filename="test.txt")
        assert result == {"filename": "test.txt"}

    async def test_deny_verdict_raises(self) -> None:
        confirm_input = {"verdict": "DENY", "reason": "malicious", "category": "exec"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("blocked"),
            ]
        )
        agent = _make_guard_agent(transport)
        guard = LLMGuard(agent, MemoryContextStore())
        with pytest.raises(GuardError, match="DENIED"):
            await guard.check(_shell_tool, command="rm -rf /")

    async def test_risky_verdict_user_allows(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "writes file", "category": "write"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("review needed"),
            ]
        )
        agent = _make_guard_agent(transport)

        async def allow(_msg: str) -> str:
            return "y"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=allow)
        result = await guard.check(_write_tool, file_path="out.txt", content="data")
        assert result == {"file_path": "out.txt", "content": "data"}

    async def test_risky_verdict_user_denies(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "dangerous", "category": "exec"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("needs review"),
            ]
        )
        agent = _make_guard_agent(transport)

        async def deny(_msg: str) -> str:
            return "n"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=deny)
        with pytest.raises(GuardError, match="denied"):
            await guard.check(_shell_tool, command="sudo reboot")

    async def test_always_allows_category(self) -> None:
        confirm_input = {"verdict": "RISKY", "reason": "writes", "category": "write_file"}
        transport = StubTransport(
            [
                make_tool_use_response("confirm", "call_1", confirm_input),
                make_text_response("ok"),
                # Second call - should not reach transport since category is pre-approved
            ]
        )
        agent = _make_guard_agent(transport)

        async def always(_msg: str) -> str:
            return "always"

        guard = LLMGuard(agent, MemoryContextStore(), prompt_fn=always)
        await guard.check(_write_tool, file_path="a.txt", content="data")
        assert "write_file" in guard.allowed

    async def test_extract_confirm_from_context(self) -> None:
        guard = LLMGuard.__new__(LLMGuard)
        ctx = MemoryContextStore()
        await ctx.append(Message(role="user", content=[TextBlock(text="check this")]))
        await ctx.append(
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="c1", name="confirm", input={"verdict": "SAFE", "reason": "ok", "category": "read"}
                    ),
                ],
            )
        )
        confirm_result = await guard.extract_confirm(ctx)
        assert confirm_result.verdict == "SAFE"
        assert confirm_result.category == "read"

    async def test_extract_confirm_no_verdict(self) -> None:
        guard = LLMGuard.__new__(LLMGuard)
        ctx = MemoryContextStore()
        await ctx.append(Message(role="user", content=[TextBlock(text="check")]))
        await ctx.append(Message(role="assistant", content=[TextBlock(text="no tool call")]))
        confirm_result = await guard.extract_confirm(ctx)
        assert confirm_result.verdict == "SAFE"
        assert confirm_result.reason == "No verdict provided"


def _make_guard_agent(transport: Any) -> Any:
    from axio.agent import Agent
    from axio.tool import Tool

    return Agent(
        system="You are a safety classifier.",
        tools=[
            Tool(name="status_line", handler=status_line),
            Tool(name="confirm", handler=confirm),
        ],
        transport=transport,
        max_iterations=5,
    )
