"""Tests for Agent permission paths: allow, deny, modify, guard errors."""

from __future__ import annotations

from typing import Any

from axio.agent import Agent
from axio.blocks import ToolResultBlock
from axio.context import MemoryContextStore
from axio.events import SessionEndEvent, StreamEvent, ToolResult
from axio.permission import PermissionGuard
from axio.testing import StubTransport, make_text_response, make_tool_use_response
from axio.tool import Tool, ToolHandler


class EchoHandler(ToolHandler):
    msg: str

    async def __call__(self) -> str:
        return self.model_dump_json()


class _AllowGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        return handler


class _DenyGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        raise RuntimeError("test-denied")


class _ModifyGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        return handler.model_copy(update={"msg": "modified"})


class _FailingGuard(PermissionGuard):
    async def check(self, handler: Any) -> Any:
        raise ValueError("guard crashed")


class TestAllowGuard:
    async def test_handler_called_normally(self) -> None:
        calls: list[dict[str, Any]] = []

        class Tracking(ToolHandler):
            msg: str

            async def __call__(self) -> str:
                calls.append(self.model_dump())
                return self.model_dump_json()

        tool = Tool(name="echo", description="echo", handler=Tracking, guards=(_AllowGuard(),))
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(calls) == 1


class TestDenyGuard:
    async def test_handler_not_called(self) -> None:
        """C5: deny never crashes, produces ToolResultBlock(is_error=True)."""
        calls: list[dict[str, Any]] = []

        class Tracking(ToolHandler):
            msg: str

            async def __call__(self) -> str:
                calls.append(self.model_dump())
                return self.model_dump_json()

        tool = Tool(name="echo", description="echo", handler=Tracking, guards=(_DenyGuard(),))
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        ctx = MemoryContextStore()
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", ctx):
            events.append(e)

        assert len(calls) == 0
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error

        history = await ctx.get_history()
        user_msgs = [m for m in history if m.role == "user"]
        result_blocks = [b for m in user_msgs for b in m.content if isinstance(b, ToolResultBlock)]
        assert any(b.is_error and "test-denied" in str(b.content) for b in result_blocks)


class TestModifyGuard:
    async def test_handler_called_with_modified_input(self) -> None:
        """C6: handler receives modified_input, not original."""
        calls: list[dict[str, Any]] = []

        class Tracking(ToolHandler):
            msg: str

            async def __call__(self) -> str:
                calls.append(self.model_dump())
                return self.model_dump_json()

        tool = Tool(name="echo", description="echo", handler=Tracking, guards=(_ModifyGuard(),))
        transport = StubTransport(
            [make_tool_use_response("echo", "c1", {"msg": "original"}), make_text_response("Done")]
        )
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(calls) == 1
        assert calls[0] == {"msg": "modified"}


class TestGuardException:
    async def test_treated_as_deny(self) -> None:
        calls: list[dict[str, Any]] = []

        class Tracking(ToolHandler):
            msg: str

            async def __call__(self) -> str:
                calls.append(self.model_dump())
                return self.model_dump_json()

        tool = Tool(name="echo", description="echo", handler=Tracking, guards=(_FailingGuard(),))
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        events: list[StreamEvent] = []
        async for e in agent.run_stream("go", MemoryContextStore()):
            events.append(e)

        assert len(calls) == 0
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error
        assert isinstance(events[-1], SessionEndEvent)


class TestNoGuard:
    async def test_handler_called_without_guard(self) -> None:
        calls: list[dict[str, Any]] = []

        class Tracking(ToolHandler):
            msg: str

            async def __call__(self) -> str:
                calls.append(self.model_dump())
                return self.model_dump_json()

        tool = Tool(name="echo", description="echo", handler=Tracking)
        transport = StubTransport([make_tool_use_response("echo", "c1", {"msg": "hi"}), make_text_response("Done")])
        agent = Agent(system="test", tools=[tool], transport=transport)
        await agent.run("go", MemoryContextStore())
        assert len(calls) == 1
