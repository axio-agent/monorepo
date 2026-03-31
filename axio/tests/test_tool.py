"""Tests for axon.tool: ToolHandler, Tool with __call__."""

from __future__ import annotations

from typing import Any

from axio.exceptions import GuardError, HandlerError
from axio.permission import PermissionGuard
from axio.tool import Tool, ToolHandler


class EmptyHandler(ToolHandler):
    async def __call__(self) -> str:
        return "empty"


class MsgHandler(ToolHandler):
    msg: str

    async def __call__(self) -> str:
        return self.msg


class TestToolHandler:
    async def test_call_with_fields(self) -> None:
        h = MsgHandler(msg="hello")
        assert await h() == "hello"

    async def test_base_raises(self) -> None:
        try:
            await ToolHandler()()
            assert False, "should raise"
        except NotImplementedError:
            pass

    def test_schema_from_fields(self) -> None:
        schema = MsgHandler.model_json_schema()
        assert schema["type"] == "object"
        assert "msg" in schema["properties"]

    def test_repr(self) -> None:
        h = MsgHandler(msg="hello")
        assert "msg='hello'" in repr(h)


class TestTool:
    def test_no_guards(self) -> None:
        t = Tool(name="echo", description="test", handler=EmptyHandler)
        assert t.guards == ()

    def test_with_guards(self) -> None:
        class _G(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                return handler

        guard = _G()
        t = Tool(name="echo", description="test", handler=EmptyHandler, guards=(guard,))
        assert t.guards == (guard,)

    def test_input_schema_derived(self) -> None:
        t = Tool(name="t", description="t", handler=MsgHandler)
        schema = t.input_schema
        assert schema["type"] == "object"
        assert "msg" in schema["properties"]

    def test_concurrency(self) -> None:
        t = Tool(name="c", description="concurrent", handler=EmptyHandler, concurrency=3)
        assert t.concurrency == 3

    def test_concurrency_default_none(self) -> None:
        t = Tool(name="c", description="concurrent", handler=EmptyHandler)
        assert t.concurrency is None

    def test_frozen(self) -> None:
        t = Tool(name="t", description="t", handler=EmptyHandler)
        try:
            t.name = "other"  # type: ignore[misc]
            assert False, "should raise"
        except AttributeError:
            pass


class TestToolCall:
    async def test_kwargs_validate_and_execute(self) -> None:
        t = Tool(name="t", description="t", handler=MsgHandler)
        assert await t(msg="hello") == "hello"

    async def test_validation_error(self) -> None:
        t = Tool(name="t", description="t", handler=MsgHandler)
        try:
            await t(wrong_field="x")
            assert False, "should raise"
        except Exception:
            pass

    async def test_allow_guard(self) -> None:
        class _Allow(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                return handler

        t = Tool(name="t", description="t", handler=MsgHandler, guards=(_Allow(),))
        assert await t(msg="hello") == "hello"

    async def test_deny_guard(self) -> None:
        class _Deny(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                raise RuntimeError("nope")

        t = Tool(name="t", description="t", handler=MsgHandler, guards=(_Deny(),))
        try:
            await t(msg="hello")
            assert False, "should raise"
        except GuardError as exc:
            assert "nope" in str(exc)

    async def test_modify_guard(self) -> None:
        class _Modify(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                return handler.model_copy(update={"msg": "modified"})

        t = Tool(name="t", description="t", handler=MsgHandler, guards=(_Modify(),))
        assert await t(msg="original") == "modified"

    async def test_guard_receives_handler_instance(self) -> None:
        received: list[Any] = []

        class _Spy(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                received.append(handler)
                return handler

        t = Tool(name="t", description="t", handler=MsgHandler, guards=(_Spy(),))
        await t(msg="hello")
        assert len(received) == 1
        assert isinstance(received[0], MsgHandler)
        assert received[0].msg == "hello"

    async def test_guard_can_dump_to_dict(self) -> None:
        captured: list[dict[str, Any]] = []

        class _Dump(PermissionGuard):
            async def check(self, handler: Any) -> Any:
                captured.append(handler.model_dump())
                return handler

        t = Tool(name="t", description="t", handler=MsgHandler, guards=(_Dump(),))
        await t(msg="hello")
        assert captured[0] == {"msg": "hello"}

    async def test_handler_error_wrapping(self) -> None:
        class _Failing(ToolHandler):
            async def __call__(self) -> str:
                raise ValueError("handler boom")

        t = Tool(name="t", description="t", handler=_Failing)
        try:
            await t()
            assert False, "should raise"
        except HandlerError as exc:
            assert "handler boom" in str(exc)
