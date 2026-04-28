"""Tests for axio.tool: Tool with plain async handlers."""

from __future__ import annotations

import asyncio
import logging
from types import MappingProxyType
from typing import Annotated, Any

import pytest

from axio.exceptions import GuardError, HandlerError
from axio.field import Field, StrictStr
from axio.permission import PermissionGuard
from axio.schema import build_tool_schema
from axio.tool import CONTEXT, Tool


async def _empty() -> str:
    return "empty"


async def _msg(msg: str) -> str:
    return msg


class TestBuildSchema:
    def test_schema_from_function(self) -> None:
        schema = build_tool_schema(_msg)
        assert schema["type"] == "object"
        assert "msg" in schema["properties"]

    def test_return_excluded(self) -> None:
        schema = build_tool_schema(_msg)
        assert "return" not in schema["properties"]

    def test_default_not_required(self) -> None:
        async def f(query: str, limit: Annotated[int, Field(default=10)]) -> str:
            return query

        schema = build_tool_schema(f)
        assert "query" in schema["required"]
        assert "limit" not in schema.get("required", [])

    def test_py_default_not_required(self) -> None:
        async def f(query: str, limit: int = 10) -> str:
            return query

        schema = build_tool_schema(f)
        assert "query" in schema["required"]
        assert "limit" not in schema.get("required", [])

    def test_annotated_description(self) -> None:
        async def f(query: Annotated[str, Field(description="search query")]) -> str:
            return query

        schema = build_tool_schema(f)
        assert schema["properties"]["query"]["description"] == "search query"


class TestTool:
    def test_sync_handler_raises_type_error(self) -> None:
        def sync_fn() -> str:
            return "sync"

        with pytest.raises(TypeError, match="async"):
            Tool(name="t", description="t", handler=sync_fn)  # type: ignore[arg-type]

    def test_no_guards(self) -> None:
        t: Tool[Any] = Tool(name="echo", description="test", handler=_empty)
        assert t.guards == ()

    def test_with_guards(self) -> None:
        class _G(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                return kwargs

        guard = _G()
        t: Tool[Any] = Tool(name="echo", description="test", handler=_empty, guards=(guard,))
        assert t.guards == (guard,)

    def test_input_schema_derived(self) -> None:
        t: Tool[Any] = Tool(name="t", description="t", handler=_msg)
        schema = t.input_schema
        assert schema["type"] == "object"
        assert "msg" in schema["properties"]

    def test_concurrency(self) -> None:
        t: Tool[Any] = Tool(name="c", description="concurrent", handler=_empty, concurrency=3)
        assert t.concurrency == 3

    def test_concurrency_default_none(self) -> None:
        t: Tool[Any] = Tool(name="c", description="concurrent", handler=_empty)
        assert t.concurrency is None

    async def test_concurrency_limits_parallel_calls(self) -> None:
        """concurrency=N must prevent more than N simultaneous handler executions."""
        running = 0
        max_running = 0

        async def handler() -> str:
            nonlocal running, max_running
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0)
            running -= 1
            return "ok"

        t: Tool[Any] = Tool(name="t", handler=handler, concurrency=3)
        await asyncio.gather(*[t() for _ in range(10)])
        assert max_running <= 3

    def test_frozen(self) -> None:
        t: Tool[Any] = Tool(name="t", description="t", handler=_empty)
        with pytest.raises(AttributeError):
            t.name = "other"  # type: ignore[misc]

    def test_non_str_return_annotation_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        async def f() -> int:
            return 42

        with caplog.at_level(logging.WARNING, logger="axio.tool"):
            Tool(name="t", description="t", handler=f)  # type: ignore[arg-type]

        assert any("expected str" in r.message for r in caplog.records)

    def test_str_return_annotation_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="axio.tool"):
            Tool(name="t", description="t", handler=_empty)

        assert not any("expected str" in r.message for r in caplog.records)


class TestToolCall:
    async def test_basic_call(self) -> None:
        t: Tool[Any] = Tool(name="t", description="t", handler=_msg)
        assert await t(msg="hello") == "hello"

    async def test_allow_guard(self) -> None:
        class _Allow(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                return kwargs

        t: Tool[Any] = Tool(name="t", description="t", handler=_msg, guards=(_Allow(),))
        assert await t(msg="hello") == "hello"

    async def test_deny_guard(self) -> None:
        class _Deny(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                raise RuntimeError("nope")

        t: Tool[Any] = Tool(name="t", description="t", handler=_msg, guards=(_Deny(),))
        with pytest.raises(GuardError, match="nope"):
            await t(msg="hello")

    async def test_modify_guard(self) -> None:
        class _Modify(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                return {**kwargs, "msg": "modified"}

        t: Tool[Any] = Tool(name="t", description="t", handler=_msg, guards=(_Modify(),))
        assert await t(msg="original") == "modified"

    async def test_guard_sees_tool_and_kwargs(self) -> None:
        received: list[Any] = []

        class _Spy(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                received.append((tool, kwargs))
                return kwargs

        t: Tool[Any] = Tool(name="t", description="t", handler=_msg, guards=(_Spy(),))
        await t(msg="hello")
        assert len(received) == 1
        assert received[0][0] is t
        assert received[0][1] == {"msg": "hello"}

    async def test_guard_sees_materialised_field_defaults(self) -> None:
        """Guards must receive kwargs with FieldInfo defaults already injected."""
        received: list[dict[str, Any]] = []

        class _Spy(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                received.append(dict(kwargs))
                return kwargs

        async def f(query: str, limit: Annotated[int, Field(default=5)]) -> str:
            return f"{query}:{limit}"

        t: Tool[Any] = Tool(name="f", handler=f, guards=(_Spy(),))
        await t(query="hi")
        assert received == [{"query": "hi", "limit": 5}]

    async def test_guard_sees_materialised_signature_defaults(self) -> None:
        """Guards must receive kwargs with plain Python signature defaults injected."""
        received: list[dict[str, Any]] = []

        class _Spy(PermissionGuard):
            async def check(self, tool: Tool[Any], **kwargs: Any) -> dict[str, Any]:
                received.append(dict(kwargs))
                return kwargs

        async def f(query: str, cwd: str = ".") -> str:
            return f"{query}@{cwd}"

        t: Tool[Any] = Tool(name="f", handler=f, guards=(_Spy(),))
        await t(query="ls")
        assert received == [{"query": "ls", "cwd": "."}]

    async def test_handler_error_wrapping(self) -> None:
        async def _fail() -> str:
            raise ValueError("handler boom")

        t: Tool[Any] = Tool(name="t", description="t", handler=_fail)
        with pytest.raises(HandlerError, match="handler boom"):
            await t()

    async def test_context_var_set(self) -> None:
        captured: list[Any] = []

        async def f(x: str) -> str:
            captured.append(CONTEXT.get())
            return x

        t: Tool[str] = Tool(name="f", description="test", handler=f, context="cv")
        await t(x="hi")
        assert captured == ["cv"]

    async def test_fieldinfodefault_applied(self) -> None:
        async def f(query: str, limit: Annotated[int, Field(default=5)]) -> str:
            return f"{query}:{limit}"

        t: Tool[Any] = Tool(name="f", description="test", handler=f)
        assert await t(query="hello") == "hello:5"

    async def test_py_default_applied(self) -> None:
        async def f(query: str, limit: int = 7) -> str:
            return f"{query}:{limit}"

        t: Tool[Any] = Tool(name="f", description="test", handler=f)
        assert await t(query="hello") == "hello:7"

    async def test_result_coerced_to_str(self) -> None:
        async def f(n: int) -> str:
            return str(n * 2)

        t: Tool[Any] = Tool(name="f", description="test", handler=f)
        result = await t(n=21)
        assert result == "42"
        assert isinstance(result, str)

    async def test_concurrent_tools_get_isolated_context(self) -> None:
        """100 concurrent tool calls each see their own context, even across yields."""

        async def handler() -> str:
            before = CONTEXT.get()
            await asyncio.sleep(0)  # yield — let other coroutines interleave
            after = CONTEXT.get()
            assert before == after, f"context changed during yield: {before!r} → {after!r}"
            return str(before)

        tools = [Tool(name=f"t{i}", handler=handler, context=i) for i in range(100)]
        results = await asyncio.gather(*[t() for t in tools])
        assert results == [str(i) for i in range(100)]

    async def test_context_reset_after_call(self) -> None:
        """CONTEXT must be restored to its previous value after a tool call completes.

        Without token/reset, sequential calls in the same coroutine would leak
        the tool's context value to subsequent code.
        """
        sentinel = object()
        outer_token = CONTEXT.set(sentinel)
        try:

            async def handler() -> str:
                return str(CONTEXT.get())

            tool_a: Tool[str] = Tool(name="a", handler=handler, context="ctx_a")
            tool_b: Tool[str] = Tool(name="b", handler=handler, context="ctx_b")

            assert await tool_a() == "ctx_a"
            assert CONTEXT.get() is sentinel, "tool_a leaked context to caller"

            assert await tool_b() == "ctx_b"
            assert CONTEXT.get() is sentinel, "tool_b leaked context to caller"
        finally:
            CONTEXT.reset(outer_token)


class TestToolValidation:
    async def test_ge_violation_raises_handler_error(self) -> None:
        async def f(n: Annotated[int, Field(ge=1)]) -> str:
            return str(n)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError, match=">= 1"):
            await t(n=0)

    async def test_ge_boundary_passes(self) -> None:
        async def f(n: Annotated[int, Field(ge=1)]) -> str:
            return str(n)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        assert await t(n=1) == "1"

    async def test_le_violation_raises_handler_error(self) -> None:
        async def f(n: Annotated[int, Field(le=10)]) -> str:
            return str(n)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError, match="<= 10"):
            await t(n=11)

    async def test_le_boundary_passes(self) -> None:
        async def f(n: Annotated[int, Field(le=10)]) -> str:
            return str(n)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        assert await t(n=10) == "10"

    async def test_ge_below_range_raises(self) -> None:
        async def f(pct: Annotated[float, Field(ge=0.0, le=1.0)]) -> str:
            return str(pct)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError):
            await t(pct=-0.1)

    async def test_le_above_range_raises(self) -> None:
        async def f(pct: Annotated[float, Field(ge=0.0, le=1.0)]) -> str:
            return str(pct)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError):
            await t(pct=1.1)

    async def test_strict_str_rejects_int(self) -> None:
        async def f(name: StrictStr) -> str:
            return name

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError, match="str"):
            await t(name=42)

    async def test_strict_str_accepts_str(self) -> None:
        async def f(name: StrictStr) -> str:
            return name

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        assert await t(name="alice") == "alice"

    async def test_missing_required_param_raises_handler_error(self) -> None:
        async def f(required: str) -> str:
            return required

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        with pytest.raises(HandlerError, match="required"):
            await t()

    async def test_validation_only_on_provided_params(self) -> None:
        # ge constraint should not fire for a param that has a default and is omitted
        async def f(n: Annotated[int, Field(ge=1, default=5)]) -> str:
            return str(n)

        t: Tool[Any] = Tool(name="f", description="f", handler=f)
        assert await t() == "5"


class TestToolCustomSchema:
    """Tool accepts an explicit schema that overrides auto-generation."""

    def test_custom_schema_replaces_auto(self) -> None:
        custom: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        t: Tool[Any] = Tool(name="t", handler=_msg, schema=MappingProxyType(custom))
        assert t.input_schema == custom

    def test_auto_schema_when_not_provided(self) -> None:
        t: Tool[Any] = Tool(name="t", handler=_msg)
        assert "msg" in t.input_schema["properties"]

    def test_custom_schema_suppresses_handler_introspection(self) -> None:
        """Properties from type hints must NOT appear when a custom schema is given."""
        custom: dict[str, Any] = {
            "type": "object",
            "properties": {"blob": {"type": "string"}},
            "required": ["blob"],
        }
        t: Tool[Any] = Tool(name="t", handler=_msg, schema=MappingProxyType(custom))
        assert "msg" not in t.input_schema["properties"]
        assert "blob" in t.input_schema["properties"]

    def test_input_schema_returns_plain_dict(self) -> None:
        custom: dict[str, Any] = {"type": "object", "properties": {}}
        t: Tool[Any] = Tool(name="t", handler=_empty, schema=MappingProxyType(custom))
        result = t.input_schema
        assert isinstance(result, dict)
        assert not isinstance(result, MappingProxyType)

    def test_schema_is_immutable(self) -> None:
        """Stored schema must remain a MappingProxyType (frozen)."""
        custom: dict[str, Any] = {"type": "object", "properties": {}}
        t: Tool[Any] = Tool(name="t", handler=_empty, schema=MappingProxyType(custom))
        assert isinstance(t.schema, MappingProxyType)

    def test_empty_dict_schema_triggers_auto(self) -> None:
        """Passing an empty MappingProxyType is the same as not passing a schema."""
        t: Tool[Any] = Tool(name="t", handler=_msg, schema=MappingProxyType({}))
        assert "msg" in t.input_schema["properties"]

    def test_custom_schema_no_titles(self) -> None:
        """Custom schemas are passed through as-is - no titles injected."""
        custom: dict[str, Any] = {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }
        t: Tool[Any] = Tool(name="t", handler=_empty, schema=MappingProxyType(custom))
        assert "title" not in t.input_schema
        assert "title" not in t.input_schema["properties"].get("q", {})
