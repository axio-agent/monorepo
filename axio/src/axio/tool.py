"""Tool: frozen dataclass binding a handler callable to a name, guards, and concurrency."""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from types import MappingProxyType
from typing import Any, get_type_hints

from .exceptions import GuardError, HandlerError
from .field import MISSING, FieldInfo, get_field_info
from .permission import PermissionGuard
from .schema import build_tool_schema
from .types import ToolName

type JSONSchema = dict[str, Any]

logger = logging.getLogger(__name__)

# Universal opt-in argument understood by the agent loop and never passed to a
# handler; named here so the schema and the dispatcher cannot drift apart.
BACKGROUND_PARAM = "background"

# Maps JSON Schema primitive type names to Python types used for validation.
SCHEMA_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def hint_from_json_schema(prop_schema: dict[str, Any]) -> Any:
    """Return the Python type hint for a single JSON Schema property definition."""
    t = prop_schema.get("type")
    if t is not None:
        return SCHEMA_JSON_TYPE_MAP.get(t, object)
    any_of = prop_schema.get("anyOf")
    if any_of is not None:
        non_null = [s for s in any_of if s.get("type") != "null"]
        has_null = len(non_null) < len(any_of)
        if len(non_null) == 1:
            inner = hint_from_json_schema(non_null[0])
            return (inner | None) if has_null else inner
    return object


# Set to the tool's ``context`` value before each handler invocation.
# Handlers that cannot receive context as a parameter retrieve it via ``CONTEXT.get()``.
CONTEXT: ContextVar[Any] = ContextVar("CONTEXT")


def _default_format_stream_result(chunks: list[tuple[float, str, str]]) -> str:
    """Default streaming-aggregator: join all text, discard keys/timestamps."""
    return "".join(text for _, _, text in chunks)


@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName
    handler: Callable[..., Awaitable[Any]]
    description: str = ""
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None

    context: T = field(default=MappingProxyType({}), compare=False)  # type: ignore[assignment]
    schema: MappingProxyType[str, Any] = field(default=MappingProxyType({}), repr=False, compare=False)
    _semaphore: asyncio.Semaphore | None = field(init=False, default=None, repr=False, compare=False)
    _fields: Mapping[str, tuple[Any, FieldInfo]] = field(
        init=False, repr=False, compare=False, default_factory=lambda: MappingProxyType({})
    )
    _accepts_var_kwargs: bool = field(init=False, default=False, repr=False, compare=False)
    _schema_explicit: bool = field(init=False, default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not inspect.iscoroutinefunction(self.handler):
            raise TypeError(
                f"Tool {self.name!r} handler {self.handler!r} must be an async function (coroutinefunction)."
            )
        if not self.description:
            object.__setattr__(self, "description", self.handler.__doc__ or "")
        hints = get_type_hints(self.handler, include_extras=True)
        param_hints = {k: v for k, v in hints.items() if k != "return"}
        try:
            sig = inspect.signature(self.handler)
        except (ValueError, TypeError):
            sig = None
        fields: dict[str, tuple[Any, FieldInfo]] = {}
        for name, hint in param_hints.items():
            fi = get_field_info(hint) or FieldInfo()
            if sig is not None and name in sig.parameters:
                param = sig.parameters[name]
                if param.default is not inspect.Parameter.empty and fi.default is MISSING:
                    # Merge sig default into FieldInfo (covers StrictStr and plain defaults).
                    fi = dc_replace(fi, default=param.default)
            fields[name] = (hint, fi)
        param_fields = MappingProxyType(fields)
        accepts_var_kwargs = sig is not None and any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        schema_explicit = bool(self.schema)
        if not self.schema:
            object.__setattr__(self, "schema", MappingProxyType(build_tool_schema(self.handler, hints=param_hints)))
        # For handlers with an explicit schema: synthesise _fields from schema properties
        # so type validation, default injection, and kwarg filtering all use the schema.
        if schema_explicit:
            schema_props: dict[str, Any] = dict(self.schema).get("properties") or {}
            schema_fields: dict[str, tuple[Any, FieldInfo]] = {}
            for prop_name, prop_schema in schema_props.items():
                hint = hint_from_json_schema(prop_schema)
                default = prop_schema.get("default", MISSING)
                schema_fields[prop_name] = (hint, FieldInfo(default=default))
            if schema_fields:
                param_fields = MappingProxyType(schema_fields)
        object.__setattr__(self, "_fields", param_fields)
        object.__setattr__(self, "_accepts_var_kwargs", accepts_var_kwargs)
        object.__setattr__(self, "_schema_explicit", schema_explicit)
        if self.concurrency is not None:
            object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.concurrency))

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[None, None]:
        if self._semaphore is None:
            yield
            return
        async with self._semaphore:
            yield

    @property
    def input_schema(self) -> JSONSchema:
        schema = copy.deepcopy(dict(self.schema))
        # Offered on every tool rather than a chosen few: whether a call is
        # worth detaching depends on its arguments, not on the tool.
        properties = schema.setdefault("properties", {})
        if isinstance(properties, dict) and BACKGROUND_PARAM not in properties:
            properties[BACKGROUND_PARAM] = {
                "type": "boolean",
                "default": False,
                "description": (
                    "Run detached and return a handle instead of the result. Use it when the call is slow "
                    "enough to be worth doing while you carry on; collect the output later with "
                    "monitor(tasks=[handle])."
                ),
            }
        return schema

    @property
    def supports_streaming(self) -> bool:
        """Handler supports streaming if it exposes a ``.stream`` async-generator attribute."""
        return callable(getattr(self.handler, "stream", None))

    def format_stream_result(self, chunks: list[tuple[float, str, str]]) -> str:
        """Aggregate streamed chunks into the final tool result string.

        Handlers may attach a ``format_stream_result`` callable for structured
        output (e.g. shell log records). Defaults to text concatenation.
        """
        fn = getattr(self.handler, "format_stream_result", None)
        if callable(fn):
            result = fn(chunks)
            return result if isinstance(result, str) else str(result)
        return _default_format_stream_result(chunks)

    def _prepare_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inject defaults, validate types, and strip extras per schema."""
        required_set: set[str] = set(self.schema.get("required", []))
        for name, (hint, fi) in self._fields.items():
            if name not in kwargs:
                if fi.default is not MISSING and name not in required_set:
                    kwargs[name] = fi.default
            else:
                fi.validate(kwargs[name], name, hint)
        missing = [name for name in required_set if name not in kwargs]
        if missing:
            raise HandlerError(f"Missing required field(s): {', '.join(missing)}")
        if self._schema_explicit:
            schema_props = self.schema.get("properties")
            if schema_props is not None:
                kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
        elif not self._accepts_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self._fields}
        return kwargs

    async def __call__(self, **kwargs: Any) -> Any:
        async with self._acquire():
            try:
                kwargs = self._prepare_kwargs(kwargs)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

            for guard in self.guards:
                try:
                    kwargs = await guard(self, **kwargs)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardError(str(exc)) from exc

            try:
                if self._schema_explicit:
                    schema_props = self.schema.get("properties")
                    if schema_props is not None:
                        kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
                elif not self._accepts_var_kwargs:
                    kwargs = {k: v for k, v in kwargs.items() if k in self._fields}
                token = CONTEXT.set(self.context)
                try:
                    return await self.handler(**kwargs)
                finally:
                    CONTEXT.reset(token)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

    async def call_streaming(self, **kwargs: Any) -> AsyncGenerator[tuple[str, str], None]:
        """Execute handler, yielding ``(key, text)`` chunks for streaming output.

        Uses ``handler.stream(**kwargs)`` if the handler exposes one. Otherwise
        falls back to ``__call__()`` and yields the full result as a single
        ``("output", ...)`` chunk. Semaphore is held for the entire iteration.
        """
        async with self._acquire():
            try:
                kwargs = self._prepare_kwargs(kwargs)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

            for guard in self.guards:
                try:
                    kwargs = await guard(self, **kwargs)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardError(str(exc)) from exc

            if self._schema_explicit:
                schema_props = self.schema.get("properties")
                if schema_props is not None:
                    kwargs = {k: v for k, v in kwargs.items() if k in schema_props}
            elif not self._accepts_var_kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in self._fields}

            stream_fn = getattr(self.handler, "stream", None)
            token = CONTEXT.set(self.context)
            try:
                if callable(stream_fn):
                    async for chunk in stream_fn(**kwargs):
                        yield chunk
                else:
                    result = await self.handler(**kwargs)
                    if isinstance(result, str):
                        yield ("output", result)
                    else:
                        yield ("output", str(result))
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc
            finally:
                CONTEXT.reset(token)
