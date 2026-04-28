"""Tool: frozen dataclass binding a handler callable to a name, guards, and concurrency."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, get_type_hints

from .exceptions import GuardError, HandlerError
from .field import MISSING, FieldInfo, get_field_info
from .permission import PermissionGuard
from .schema import build_tool_schema
from .types import ToolName

type JSONSchema = dict[str, Any]

logger = logging.getLogger(__name__)

# Set to the tool's ``context`` value before each handler invocation.
# Handlers that cannot receive context as a parameter retrieve it via ``CONTEXT.get()``.
CONTEXT: ContextVar[Any] = ContextVar("CONTEXT")


@dataclass(frozen=True, slots=True)
class Tool[T]:
    name: ToolName
    handler: Callable[..., Awaitable[str]]
    description: str = ""
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None

    context: T = field(default=MappingProxyType({}), compare=False)  # type: ignore[assignment]
    schema: MappingProxyType[str, Any] = field(default=MappingProxyType({}), repr=False, compare=False)
    _semaphore: asyncio.Semaphore | None = field(init=False, default=None, repr=False, compare=False)
    _fields: Mapping[str, tuple[Any, FieldInfo | None]] = field(
        init=False, repr=False, compare=False, default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not inspect.iscoroutinefunction(self.handler):
            raise TypeError(
                f"Tool {self.name!r} handler {self.handler!r} must be an async function (coroutinefunction)."
            )
        if not self.description:
            object.__setattr__(self, "description", self.handler.__doc__ or "")
        hints = get_type_hints(self.handler, include_extras=True)
        return_hint = hints.get("return")
        if return_hint is not None and return_hint is not str:
            logger.warning(
                "Tool %r handler %r has return annotation %r, expected str. Non-str values will be coerced via str().",
                self.name,
                getattr(self.handler, "__qualname__", self.handler),
                return_hint,
            )
        param_hints = {k: v for k, v in hints.items() if k != "return"}
        try:
            sig = inspect.signature(self.handler)
        except (ValueError, TypeError):
            sig = None
        fields: dict[str, tuple[Any, FieldInfo | None]] = {}
        for name, hint in param_hints.items():
            fi = get_field_info(hint)
            if fi is None and sig is not None and name in sig.parameters:
                param = sig.parameters[name]
                if param.default is not inspect.Parameter.empty:
                    fi = FieldInfo(default=param.default)
            fields[name] = (hint, fi)
        param_fields = MappingProxyType(fields)
        if not self.schema:
            object.__setattr__(self, "schema", MappingProxyType(build_tool_schema(self.handler, hints=param_hints)))
        object.__setattr__(self, "_fields", param_fields)
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
        return dict(self.schema)

    async def __call__(self, **kwargs: Any) -> Any:
        async with self._acquire():
            # 1. Inject defaults and validate - guards see fully materialised kwargs.
            try:
                for name, (hint, fi) in self._fields.items():
                    if name not in kwargs:
                        if fi is not None and fi.default is not MISSING:
                            kwargs[name] = fi.default
                    elif fi is not None:
                        fi.validate(kwargs[name], name, hint)

                required = self.schema.get("required", [])
                missing = [name for name in required if name not in kwargs]
                if missing:
                    raise HandlerError(f"Missing required field(s): {', '.join(missing)}")
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc

            # 2. Guards run sequentially on materialised kwargs.
            for guard in self.guards:
                try:
                    kwargs = await guard(self, **kwargs)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardError(str(exc)) from exc

            # 3. Execute handler.
            try:
                token = CONTEXT.set(self.context)
                try:
                    return str(await self.handler(**kwargs))
                finally:
                    CONTEXT.reset(token)
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc
