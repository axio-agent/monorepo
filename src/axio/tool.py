"""Tool: frozen dataclass binding a ToolHandler to a name, guard, and concurrency."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from axio.exceptions import GuardError, HandlerError
from axio.permission import PermissionGuard
from axio.types import ToolName

type JSONSchema = dict[str, Any]


class ToolHandler(BaseModel):
    """Base for tool handlers.

    Subclass fields define the input JSON-schema.
    Override ``async def __call__`` to implement execution logic.
    Pydantic provides ``__repr__`` automatically — override for custom display.
    """

    async def __call__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Tool:
    name: ToolName
    description: str
    handler: type[ToolHandler]
    guards: tuple[PermissionGuard, ...] = ()
    concurrency: int | None = None

    _semaphore: asyncio.Semaphore | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
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
        return self.handler.model_json_schema()

    async def __call__(self, **kwargs: Any) -> Any:
        async with self._acquire():
            instance = self.handler.model_validate(kwargs)
            for guard in self.guards:
                try:
                    instance = await guard(instance)
                except GuardError:
                    raise
                except Exception as exc:
                    raise GuardError(str(exc)) from exc
            try:
                return await instance()
            except HandlerError:
                raise
            except Exception as exc:
                raise HandlerError(str(exc)) from exc
