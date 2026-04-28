"""Field metadata for tool handler functions - lightweight replacement for pydantic.Field."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Final, get_args, get_origin


class MissingSentinel:
    """Singleton sentinel - distinguishes 'no default' from ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


#: Sentinel value meaning "this field has no default and is required".
MISSING: Final[MissingSentinel] = MissingSentinel()


@dataclass(frozen=True)
class FieldInfo:
    """Metadata attached to a handler parameter via ``Annotated[T, FieldInfo(...)]``."""

    description: str = ""
    default: Any = MISSING
    ge: int | float | None = None  # minimum (≥)
    le: int | float | None = None  # maximum (≤)
    strict: bool = False  # reject implicit type coercion

    def validate(self, value: Any, name: str, hint: Any) -> None:
        """Validate *value* against this FieldInfo's constraints, raising if invalid."""
        if self.strict:
            b = bare_type(hint)
            if not isinstance(value, b):
                raise TypeError(f"Field '{name}' requires {b.__name__}, got {type(value).__name__}")
        if self.ge is not None and value < self.ge:
            raise ValueError(f"Field '{name}' must be >= {self.ge}")
        if self.le is not None and value > self.le:
            raise ValueError(f"Field '{name}' must be <= {self.le}")


# noinspection PyPep8Naming
def Field(
    description: str = "",
    default: Any = MISSING,
    ge: int | float | None = None,
    le: int | float | None = None,
) -> FieldInfo:
    """Annotate a handler parameter with metadata (description, default, constraints).

    Usage::

        async def search(
            query: Annotated[str, Field(description="Search query")],
            limit: Annotated[int, Field(default=10, ge=1, le=100)],
        ) -> str: ...
    """
    return FieldInfo(description=description, default=default, ge=ge, le=le)


#: Drop-in replacement for ``from pydantic import StrictStr``.
#: Rejects non-string values (e.g. integers) without coercion.
StrictStr = Annotated[str, FieldInfo(strict=True)]


def is_classvar(annotation: Any) -> bool:
    """Return True if *annotation* is ``ClassVar`` or ``ClassVar[X]``."""
    return get_origin(annotation) is typing.ClassVar or annotation is typing.ClassVar


def get_field_info(annotation: Any) -> FieldInfo | None:
    """Extract a ``FieldInfo`` from an ``Annotated`` annotation, or return ``None``."""
    if get_origin(annotation) is not typing.Annotated:
        return None
    return next((a for a in get_args(annotation)[1:] if isinstance(a, FieldInfo)), None)


def bare_type(hint: Any) -> type:
    """Return the base Python type, stripping ``Annotated`` and ``Optional`` wrappers."""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is typing.Annotated:
        return bare_type(args[0])
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return bare_type(non_none[0]) if len(non_none) == 1 else object
    return hint if isinstance(hint, type) else object
