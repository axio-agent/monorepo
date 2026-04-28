"""Field metadata for tool handler functions - lightweight replacement for pydantic.Field."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Final, Literal, get_args, get_origin

# Types for which basic isinstance checks are applied in non-strict mode.
PRIMITIVE_TYPES: frozenset[type] = frozenset({str, int, float, bool, list, dict})


def unwrap_hint(hint: Any) -> tuple[Any, bool]:
    """Strip ``Annotated`` and ``Optional`` wrappers.

    Returns ``(inner_type, is_optional)`` where *inner_type* has no Annotated or
    Union-with-None wrappers and *is_optional* is True when ``None`` was one of
    the Union members.
    """
    inner = get_args(hint)[0] if get_origin(hint) is typing.Annotated else hint
    origin = get_origin(inner)
    if origin is types.UnionType or origin is typing.Union:
        inner_args = get_args(inner)
        non_none = [a for a in inner_args if a is not type(None)]
        is_optional = len(non_none) < len(inner_args)
        inner = non_none[0] if len(non_none) == 1 else inner
        return inner, is_optional
    return inner, False


def check_scalar(value: Any, name: str, b: type, strict: bool) -> None:
    """Raise TypeError when *value* does not satisfy the scalar type *b*."""
    if strict:
        if not isinstance(value, b):
            raise TypeError(f"Field '{name}' requires {b.__name__}, got {type(value).__name__}")
    elif b is float and isinstance(value, int):
        pass  # int is a valid JSON "number"
    elif b in PRIMITIVE_TYPES and value is not None and not isinstance(value, b):
        raise TypeError(f"Field '{name}' requires {b.__name__}, got {type(value).__name__}")


def check_list_items(value: list[Any], name: str, inner: Any) -> None:
    """Raise TypeError when any list element violates the generic item type."""
    item_args = get_args(inner)
    if not item_args:
        return
    item_hint = item_args[0]
    item_b = bare_type(item_hint)
    if item_b not in PRIMITIVE_TYPES:
        return
    for idx, elem in enumerate(value):
        try:
            check_type(elem, name, item_hint, strict=False)
        except (TypeError, ValueError):
            raise TypeError(f"Field '{name}' element {idx} requires {item_b.__name__}, got {type(elem).__name__}")


def check_type(value: Any, name: str, inner: Any, *, strict: bool) -> None:
    """Dispatch type validation for *value* against *inner* (already unwrapped)."""
    origin = get_origin(inner)
    b = bare_type(inner)

    # Multi-branch union (e.g. str | int): value must satisfy at least one branch.
    # unwrap_hint already stripped Optional (| None), so non-None multi-unions land here.
    if b is object and (origin is types.UnionType or origin is typing.Union):
        branches = [br for br in get_args(inner) if br is not type(None)]
        for branch in branches:
            try:
                check_type(value, name, branch, strict=strict)
                return
            except (TypeError, ValueError):
                continue
        branch_names = " | ".join(getattr(bare_type(br), "__name__", str(br)) for br in branches)
        raise TypeError(f"Field '{name}' requires {branch_names}, got {type(value).__name__}")

    # bool is an int subclass - reject it for non-bool hints before numeric dispatch.
    if isinstance(value, bool) and b is not bool:
        raise TypeError(f"Field '{name}' requires {b.__name__}, got bool")

    if origin is Literal:
        allowed = get_args(inner)
        if value not in allowed:
            raise ValueError(f"Field '{name}' must be one of {list(allowed)!r}, got {value!r}")
        return

    check_scalar(value, name, b, strict)

    if origin is list and isinstance(value, list):
        check_list_items(value, name, inner)


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
        inner, is_optional = unwrap_hint(hint)
        if value is None and is_optional:
            return
        check_type(value, name, inner, strict=self.strict)
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
    """Return the base Python type, stripping ``Annotated``, ``Optional``, and generic wrappers."""
    origin = get_origin(hint)
    args = get_args(hint)
    if origin is typing.Annotated:
        return bare_type(args[0])
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return bare_type(non_none[0]) if len(non_none) == 1 else object
    if origin is not None:
        # Generic alias: list[int] → list, dict[str, Any] → dict, etc.
        return origin if isinstance(origin, type) else object
    return hint if isinstance(hint, type) else object
