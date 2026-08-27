"""Payload shapes: one class per wire name, read into declared fields."""

from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from functools import cache
from types import UnionType
from typing import Any, ClassVar, Self, Union, get_args, get_origin, get_type_hints

from .event import Payload


class Wire:
    """One payload shape, named by the wire name it arrives under::

        @dataclass(frozen=True, slots=True)
        class OutputTextDelta(Wire, name="response.output_text.delta"):
            delta: str = ""
            output_index: int = 0

    Every field is read by its declared name and type, so a misspelled key is a type error at the
    place that uses it rather than a default quietly standing in for the value. A field the
    provider did not send, sent as null, or sent as the wrong type takes its default. That is what
    an optional provider field is, and one bad field must not lose the whole event.

    A nested object is another ``Wire``; a list of them is ``list[ThatWire]``. Give a shape no
    ``name=`` and it is only ever nested, never dispatched to.

    Declare a field ``raw: Payload`` and it receives the whole payload, for a shape that varies too
    much to declare whole. A citation arrives under five shapes and each names its span
    differently, so the fields worth reading are declared and the rest travels beside them.

    Declaring a shape registers it nowhere. A ``Reader`` claims it with ``@on(ThatShape)``.
    """

    #: Every name this shape arrives under, from ``name=`` and ``also=`` on the class line.
    names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, *, name: str = "", also: str | Iterable[str] = (), **rest: object) -> None:
        super().__init_subclass__(**rest)
        if also and not name:
            raise ValueError(f"{cls.__name__} gives also= without name=; a shape names itself whole")
        if name:
            # Replaces rather than extends: a renamed subclass must not keep its parent's names.
            cls.names = (name, *((also,) if isinstance(also, str) else also))

    @classmethod
    def read(cls, payload: Payload) -> Self:
        """This payload as this shape. Extra keys are ignored, missing ones take their defaults."""
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass, so it has no fields to read into")
        hints = _hints(cls)
        made: dict[str, Any] = {}
        for field in fields(cls):
            if field.name == "raw" and hints[field.name] is Payload:
                made[field.name] = payload
                continue
            if field.name not in payload:
                continue
            value = _as(hints[field.name], payload[field.name])
            if value is not None:
                made[field.name] = value
        return cls(**made)


@cache
def _hints(cls: type) -> Mapping[str, Any]:
    """The declared types of one shape, worked out once.

    Annotations do not change, and every transport uses ``from __future__ import annotations``, so
    without this each event re-evaluates every annotation from its string form. Measured on a real
    text delta that was nine tenths of the cost of reading the event.
    """
    return get_type_hints(cls)


def _as(kind: Any, raw: Any) -> Any:
    """``raw`` as this declared type, or None where it is not that and the default should stand."""
    origin = get_origin(kind)
    if origin is UnionType or origin is Union:
        rest = [arg for arg in get_args(kind) if arg is not type(None)]
        for member in rest:
            if (read := _as(member, raw)) is not None:
                return read
        return None

    if isinstance(kind, type) and issubclass(kind, Wire):
        return kind.read(Payload(raw)) if isinstance(raw, dict) else None
    if origin is list:
        if not isinstance(raw, list):
            return None
        args = get_args(kind)
        if not args:
            return list(raw)
        read = [_as(args[0], one) for one in raw]
        return [one for one in read if one is not None]
    if kind is str:
        return raw if isinstance(raw, str) else None
    # bool is an int in Python, so each has to refuse the other.
    if kind is bool:
        return raw if isinstance(raw, bool) else None
    if kind is int:
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    if kind is float:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return None
        try:
            return float(raw)
        except OverflowError:
            # A JSON integer is unbounded and float() is not, so a value the caller cannot represent
            # takes its default.
            return None
    if kind is Payload or kind is dict or origin is dict:
        return Payload(raw) if isinstance(raw, dict) else None
    return raw
