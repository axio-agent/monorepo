"""JSON-schema builder for plain async handler functions.

Produces clean schemas with no ``"title"`` keys - no post-processing needed
in transports.
"""

from __future__ import annotations

import inspect
import types
import typing
from typing import Any, Literal, get_args, get_origin, get_type_hints

from .field import MISSING, FieldInfo, get_field_info, is_classvar

PRIMITIVE: dict[type[Any], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}

VAR_KINDS = frozenset({inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD})


def property_schema(annotation: Any) -> dict[str, Any]:
    """Recursively convert a Python type annotation to a JSON schema fragment."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Annotated[X, FieldInfo(...)] - unwrap and merge metadata
    if origin is typing.Annotated:
        inner_type = args[0]
        field_info = next((a for a in args[1:] if isinstance(a, FieldInfo)), None)
        schema = property_schema(inner_type)
        if field_info:
            if field_info.description:
                schema["description"] = field_info.description
            if field_info.ge is not None:
                schema["minimum"] = field_info.ge
            if field_info.le is not None:
                schema["maximum"] = field_info.le
            if field_info.default is not MISSING:
                schema["default"] = field_info.default
        return schema

    # X | None  or  Optional[X]  (both UnionType and Union)
    if origin is types.UnionType or origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        if len(non_none) == 1:
            base = property_schema(non_none[0])
            if has_none:
                return {"anyOf": [base, {"type": "null"}]}
            return base
        parts = [property_schema(a) for a in non_none]
        if has_none:
            parts.append({"type": "null"})
        return {"anyOf": parts}

    # Literal["a", "b", ...]
    if origin is Literal:
        return {"enum": list(args)}

    # list[X]
    if origin is list:
        item_schema = property_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    # dict  or  dict[K, V]
    if origin is dict or annotation is dict:
        return {"type": "object"}

    # Primitive scalars
    if annotation in PRIMITIVE:
        return {"type": PRIMITIVE[annotation]}

    # Unknown - expose as bare object schema
    return {}


def build_tool_schema(
    fn: Any,
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON schema object for *fn* (a callable or class).

    The schema has the form::

        {
            "type": "object",
            "properties": {"field": {"type": "string"}, ...},
            "required": ["field", ...]   # only when non-empty
        }

    No ``"title"`` keys are emitted anywhere in the schema.

    Parameters
    ----------
    fn:
        A plain async function or a class whose annotations define the fields.
    hints:
        Pre-computed ``get_type_hints(fn, include_extras=True)`` result.
        When supplied, the call to ``get_type_hints`` is skipped.
    """
    if hints is None:
        hints = get_type_hints(fn, include_extras=True)

    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        sig = None

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, hint in hints.items():
        if name.startswith("_") or is_classvar(hint) or name == "return":
            continue
        if sig is not None and name in sig.parameters and sig.parameters[name].kind in VAR_KINDS:
            continue

        prop = property_schema(hint)

        fi = get_field_info(hint)
        has_fi_default = fi is not None and fi.default is not MISSING
        has_sig_default = (
            sig is not None and name in sig.parameters and sig.parameters[name].default is not inspect.Parameter.empty
        )
        has_class_default = isinstance(fn, type) and name in fn.__dict__

        # Emit default from signature when not already set by FieldInfo
        if has_sig_default and "default" not in prop:
            sig_default = sig.parameters[name].default  # type: ignore[union-attr]
            prop["default"] = sig_default

        properties[name] = prop

        if not has_fi_default and not has_sig_default and not has_class_default:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
