from __future__ import annotations

from typing import Annotated, Any

import pytest

from axio.exceptions import HandlerError
from axio.field import Field, FieldInfo
from axio.tool import Tool


async def _takes_list(items: list[str]) -> str:
    return ",".join(items)


async def _takes_float_list(items: list[float]) -> str:
    return ",".join(str(item) for item in items)


async def _takes_int(start_line: int = 1) -> str:
    return f"{start_line}:{type(start_line).__name__}"


async def _takes_float(temperature: float) -> str:
    return f"{temperature}:{type(temperature).__name__}"


async def _takes_number(value: int | float) -> str:
    return f"{value}:{type(value).__name__}"


async def _takes_int_or_text(value: int | str) -> str:
    return f"{value}:{type(value).__name__}"


async def _takes_list_or_int(value: list[str] | int) -> str:
    return f"{value}:{type(value).__name__}"


async def _takes_flag(regex: bool = False) -> str:
    return f"{regex}:{type(regex).__name__}"


async def _takes_optional_list(tasks: list[str] | None = None) -> str:
    return ",".join(tasks or [])


async def _takes_strict_int(value: Annotated[int, FieldInfo(strict=True)]) -> str:
    return str(value)


async def _takes_bounded_float(value: Annotated[float, Field(ge=0, le=1)]) -> str:
    return str(value)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        ('["a", "b"]', "a,b"),
        ('["x"]', "x"),
    ],
)
async def test_json_encoded_array_is_accepted(encoded: str, expected: str) -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_list)

    assert await tool(items=encoded) == expected


async def test_json_encoded_array_on_an_optional_field() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_optional_list)

    assert await tool(tasks='["x"]') == "x"


async def test_numeric_strings_become_numbers() -> None:
    integer: Tool[Any] = Tool(name="integer", handler=_takes_int)
    floating: Tool[Any] = Tool(name="floating", handler=_takes_float)
    union: Tool[Any] = Tool(name="union", handler=_takes_number)

    assert await integer(start_line="10") == "10:int"
    assert await floating(temperature="0.5") == "0.5:float"
    assert await union(value="0.5") == "0.5:float"


async def test_string_is_preserved_when_it_is_an_allowed_union_branch() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_int_or_text)

    assert await tool(value="10") == "10:str"


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [('["x"]', "['x']:list"), ("10", "10:int")],
)
async def test_mixed_list_and_scalar_union_tries_each_target(encoded: str, expected: str) -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_list_or_int)

    assert await tool(value=encoded) == expected


@pytest.mark.parametrize(("encoded", "expected"), [("true", "True:bool"), (" FALSE ", "False:bool")])
async def test_boolean_strings_become_booleans(encoded: str, expected: str) -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_flag)

    assert await tool(regex=encoded) == expected


async def test_real_values_are_untouched() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_int)

    assert await tool(start_line=7) == "7:int"


async def test_strict_fields_are_not_coerced() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_strict_int)

    with pytest.raises(HandlerError):
        await tool(value="10")


@pytest.mark.parametrize("encoded", ["NaN", "Infinity", "-Infinity"])
async def test_non_finite_float_strings_are_not_coerced(encoded: str) -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_bounded_float)

    with pytest.raises(HandlerError):
        await tool(value=encoded)


@pytest.mark.parametrize("encoded", ["[NaN]", "[Infinity]", "[-Infinity]", "[1e999]"])
async def test_json_arrays_with_non_finite_numbers_are_not_coerced(encoded: str) -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_float_list)

    with pytest.raises(HandlerError):
        await tool(items=encoded)


@pytest.mark.parametrize(
    ("handler", "kwargs"),
    [
        (_takes_int, {"start_line": "not a number"}),
        (_takes_list, {"items": "just text"}),
        (_takes_list, {"items": '{"not": "a list"}'}),
    ],
)
async def test_values_that_do_not_cleanly_convert_still_fail(
    handler: Any,
    kwargs: dict[str, Any],
) -> None:
    tool: Tool[Any] = Tool(name="t", handler=handler)

    with pytest.raises(HandlerError):
        await tool(**kwargs)
