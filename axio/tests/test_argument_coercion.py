from typing import Any

import pytest

from axio.exceptions import HandlerError
from axio.tool import Tool


async def _takes_list(items: list[str]) -> str:
    return ",".join(items)


async def _takes_int(start_line: int = 1) -> str:
    return f"{start_line}:{type(start_line).__name__}"


async def _takes_flag(regex: bool = False) -> str:
    return f"{regex}:{type(regex).__name__}"


async def _takes_optional_list(tasks: list[str] | None = None) -> str:
    return ",".join(tasks or [])


@pytest.mark.asyncio
async def test_json_encoded_array_is_accepted() -> None:
    # The mistake this forgives: '["a","b"]' instead of ["a","b"].
    tool: Tool[Any] = Tool(name="t", handler=_takes_list)
    assert await tool(items='["a", "b"]') == "a,b"


@pytest.mark.asyncio
async def test_json_encoded_array_on_an_optional_field() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_optional_list)
    assert await tool(tasks='["x"]') == "x"


@pytest.mark.asyncio
async def test_numeric_string_becomes_a_number() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_int)
    assert await tool(start_line="10") == "10:int"


@pytest.mark.asyncio
async def test_boolean_string_becomes_a_bool() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_flag)
    assert await tool(regex="true") == "True:bool"


@pytest.mark.asyncio
async def test_real_values_are_untouched() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_int)
    assert await tool(start_line=7) == "7:int"


@pytest.mark.asyncio
async def test_nonsense_still_fails() -> None:
    # Coercion forgives a wrong encoding, not a wrong value.
    tool: Tool[Any] = Tool(name="t", handler=_takes_int)
    with pytest.raises(HandlerError):
        await tool(start_line="not a number")


@pytest.mark.asyncio
async def test_a_plain_string_is_not_wrapped_into_a_list() -> None:
    tool: Tool[Any] = Tool(name="t", handler=_takes_list)
    with pytest.raises(HandlerError):
        await tool(items="just text")
