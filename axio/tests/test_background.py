import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from axio import background
from axio.agent import Agent
from axio.blocks import ToolUseBlock
from axio.testing import StubTransport
from axio.tool import BACKGROUND_PARAM, Tool


@pytest.fixture(autouse=True)
async def clean_registry() -> AsyncGenerator[None, None]:
    yield
    await background.cancel_all()


async def slow_echo(text: str, delay: float = 0.0) -> str:
    await asyncio.sleep(delay)
    return f"echo: {text}"


def _agent(tool: Tool[Any]) -> Agent:
    return Agent(system="test", tools=[tool], transport=StubTransport())


def test_every_tool_advertises_the_argument() -> None:
    tool = Tool[Any](name="slow_echo", handler=slow_echo)
    schema = tool.input_schema
    assert BACKGROUND_PARAM in schema["properties"]
    # The tool's own arguments must survive alongside it.
    assert "text" in schema["properties"]


def test_the_argument_is_not_stored_on_the_tool_fields() -> None:
    # It is consumed by the dispatcher; a handler must never receive it.
    tool = Tool[Any](name="slow_echo", handler=slow_echo)
    assert BACKGROUND_PARAM not in tool._fields


@pytest.mark.asyncio
async def test_ordinary_call_is_unaffected() -> None:
    agent = _agent(Tool[Any](name="slow_echo", handler=slow_echo))
    block = ToolUseBlock(id="1", name="slow_echo", input={"text": "hi"})
    [result] = await agent.dispatch_tools([block], 0)
    assert result.content == "echo: hi"
    assert not background.snapshot()


@pytest.mark.asyncio
async def test_detached_call_returns_a_handle_and_finishes_later() -> None:
    agent = _agent(Tool[Any](name="slow_echo", handler=slow_echo))
    block = ToolUseBlock(id="1", name="slow_echo", input={"text": "later", "delay": 0.05, BACKGROUND_PARAM: True})
    [result] = await agent.dispatch_tools([block], 0)
    assert "started in the background" in str(result.content)

    [call] = background.snapshot()
    assert not call.task.done()
    assert await call.task == "echo: later"
    assert call.state == "done"
    assert call.output() == "echo: later"


@pytest.mark.asyncio
async def test_detached_failure_is_kept_not_swallowed() -> None:
    async def boom() -> str:
        raise RuntimeError("nope")

    agent = _agent(Tool[Any](name="boom", handler=boom))
    block = ToolUseBlock(id="1", name="boom", input={BACKGROUND_PARAM: True})
    await agent.dispatch_tools([block], 0)

    [call] = background.snapshot()
    await asyncio.gather(call.task, return_exceptions=True)
    assert call.state == "failed"
    assert "nope" in call.output()


@pytest.mark.asyncio
async def test_streaming_tool_detaches_instead_of_streaming() -> None:
    async def streamer(text: str) -> str:
        return f"done: {text}"

    async def _stream(text: str) -> AsyncGenerator[tuple[str, str], None]:
        yield "stdout", f"chunk:{text}"

    streamer.stream = _stream  # type: ignore[attr-defined]
    tool: Tool[Any] = Tool(name="streamer", handler=streamer)
    assert tool.supports_streaming
    agent = _agent(tool)

    queue: asyncio.Queue[object] = asyncio.Queue()
    block = ToolUseBlock(id="1", name="streamer", input={"text": "x", BACKGROUND_PARAM: True})
    [result] = await agent._dispatch_tools_streaming([block], 0, queue)  # type: ignore[arg-type]
    # Intercepted before the streaming decision, so nothing was streamed.
    assert "started in the background" in str(result.content)
    assert queue.empty()

    [call] = background.snapshot()
    assert await call.task == "done: x"


@pytest.mark.asyncio
async def test_streaming_tool_still_streams_when_not_detached() -> None:
    async def streamer(text: str) -> str:
        return f"done: {text}"

    async def _stream(text: str) -> AsyncGenerator[tuple[str, str], None]:
        yield "stdout", f"chunk:{text}"

    streamer.stream = _stream  # type: ignore[attr-defined]
    agent = _agent(Tool[Any](name="streamer", handler=streamer))

    queue: asyncio.Queue[object] = asyncio.Queue()
    block = ToolUseBlock(id="1", name="streamer", input={"text": "x"})
    [result] = await agent._dispatch_tools_streaming([block], 0, queue)  # type: ignore[arg-type]
    assert "chunk:x" in str(result.content)
    assert not queue.empty()
