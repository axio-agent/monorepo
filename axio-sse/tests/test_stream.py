"""What ``events()`` and ``payloads()`` add over the decoder."""

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import axio_sse
from axio_sse import Event, payloads

#: The fixtures, typed where they are used. Importing these from conftest only resolves
#: when this package is pytest's rootdir, which breaks collection from the repository root.
type Read = Callable[..., Coroutine[Any, Any, list[Event]]]
type Stream = Callable[..., AsyncIterator[bytes | str]]


async def test_the_sentinel_closes_the_stream_and_is_not_an_event(read: Read) -> None:
    got = await read(b'data: {"a":1}\n\ndata: [DONE]\n\ndata: {"b":2}\n\n', until="[DONE]")
    assert got == [Event(data='{"a":1}')]


async def test_the_sentinel_is_seen_even_with_no_terminator_after_it(read: Read) -> None:
    # How OpenAI-shaped streams really end. The old parser dropped this whole tail.
    assert await read(b'data: {"a":1}\n\ndata: [DONE]', until="[DONE]") == [Event(data='{"a":1}')]


async def test_payloads_gives_the_json_objects_and_nothing_else(stream: Stream) -> None:
    chunks = stream(b': keep-alive\n\ndata: {"a":1}\n\ndata: [DONE]\n\n')
    assert [payload async for payload in payloads(chunks, until="[DONE]")] == [{"a": 1}]


def test_the_package_binds_to_no_async_framework() -> None:
    assert "asyncio" not in vars(axio_sse)
