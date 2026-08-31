"""Ways to drive a chunk stream, shared by every test file here."""

from collections.abc import AsyncIterator, Coroutine
from typing import Any, Protocol

import pytest

from axio_sse import Event, events


class Read(Protocol):
    def __call__(self, *chunks: bytes | str, size: int = 0, until: str = "") -> Coroutine[Any, Any, list[Event]]: ...


class Stream(Protocol):
    def __call__(self, *chunks: bytes | str) -> AsyncIterator[bytes | str]: ...


@pytest.fixture
def read() -> Read:
    """Read these chunks through ``events()``, optionally re-cut into ``size``-byte pieces first."""

    async def _read(*chunks: bytes | str, size: int = 0, until: str = "") -> list[Event]:
        if size:
            joined = b"".join(c.encode() if isinstance(c, str) else c for c in chunks)
            chunks = tuple(joined[i : i + size] for i in range(0, len(joined), size))

        async def cut() -> AsyncIterator[bytes | str]:
            for chunk in chunks:
                yield chunk

        return [event async for event in events(cut(), until=until)]

    return _read


@pytest.fixture
def stream() -> Stream:
    """These chunks as the async iterable the readers take."""

    def _stream(*chunks: bytes | str) -> AsyncIterator[bytes | str]:
        async def made() -> AsyncIterator[bytes | str]:
            for chunk in chunks:
                yield chunk

        return made()

    return _stream
