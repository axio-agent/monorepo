"""The async skin over the decoder: chunks in, events or payloads out."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

from .decoder import Decoder
from .event import Event, Payload


async def events(chunks: AsyncIterable[bytes | str], *, until: str = "") -> AsyncIterator[Event]:
    """Every event in this stream, as the chunks arrive.

    Chunks must carry their line terminators: ``aiter_lines()`` strips them, so nothing dispatches.
    A stream that stops without its final blank line still yields what it collected. ``until``
    names the data payload that closes the stream, and is not yielded.
    """
    decoder = Decoder()
    async for chunk in chunks:
        for event in decoder.decode(chunk):
            if until and event.data == until:
                return
            yield event
    for event in decoder.decode(final=True):
        if until and event.data == until:
            return
        yield event


async def payloads(chunks: AsyncIterable[bytes | str], *, until: str = "") -> AsyncIterator[Payload]:
    """The JSON object of every event in this stream. Comments, keep-alives and junk do not arrive.

    All a stream with no discriminator needs: its events are one shape, read field by field.
    """
    async for event in events(chunks, until=until):
        if (payload := event.payload()) is not None:
            yield payload
