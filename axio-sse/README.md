# axio-sse

[![PyPI](https://img.shields.io/pypi/v/axio-sse)](https://pypi.org/project/axio-sse/)
[![Python](https://img.shields.io/pypi/pyversions/axio-sse)](https://pypi.org/project/axio-sse/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Read `text/event-stream`: a decoder you feed, and a reader for what its payloads mean.

The package knows nothing about HTTP and imports no client. It has no dependencies, not even on
[axio](https://github.com/mosquito/axio-agent).

## Installation

```bash
pip install axio-sse
```

## Usage

### `payloads(chunks, *, until="")` — the JSON object of every event

All a stream with no discriminator needs. Comments, keep-alives and junk never arrive. `until` names
the one data payload that closes the stream, so a sentinel that is not JSON never reaches you.

<!-- name: test_readme_payloads -->
```python
import asyncio
from axio_sse import payloads

async def chunks():
    yield b': keep-alive\n\n'
    yield b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    yield b"data: [DONE]"

async def main() -> None:
    got = [p["choices"][0]["delta"]["content"] async for p in payloads(chunks(), until="[DONE]")]
    assert got == ["Hel", "lo"]

asyncio.run(main())
```

Feed it whatever the transport hands you. A chunk may end mid-field, mid-terminator, or mid-UTF-8
sequence. The result does not depend on where it was cut. Chunks must still carry their line
terminators, so `aiter_lines()` will not do. It strips them, and nothing ever dispatches.

A stream that stops without its final blank line still yields what it collected. The example above
ends on `data: [DONE]` with no newline after it. That is how these streams really end.

### `events(chunks, *, until="")` — the wire events themselves

<!-- name: test_readme_events -->
```python
import asyncio
from axio_sse import Event, events

async def chunks():
    yield b'data: {"first":\ndata: true}\n\n'
    yield b"event: named\r\ndata: sec"
    yield b"ond\r\n\r\n"

async def main() -> None:
    assert [e async for e in events(chunks())] == [
        Event(data='{"first":\ntrue}'),
        Event(data="second", event="named"),
    ]

asyncio.run(main())
```

`Event` carries the four fields the format defines — `data`, `event`, `id`, `retry`. An empty
`event` means unnamed, which the format reads as `"message"`. `Event.payload()` gives the JSON
object, or `None` where the event carries none.

`events()` suspends nowhere of its own accord, so it needs no async framework: asyncio, trio and
anyio all drive it. A `yield` in an async generator does not reach the event loop. A caller that
must stay fair to other tasks — a TUI redrawing, a queue being served — therefore says so itself,
with `await asyncio.sleep(0)` in its own loop where it knows what else is waiting.

### `Decoder` — the format, with no loop

`Decoder` is the format and nothing else. It is synchronous and holds no connection. Every wire case
is therefore testable without a loop. A thread or a non-asyncio caller can drive it too. Same shape
as `codecs.IncrementalDecoder`, because the problem is the same: input cut at arbitrary points,
output that only sometimes completes.

<!-- name: test_readme_decoder -->
```python
from axio_sse import Decoder, Event

decoder = Decoder()
assert decoder.decode(b"data: hel") == []
assert decoder.decode(b"lo\n\ndata: wor") == [Event(data="hello")]
assert decoder.decode(b"ld\n\n", final=True) == [Event(data="world")]
```

`final=True` closes the stream. What is still pending is discarded, which the format requires: an
event that never reached its blank line is not dispatched. Dispatched anyway, a connection cut
between a frame and the blank line after it makes a truncated turn read as a finished one.

The package takes chunks and never lines for a reason. `aiohttp`'s `readuntil` raises `LineTooLong`
past 131072 bytes. `LineTooLong` is not a `ClientError`. One large reasoning event kills a turn with
no answer.

### `Wire` — a payload shape

Declare the fields you read. Each is read by its declared name and type. A misspelled key is
therefore a type error at the place that uses it, rather than a default quietly standing in for the
value.

<!-- name: test_readme_reader -->
```python
from dataclasses import dataclass, field
from axio_sse import Payload, Wire

@dataclass(frozen=True, slots=True)
class Usage(Wire):
    """Nested, and never dispatched to: it has no name of its own."""
    output_tokens: int = 0

@dataclass(frozen=True, slots=True)
class ResponseObject(Wire):
    usage: Usage = field(default_factory=Usage)

@dataclass(frozen=True, slots=True)
class OutputTextDelta(Wire, name="response.output_text.delta"):
    delta: str = ""

@dataclass(frozen=True, slots=True)
class Completed(Wire, name="response.completed"):
    response: ResponseObject = field(default_factory=ResponseObject)
```

A field the provider did not send, sent as null, or sent as the wrong type takes its default. That
is what an optional provider field is. One bad field must not lose the whole event. A nested object
is another `Wire`. A list of them is `list[ThatWire]`. Declare a field `raw: Payload` and it
receives the whole payload, for a shape that varies too much to declare whole.

### `Reader` — one method per event

A stream that says what each event is subclasses `Reader` and writes one `@on(...)` method per
event. That class body is one endpoint's whole vocabulary. `by` on the class line names the payload
key that holds the event's name. It defaults to `"type"`.

Give `@on` a shape and the method is handed that shape. Give it names and the method is handed the
`Payload` itself. That is what a method that only forwards an event wants. Declaring a shape for a
payload nobody reads a field of would be a schema written for nothing.

One instance reads one stream. The turn's running totals and id maps live on `self` instead of
travelling through a call. Construct one per response.

<!-- name: test_readme_reader -->
```python
import asyncio
from collections.abc import Iterator
from axio_sse import Reader, on

class Responses(Reader[str]):
    """What the Responses API sends, and what each event becomes."""

    def __init__(self) -> None:
        self.output_tokens = 0

    @on(OutputTextDelta)
    def _text(self, wire: OutputTextDelta) -> Iterator[str]:
        yield wire.delta

    @on(Completed)
    def _completed(self, wire: Completed) -> None:
        self.output_tokens = wire.response.usage.output_tokens

    @on("response.created", "response.in_progress", "response.output_text.done")
    def _expected(self, payload: Payload) -> None:
        """The bookkeeping around the deltas. Named so strict fires only on something new."""

async def chunks():
    yield b'data: {"type":"response.created"}\n\n'
    yield b'data: {"type":"response.output_text.delta","delta":"Hi"}\n\n'
    yield b'data: {"type":"response.completed","response":{"usage":{"output_tokens":7}}}\n\n'

async def main() -> None:
    turn = Responses()
    assert [made async for made in turn.over(chunks())] == ["Hi"]
    assert turn.output_tokens == 7

asyncio.run(main())
```

A handler returns what the event became — an iterable, or `None` where the event only moved that
state. Several names on one method is how a stream that sends one thing under two names is written.
It is also how a group of events that means nothing here is written: a method with only a docstring.
Both stay in the class body, so no second list exists to keep in step with the first.

### `strict` — failing on the day the provider sends something new

An event no method claims is skipped and logged at DEBUG. Read with `strict=True` and it raises
instead. That is what a test holds against the provider's own published list.

<!-- name: test_readme_reader -->
```python
import pytest
from axio_sse import Event, UnknownEvent

assert Responses.names() == {
    "response.output_text.delta",
    "response.completed",
    "response.created",
    "response.in_progress",
    "response.output_text.done",
}

with pytest.raises(UnknownEvent, match="response.refusal.delta"):
    Responses().read(Event(data='{"type":"response.refusal.delta"}'), strict=True)
```

`strict` belongs to the call, not to the reader. A policy that outlived one call would leave a CI
test's strictness set for the next caller.

### `EVENT_NAME` — dispatching on the format's own field

Some streams name the event in the SSE `event:` field rather than in the payload.

<!-- name: test_readme_event_name -->
```python
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from axio_sse import EVENT_NAME, Reader, Wire, on

@dataclass(frozen=True, slots=True)
class BlockDelta(Wire):
    text: str = ""

@dataclass(frozen=True, slots=True)
class ContentBlockDelta(Wire, name="content_block_delta"):
    delta: BlockDelta = field(default_factory=BlockDelta)

class Messages(Reader[str], by=EVENT_NAME):
    @on(ContentBlockDelta)
    def _delta(self, wire: ContentBlockDelta) -> Iterator[str]:
        yield wire.delta.text

async def chunks():
    yield b'event: content_block_delta\ndata: {"delta":{"text":"Hi"}}\n\n'
    yield b"event: ping\ndata: {}\n\n"

async def main() -> None:
    assert [made async for made in Messages().over(chunks())] == ["Hi"]

asyncio.run(main())
```

### `Payload` — reading by path

`Payload` is a `dict`, so `payload["x"]`, `in` and `json.dumps` all still work. The four readers
exist so a handler carries no `Any` and no chain of `.get({})`. Each walks the path and gives the
default wherever a step is missing, null, or the wrong type. That is what an optional provider field
is.

<!-- name: test_readme_payload -->
```python
from axio_sse import Payload

payload = Payload({"message": {"usage": {"input_tokens": 7}}, "output": [{"type": "function_call"}]})

assert payload.number("message", "usage", "input_tokens") == 7
assert payload.number("message", "usage", "output_tokens") == 0
assert payload.number("message", "usage", "output_tokens", default=3) == 3
assert payload.string("message", "role") == ""
assert payload.obj("message", "usage") == {"input_tokens": 7}
assert payload.objs("output") == [{"type": "function_call"}]
assert payload.objs("nothing") == []
```

`number()` never reads a `true` as `1`. `bool` is an `int` in Python, so a flag would otherwise read
as a count and stay unnoticed:

<!-- name: test_readme_payload -->
```python
assert Payload({"flag": True}).number("flag") == 0
```

## License

MIT
