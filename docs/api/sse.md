# `axio-sse`

Read `text/event-stream`: a decoder you feed, and a reader for what its payloads mean.

The package knows nothing about HTTP and imports no client. It has no dependencies, not
even on `axio`.

Two shapes of stream, two ways to read one. A stream whose events are all one shape needs
nothing above `payloads()`. A stream that says what each event is subclasses `Reader` and
writes one `@on(...)` method per event.

<!-- name: test_sse_decoder_and_reader -->
```python
from collections.abc import Iterator
from dataclasses import dataclass

from axio_sse import Decoder, Event, Handled, Payload, Reader, Wire, on

# Chunks are cut wherever the network cut them; nothing dispatches until the blank line.
decoder = Decoder()
assert decoder.decode('event: delta\ndata: {"type": "message.delta", ') == []
events = decoder.decode('"text": "hi"}\n\n')
assert [event.event for event in events] == ["delta"]


@dataclass(frozen=True, slots=True)
class Delta(Wire, name="message.delta"):
    text: str = ""
    index: int = 0


class Messages(Reader[str]):
    @on(Delta)
    def _delta(self, wire: Delta) -> Iterator[str]:
        yield wire.text

    def unmatched(self, name: str, payload: Payload) -> Handled[str]:
        return [f"<{name}>"]


reader = Messages()
assert Messages.names() == {"message.delta"}
assert reader.read(events[0]) == ["hi"]
# Not named, so not interpreted - and forwarded rather than dropped.
assert reader.read(Event(data='{"type": "message.tool.started"}')) == ["<message.tool.started>"]
```

## What a `Reader` names, and what it does not

A `Reader` names only the events it interprets. Everything else reaches `unmatched()`,
which returns nothing by default. A reader overrides it to forward instead of drop. Both
readers in this repository do that, as `ProviderEvent` under the provider's own name.

The reason is not obvious. The instinct it contradicts - add a handler for every event
in the provider's documentation - is the wrong one. An endpoint that runs tools
publishes one event family per tool. That set therefore depends on which tools exist and
which the caller declared, not on the protocol. Named one by one, the list is stale the
day a tool is added. A new tool then reads as news about the protocol when it is news
about the tools.

Reading with `strict=True` still raises `UnknownEvent` for any name no method claims. A
test can therefore hold `names()` against the schema the provider publishes, without the
reader carrying a list it cannot keep true.

## Reading a stream

```{eval-rst}
.. autofunction:: axio_sse.payloads
```

```{eval-rst}
.. autofunction:: axio_sse.events
```

```{eval-rst}
.. autoclass:: axio_sse.Event
   :members:
```

## The format, with no loop

```{eval-rst}
.. autoclass:: axio_sse.Decoder
   :members:
```

## What the payloads mean

```{eval-rst}
.. autoclass:: axio_sse.Wire
   :members:
```

```{eval-rst}
.. autoclass:: axio_sse.Reader
   :members:
```

```{eval-rst}
.. autofunction:: axio_sse.on
```

`EVENT_NAME`
: The `by=` sentinel, `"event:"`. `by=EVENT_NAME` dispatches on the format's own `event:`
  field. Any other `by` names a key inside the payload. A JSON key holding a colon is not
  a name a provider gives a field, so the two can never mean each other.

`Handled[T]`
: What a handler returns: `Iterable[T] | None`. One rule for none, one, and many — an
  event that only moved the reader's own state returns nothing.

```{eval-rst}
.. autoclass:: axio_sse.Payload
   :members:
```

```{eval-rst}
.. autoexception:: axio_sse.UnknownEvent
```
