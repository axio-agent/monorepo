"""Read ``text/event-stream``: a decoder you feed, and a reader for what its payloads mean.

``Decoder`` is the format and nothing else. Feed it chunks — bytes or text, cut anywhere — and take
the events they completed. It is synchronous and holds no connection, so every wire case is
testable without a loop, and a thread or a non-asyncio caller can drive it. ``events()`` and
``payloads()`` are the async skin over it: chunks in, ``Event`` or ``Payload`` out. Chunks must
carry their line terminators, so an iterator of lines will not do.

A stream whose events are all one shape needs nothing above ``payloads()``: every JSON object the
stream carries, and nothing more to learn. ``until`` names the one data payload that closes the
stream — ``until="[DONE]"`` — so a sentinel that is not JSON never reaches a caller.

A stream that says what each event is subclasses ``Reader`` and writes one ``@on(...)`` method per
event. ``by`` on the class line names the payload key that holds the name, or ``EVENT_NAME`` for
the format's own ``event:`` field. That class body is one endpoint's whole vocabulary, the events
it deliberately drops included. An event no method claims is skipped and logged at DEBUG. It
raises ``UnknownEvent`` when the caller reads with ``strict=True``, which is how a test fails on
the day the provider sends something new.

This module knows nothing about HTTP and imports no client.
"""

from .decoder import Decoder, EventTooLarge
from .event import Event, MalformedPayload, Payload
from .reader import EVENT_NAME, Handled, Reader, UnknownEvent, on
from .stream import events, payloads
from .wire import Wire

__all__ = [
    "EVENT_NAME",
    "Decoder",
    "Event",
    "EventTooLarge",
    "Handled",
    "MalformedPayload",
    "Payload",
    "Reader",
    "UnknownEvent",
    "Wire",
    "events",
    "on",
    "payloads",
]
