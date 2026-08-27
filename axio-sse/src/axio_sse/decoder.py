"""The format as a state machine, with no I/O and no loop."""

import codecs
from typing import Final

from .event import Event

#: The only line endings the format allows. ``str.splitlines`` breaks on more than these.
ENDINGS: Final = ("\r\n", "\n", "\r")

#: How long a ``retry:`` value may be. ``str.isdigit`` is true for 128 characters ``int()`` refuses,
#: and CPython refuses to parse past 4300 digits, so an unbounded guard raises out of ``decode`` and
#: takes the rest of the turn with it.
_RETRY_DIGITS: Final = 18


class Decoder:
    """The format as a state machine: feed it chunks, take the events they completed.

    Same shape as ``codecs.IncrementalDecoder`` — ``decode(chunk, final)`` and ``reset()`` —
    because the problem is the same: input cut at arbitrary points, output that only sometimes
    completes. It takes chunks and never lines. ``aiohttp``'s ``readuntil`` raises ``LineTooLong``
    past 131072 bytes, and ``LineTooLong`` is not a ``ClientError``. A large reasoning event killed
    a turn with no answer.
    """

    __slots__ = ("_text", "_held", "_trailing_cr", "_opened", "_data", "_event", "_id", "_retry")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the half-read event and the half-read character, ready for another stream."""
        # Incremental: a chunk can end in the middle of a character. ``utf-8-sig`` rather than
        # ``utf-8`` because the format is read as "UTF-8 decode", which strips a leading byte order
        # mark. Left in, the mark makes the first field name ``\ufeffdata``, which is unknown, and
        # the event it opened is lost with nothing logged.
        self._text = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._held = ""
        self._opened = False
        self._trailing_cr = False
        self._data: list[str] = []
        self._event = ""
        self._id = ""
        self._retry: int | None = None

    def decode(self, chunk: bytes | str = b"", final: bool = False) -> list[Event]:
        """Every event this chunk completed. ``final=True`` also dispatches what is left over.

        Without that last call a stream that stops before its final blank line loses its last
        event, and a stream cut mid-character loses the character instead of replacing it.
        """
        text = self._text.decode(chunk) if isinstance(chunk, bytes) else chunk
        if not self._opened:
            # The byte decoder strips the mark for us. A caller feeding str has to be met here.
            text = text.removeprefix("\ufeff")
            self._opened = bool(text) or final
        if final:
            text += self._text.decode(b"", True)
        if self._trailing_cr:
            text = "\r" + text
            self._trailing_cr = False
        if text.endswith("\r") and not final:
            # A chunk can end mid-terminator. Hold the ``\r`` until the next chunk says whether a
            # ``\n`` follows. Read whole, it invents a blank line, which dispatches, and the event
            # arrives cut in two with neither half parseable.
            text, self._trailing_cr = text[:-1], True
        self._held += text

        made: list[Event] = []
        while (split := _split(self._held)) is not None:
            line, self._held = split
            if (event := self._read(line)) is not None:
                made.append(event)
        if final:
            if self._held:
                self._read(self._held)
                self._held = ""
            if self._collected():
                made.append(self._dispatch())
        return made

    def _collected(self) -> bool:
        """Whether a blank line here fires anything.

        The format dispatches on the data buffer and never on the name: an event with a name and no
        data fires nothing. ``retry`` is kept because ``Event.retry`` is public surface and arrives
        with no data by definition.
        """
        return bool(self._data) or self._retry is not None

    def _dispatch(self) -> Event:
        made = Event(data="\n".join(self._data), event=self._event, id=self._id, retry=self._retry)
        # The id survives dispatch, per the format: it is the stream's position, not this event's.
        self._data, self._event, self._retry = [], "", None
        return made

    def _read(self, line: str) -> Event | None:
        if not line:
            # A blank line dispatches, but only if something was collected: a stream of
            # keep-alives must not become a stream of empty events.
            if self._collected():
                return self._dispatch()
            # Nothing fires, and the name is cleared so it cannot leak onto the next event.
            self._data, self._event = [], ""
            return None
        if line.startswith(":"):
            return None  # comment line
        name, _, value = line.partition(":")
        value = value.removeprefix(" ")  # exactly one space, per the format
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\0" not in value:
            self._id = value
        elif name == "retry" and value.isdecimal() and len(value) <= _RETRY_DIGITS:
            self._retry = int(value)
        # Any other field is ignored, which the format requires: it is how it is extended.
        return None


def _split(held: str) -> tuple[str, str] | None:
    """The first complete line and what follows it, or None while none is complete.

    At the same position take the longest ending: splitting ``\\r`` out of ``\\r\\n`` leaves a
    ``\\n`` that reads as a blank line, which dispatches.
    """
    found = [(held.find(ending), -len(ending), ending) for ending in ENDINGS if ending in held]
    if not found:
        return None
    at, _, ending = min(found)
    return held[:at], held[at + len(ending) :]
