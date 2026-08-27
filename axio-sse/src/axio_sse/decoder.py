"""The format as a state machine, with no I/O and no loop."""

from __future__ import annotations

import codecs
from typing import Final

from .event import Event

#: The only line endings the format allows. ``str.splitlines`` breaks on more than these.
#: ``_take`` hard-codes them; changing this tuple alone changes nothing.
ENDINGS: Final = ("\r\n", "\n", "\r")

#: How long a ``retry:`` value may be. ``str.isdigit`` is true for 128 characters ``int()``
#: refuses, and CPython refuses to parse past 4300 digits.
_RETRY_DIGITS: Final = 18


class Decoder:
    """The format as a state machine: feed it chunks, take the events they completed.

    Same shape as ``codecs.IncrementalDecoder``: ``decode(chunk, final)`` and ``reset()``. The
    problem is the same one. Input is cut at arbitrary points, and output only sometimes
    completes. It takes chunks and never lines. ``aiohttp``'s ``readuntil`` raises ``LineTooLong``
    past 131072 bytes, and ``LineTooLong`` is not a ``ClientError``. A large reasoning event killed
    a turn with no answer.

    Held text costs time for its size, and never for its square. Chunks with no terminator wait
    in a list. A read line is left behind rather than sliced out. A scanned tail is never scanned
    twice.
    """

    __slots__ = (
        "_text",
        "_held",
        "_parts",
        "_start",
        "_scan",
        "_trailing_cr",
        "_opened",
        "_data",
        "_event",
        "_id",
        "_retry",
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the half-read event and the half-read character, ready for another stream."""
        # Incremental: a chunk can end in the middle of a character. ``utf-8-sig`` because the
        # format is read as "UTF-8 decode", which strips a leading byte order mark. Left in, the
        # mark makes the first field name ``\ufeffdata``.
        self._text = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._held = ""
        # Chunks that carry no terminator, held apart until one arrives. Joining each chunk into
        # the buffer copies the whole held event again, once per chunk.
        self._parts: list[str] = []
        # How much of ``_held`` is read, and how far it is known to carry no terminator.
        self._start = 0
        self._scan = 0
        self._opened = False
        self._trailing_cr = False
        self._data: list[str] = []
        self._event = ""
        self._id = ""
        self._retry: int | None = None

    def decode(self, chunk: bytes | str = b"", final: bool = False) -> list[Event]:
        """Every event this chunk completed. ``final=True`` also dispatches what is left over.

        Without that last call a stream that stops before its final blank line loses its last
        event. A stream cut mid-character loses the character instead of replacing it.
        """
        if isinstance(chunk, bytes):
            text = self._text.decode(chunk)
        else:
            # A caller that switches from bytes to str leaves the byte decoder holding half a character
            # that can never complete. The state is cleared as well as replaced: a final flush leaves a
            # partial BOM in place, and it then corrupts the next byte chunk.
            if pending := self._text.getstate()[0]:
                self._text.setstate((b"", 0))
            text = pending.decode("utf-8", "replace") + chunk
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
            # ``\n`` follows, or it invents a blank line and dispatches half an event.
            text, self._trailing_cr = text[:-1], True
        if text:
            self._parts.append(text)
        if not final and "\n" not in text and "\r" not in text:
            # No terminator arrived, so no line can complete. What is held stays in pieces.
            return []
        self._join()

        made: list[Event] = []
        while (line := self._take()) is not None:
            if (event := self._read(line)) is not None:
                made.append(event)
        if self._start * 2 >= len(self._held) and self._start:
            # Drop what is read once it outweighs what is not. Left until the next terminator
            # arrives, the buffer pins a second copy of the event just handed back.
            self._held = self._held[self._start :]
            self._scan -= self._start
            self._start = 0
        if final:
            if rest := self._held[self._start :]:
                self._read(rest)
            self._held, self._start, self._scan = "", 0, 0
            if self._collected():
                made.append(self._dispatch())
        return made

    def _join(self) -> None:
        """Make the held text one string again, and drop the lines already read."""
        self._parts.insert(0, self._held[self._start :])
        self._held = "".join(self._parts)
        self._parts.clear()
        self._scan -= self._start
        self._start = 0

    def _take(self) -> str | None:
        """The next complete line, or None while none is complete.

        At the same position take the longest ending: splitting ``\\r`` out of ``\\r\\n`` leaves a
        ``\\n`` that reads as a blank line, which dispatches.
        """
        held = self._held
        nl = held.find("\n", self._scan)
        # Look for a ``\r`` only before that ``\n``. A search for the two-character ``\r\n`` runs to
        # the end of an LF-only buffer, at a fraction of the speed of a one-character search.
        cr = held.find("\r", self._scan, len(held) if nl == -1 else nl)
        if cr != -1:
            at, after = cr, cr + 2 if cr + 1 == nl else cr + 1
        elif nl != -1:
            at, after = nl, nl + 1
        else:
            # The tail carries no terminator, so no later chunk scans it again. A trailing ``\r``
            # is held out of the buffer, so it cannot pair with a ``\n`` that arrives next.
            self._scan = len(held)
            return None
        line = held[self._start : at]
        self._start = self._scan = after
        return line

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
            # A blank line dispatches, but only if something was collected: a stream of keep-alives
            # must not become a stream of empty events.
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
