"""The format as a state machine, with no I/O and no loop."""

from __future__ import annotations

import codecs
from typing import Final

from .event import Event

#: The only line endings the format allows. ``str.splitlines`` breaks on more than these.
#: ``_take`` hard-codes them; changing this tuple alone changes nothing.
ENDINGS: Final = ("\r\n", "\n", "\r")

#: How large a held piece grows before the next chunk starts a new one. Bounds the number of
#: string headers a fragmented event costs, at no measurable cost to an ordinary read buffer.
_MIN_PIECE = 4096

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
        # What has arrived. ``utf-8-sig`` strips a leading byte order mark, which the format
        # requires. ``_parts`` holds chunks apart until a terminator arrives, because joining each
        # one into the buffer copies the whole held event again.
        self._text = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
        self._parts: list[str] = []
        self._held = ""
        self._start = 0
        self._scan = 0
        self._opened = False
        self._trailing_cr = False

        # What the lines read so far have collected, for the next dispatch.
        self._data: list[str] = []
        self._event = ""
        self._id = ""
        self._retry: int | None = None

    def decode(self, chunk: bytes | str = b"", final: bool = False) -> list[Event]:
        """Every event this chunk completed.

        ``final=True`` closes the stream: what is still pending is discarded, which the format
        requires of an event that never reached its blank line. Without that last call a stream
        cut mid-character keeps the half character instead of replacing it.
        """
        text = self._text_of(chunk, final)
        self._hold(text)
        if not final and "\n" not in text and "\r" not in text:
            return []
        self._join()

        made: list[Event] = []
        while (line := self._take()) is not None:
            if (event := self._read(line)) is not None:
                made.append(event)
        if final:
            self._forget()
        elif self._start and self._start * 2 >= len(self._held):
            self._compact()
        return made

    def _text_of(self, chunk: bytes | str, final: bool) -> str:
        """This chunk as text, with the byte order mark and a split terminator dealt with."""
        if isinstance(chunk, bytes):
            text = self._text.decode(chunk)
        else:
            # Bytes left half a character behind. A final flush keeps a partial mark, so the state
            # is cleared as well as replaced.
            if pending := self._text.getstate()[0]:
                self._text.setstate((b"", 0))
            text = pending.decode("utf-8", "replace") + chunk
        if not self._opened:
            # The byte decoder took one mark already, and a second is data. Stripped here as well,
            # two marks would read differently as bytes than as text.
            if not isinstance(chunk, bytes):
                text = text.removeprefix("\ufeff")
            if text or final:
                self._opened = True
                # Flag 0 means no mark is expected, which the byte decoder cannot know on its own.
                # The pending bytes stay: they are half a character, not a mark.
                self._text.setstate((self._text.getstate()[0], 0))
        if final:
            text += self._text.decode(b"", True)
        if self._trailing_cr:
            text, self._trailing_cr = "\r" + text, False
        if text.endswith("\r") and not final:
            # A chunk can end mid-terminator. Hold the ``\r`` until the next chunk says whether a
            # ``\n`` follows, or it invents a blank line and dispatches half an event.
            text, self._trailing_cr = text[:-1], True
        return text

    def _hold(self, text: str) -> None:
        """Keep this text until a terminator arrives."""
        if not text:
            return
        # Every piece costs a string header, so a byte at a time held forty times its size.
        if self._parts and len(self._parts[-1]) < _MIN_PIECE:
            self._parts[-1] += text
        else:
            self._parts.append(text)

    def _compact(self) -> None:
        """Drop the lines already read, once they outweigh what is left."""
        self._held = self._held[self._start :]
        self._scan -= self._start
        self._start = 0

    def _forget(self) -> None:
        """Discard what never completed, which is what end of file means for this format.

        Dispatched instead, a connection cut between a frame and the blank line after it makes a
        truncated turn read as a finished one.
        """
        self._held, self._start, self._scan = "", 0, 0
        self._data, self._event, self._retry = [], "", None

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
            # The tail carries no terminator, so no later chunk scans it again.
            self._scan = len(held)
            return None
        line = held[self._start : at]
        self._start = self._scan = after
        return line

    def _collected(self) -> bool:
        """Whether a blank line here fires anything.

        The format dispatches on the data buffer, and on nothing else. A name alone fires nothing,
        and so does a ``retry:``, which sets the stream's reconnection time rather than sending
        anything. ``Event.retry`` still reports the value where data arrived beside it.
        """
        return bool(self._data)

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
        elif name == "retry" and value.isascii() and value.isdigit() and len(value) <= _RETRY_DIGITS:
            self._retry = int(value)
        # Any other field is ignored, which the format requires: it is how it is extended.
        return None
