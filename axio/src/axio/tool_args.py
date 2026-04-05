"""Incremental streaming parser for tool call JSON arguments.

Feeds partial JSON chunks (from ``ToolInputDelta.partial_json``) and emits
structured ``ToolField*`` events as top-level object fields are discovered.

Top-level *string* values are decoded (escape sequences resolved, quotes
stripped).  All other top-level values are emitted as raw JSON fragments.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

from axio.events import ToolFieldDelta, ToolFieldEnd, ToolFieldStart

type ToolFieldEvent = ToolFieldStart | ToolFieldDelta | ToolFieldEnd


class State(IntEnum):
    INIT = 0
    OBJ = 1
    KEY = 2
    COLON = 3
    VAL = 4
    STR = 5
    RAW = 6
    AFTER = 7
    ESC = 8
    UESC = 9


ESCAPES: Mapping[str, str] = MappingProxyType(
    {
        "n": "\n",
        "t": "\t",
        "r": "\r",
        "b": "\b",
        "f": "\f",
        '"': '"',
        "\\": "\\",
        "/": "/",
    }
)


class ToolArgStream:
    """O(1)-per-character streaming parser for tool argument JSON.

    Usage::

        stream = ToolArgStream("call_1")
        events = stream.feed('{"path":"/tmp/f')
        # [ToolFieldStart(0, "call_1", "path"), ToolFieldDelta(0, "call_1", "path", "/tmp/f")]
        events = stream.feed('oo.py"}')
        # [ToolFieldDelta(0, "call_1", "path", "oo.py"), ToolFieldEnd(0, "call_1", "path")]
    """

    __slots__ = (
        "_id",
        "_idx",
        "_st",
        "_key_chars",
        "_key",
        "_buf",
        "_u",
        "_high",
        "_depth",
        "_raw_str",
        "_raw_esc",
        "_esc_key",
        "_esc_ret",
        "_events",
    )

    def __init__(self, tool_use_id: str, index: int = 0) -> None:
        self._id = tool_use_id
        self._idx = index
        self._st = State.INIT
        self._key_chars: list[str] = []
        self._key = ""
        self._buf: list[str] = []
        self._u: list[str] = []
        self._high = 0
        self._depth = 0
        self._raw_str = False
        self._raw_esc = False
        self._esc_key = False
        self._esc_ret = State.KEY
        self._events: list[ToolFieldEvent] = []

    @property
    def current_key(self) -> str:
        """The field currently being streamed, or ``""``."""
        return self._key

    def feed(self, chunk: str) -> list[ToolFieldEvent]:
        """Process a partial JSON chunk and return any field events produced."""
        self._events = []
        for ch in chunk:
            self._step(ch)
        self._flush()
        return self._events

    def _flush(self) -> None:
        if self._buf:
            self._events.append(ToolFieldDelta(self._idx, self._id, self._key, "".join(self._buf)))
            self._buf.clear()

    def _start(self) -> None:
        self._flush()
        self._events.append(ToolFieldStart(self._idx, self._id, self._key))

    def _end(self) -> None:
        self._flush()
        self._events.append(ToolFieldEnd(self._idx, self._id, self._key))

    def _step(self, ch: str) -> None:  # noqa: PLR0912
        match self._st:
            case State.INIT:
                if ch == "{":
                    self._st = State.OBJ

            case State.OBJ:
                if ch == '"':
                    self._key_chars.clear()
                    self._st = State.KEY
                elif ch == "}":
                    self._st = State.INIT

            case State.KEY:
                if ch == "\\":
                    self._esc_key = True
                    self._esc_ret = State.KEY
                    self._st = State.ESC
                elif ch == '"':
                    self._key = "".join(self._key_chars)
                    self._st = State.COLON
                else:
                    self._key_chars.append(ch)

            case State.COLON:
                if ch == ":":
                    self._start()
                    self._st = State.VAL

            case State.VAL:
                if ch in " \t\r\n":
                    pass
                elif ch == '"':
                    self._st = State.STR
                else:
                    self._buf.append(ch)
                    self._depth = 1 if ch in "{[" else 0
                    self._raw_str = False
                    self._raw_esc = False
                    self._st = State.RAW

            case State.STR:
                if ch == "\\":
                    self._esc_key = False
                    self._esc_ret = State.STR
                    self._st = State.ESC
                elif ch == '"':
                    if self._high:
                        self._buf.append("\ufffd")
                        self._high = 0
                    self._end()
                    self._st = State.AFTER
                else:
                    if self._high:
                        self._buf.append("\ufffd")
                        self._high = 0
                    self._buf.append(ch)

            case State.RAW:
                if self._raw_str:
                    self._buf.append(ch)
                    if self._raw_esc:
                        self._raw_esc = False
                    elif ch == "\\":
                        self._raw_esc = True
                    elif ch == '"':
                        self._raw_str = False
                elif self._depth == 0 and ch in " \t\r\n,}":
                    # simple value (number/bool/null) ends on whitespace or delimiter
                    self._end()
                    self._st = State.AFTER
                    if ch in ",}":
                        self._step(ch)  # reprocess delimiter
                elif ch == '"':
                    self._buf.append(ch)
                    self._raw_str = True
                elif ch in "{[":
                    self._buf.append(ch)
                    self._depth += 1
                elif ch in "}]":
                    self._buf.append(ch)
                    self._depth -= 1
                    if self._depth == 0:
                        self._end()
                        self._st = State.AFTER
                else:
                    self._buf.append(ch)

            case State.AFTER:
                if ch == ",":
                    self._st = State.OBJ
                elif ch == "}":
                    self._st = State.INIT

            case State.ESC:
                if ch == "u":
                    self._u.clear()
                    self._st = State.UESC
                else:
                    dec = ESCAPES.get(ch, ch)
                    if self._esc_key:
                        self._key_chars.append(dec)
                    else:
                        self._buf.append(dec)
                    self._st = self._esc_ret

            case State.UESC:
                self._u.append(ch)
                if len(self._u) == 4:
                    code = int("".join(self._u), 16)
                    if self._esc_key:
                        self._key_chars.append(chr(code))
                    elif self._high:
                        if 0xDC00 <= code <= 0xDFFF:
                            full = 0x10000 + (self._high - 0xD800) * 0x400 + (code - 0xDC00)
                            self._buf.append(chr(full))
                        else:
                            self._buf.append("\ufffd")
                            self._buf.append(chr(code))
                        self._high = 0
                    elif 0xD800 <= code <= 0xDBFF:
                        self._high = code
                    else:
                        self._buf.append(chr(code))
                    self._st = self._esc_ret
