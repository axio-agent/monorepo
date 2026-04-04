"""Incremental streaming parser for tool call JSON arguments.

Feeds partial JSON chunks (from ``ToolInputDelta.partial_json``) and emits
structured ``ToolField*`` events as top-level object fields are discovered
and populated.

Top-level *string* values are decoded (escape sequences resolved, quotes
stripped).  All other top-level values (numbers, booleans, objects, arrays)
are emitted as raw JSON fragments.
"""

from __future__ import annotations

from axio.events import ToolFieldDelta, ToolFieldEnd, ToolFieldStart

type ToolFieldEvent = ToolFieldStart | ToolFieldDelta | ToolFieldEnd


class ToolArgStream:
    """O(1)-per-character JSON state machine for streaming tool arguments.

    Usage::

        stream = ToolArgStream("call_1")
        events = stream.feed('{"path":"/tmp/f')
        # [ToolFieldStart("call_1", "path"), ToolFieldDelta("call_1", "path", "/tmp/f")]
        events = stream.feed('oo.py"}')
        # [ToolFieldDelta("call_1", "path", "oo.py"), ToolFieldEnd("call_1", "path")]
    """

    _ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}

    def __init__(self, tool_use_id: str, index: int = 0) -> None:
        self._tool_use_id = tool_use_id
        self._index = index
        self._st = "INIT"
        self._stack: list[str] = []  # "o" (object) / "a" (array)
        self._key_buf: list[str] = []
        self._current_key = ""
        self._mode = ""  # "" | "raw" | "str"
        self._u_buf: list[str] = []
        self._high = 0  # pending high surrogate for \uXXXX pairs
        self._text_buf: list[str] = []
        self._events: list[ToolFieldEvent] = []

    @property
    def current_key(self) -> str:
        """The field currently being streamed, or ``""``."""
        return self._current_key

    def feed(self, chunk: str) -> list[ToolFieldEvent]:
        """Process a partial JSON *chunk* and return any field events produced."""
        self._events = []
        for ch in chunk:
            self._step(ch)
        self._flush_text()
        return self._events

    # -- output helpers ------------------------------------------------

    def _out(self, ch: str) -> None:
        if self._mode:
            self._text_buf.append(ch)

    def _flush_text(self) -> None:
        if self._text_buf and self._current_key:
            self._events.append(
                ToolFieldDelta(
                    index=self._index,
                    tool_use_id=self._tool_use_id,
                    key=self._current_key,
                    text="".join(self._text_buf),
                )
            )
            self._text_buf.clear()

    def _start_val(self, is_str: bool) -> None:
        if len(self._stack) == 1:
            self._mode = "str" if is_str else "raw"

    def _end_val(self) -> None:
        if len(self._stack) == 1:
            self._flush_text()
            if self._current_key:
                self._events.append(
                    ToolFieldEnd(
                        index=self._index,
                        tool_use_id=self._tool_use_id,
                        key=self._current_key,
                    )
                )
            self._mode = ""

    def _pop(self) -> None:
        self._stack.pop()
        self._end_val()
        self._st = "AFTER" if self._stack else "DONE"

    def _flush_high(self) -> None:
        if self._mode == "str":
            self._text_buf.append("\ufffd")
        elif self._mode == "raw":
            self._text_buf.append(f"\\u{self._high:04x}")
        self._high = 0

    def _emit_uchar(self, code: int) -> None:
        if self._mode == "str":
            self._text_buf.append(chr(code))
        elif self._mode == "raw":
            self._text_buf.append("\\u" + "".join(self._u_buf))

    # -- value / delimiter dispatch ------------------------------------

    def _val_start(self, ch: str) -> None:
        if ch == '"':
            self._start_val(True)
            if self._mode == "raw":
                self._text_buf.append('"')
            self._st = "STR"
        elif ch == "{":
            self._start_val(False)
            self._out("{")
            self._stack.append("o")
            self._st = "OBJ"
        elif ch == "[":
            self._start_val(False)
            self._out("[")
            self._stack.append("a")
            self._st = "ARR"
        elif ch in "-0123456789":
            self._start_val(False)
            self._out(ch)
            self._st = "NUM"
        elif ch in "tfn":
            self._start_val(False)
            self._out(ch)
            self._st = "LIT"

    def _after(self, ch: str) -> None:
        if ch == ",":
            self._out(",")
            top = self._stack[-1] if self._stack else None
            self._st = "OBJ" if top == "o" else "VAL"
        elif ch == "}":
            self._out("}")
            self._pop()
        elif ch == "]":
            self._out("]")
            self._pop()
        else:
            self._out(ch)
            self._st = "AFTER"

    # -- main state machine --------------------------------------------

    def _step(self, ch: str) -> None:
        st = self._st

        if st == "INIT":
            if ch == "{":
                self._stack.append("o")
                self._st = "OBJ"

        elif st == "OBJ":
            if ch == '"':
                self._out('"')
                self._key_buf.clear()
                self._st = "KEY"
            elif ch == "}":
                self._out("}")
                self._pop()
            else:
                self._out(ch)

        elif st == "KEY":
            if ch == "\\":
                self._out("\\")
                self._st = "KESC"
            elif ch == '"':
                if len(self._stack) == 1:
                    self._current_key = "".join(self._key_buf)
                self._out('"')
                self._st = "COL"
            else:
                self._key_buf.append(ch)
                self._out(ch)

        elif st == "KESC":
            if ch == 'u':
                self._out(ch)
                self._u_buf.clear()
                self._st = "KUESC"
            else:
                self._key_buf.append(self._ESC.get(ch, ch))
                self._out(ch)
                self._st = "KEY"

        elif st == "KUESC":
            self._u_buf.append(ch)
            self._out(ch)
            if len(self._u_buf) == 4:
                code = int("".join(self._u_buf), 16)
                self._key_buf.append(chr(code))
                self._st = "KEY"

        elif st == "COL":
            if ch == ":":
                if len(self._stack) == 1:
                    self._flush_text()
                    self._events.append(
                        ToolFieldStart(
                            index=self._index,
                            tool_use_id=self._tool_use_id,
                            key=self._current_key,
                        )
                    )
                else:
                    self._out(":")
                self._st = "VAL"
            else:
                self._out(ch)

        elif st == "VAL":
            if ch in " \t\r\n":
                self._out(ch)
            else:
                self._val_start(ch)

        elif st == "ARR":
            if ch in " \t\r\n":
                self._out(ch)
            elif ch == "]":
                self._out("]")
                self._pop()
            else:
                self._val_start(ch)

        elif st == "STR":
            if ch == "\\":
                self._st = "SESC"
            elif ch == '"':
                if self._high:
                    self._flush_high()
                if self._mode == "raw":
                    self._text_buf.append('"')
                self._end_val()
                self._st = "AFTER"
            else:
                if self._high:
                    self._flush_high()
                self._out(ch)

        elif st == "SESC":
            if self._high and ch != "u":
                self._flush_high()
            if ch == "u":
                self._u_buf.clear()
                self._st = "UESC"
            elif self._mode == "str":
                self._text_buf.append(self._ESC.get(ch, ch))
                self._st = "STR"
            elif self._mode == "raw":
                self._text_buf.append("\\")
                self._text_buf.append(ch)
                self._st = "STR"
            else:
                self._st = "STR"

        elif st == "UESC":
            self._u_buf.append(ch)
            if len(self._u_buf) == 4:
                code = int("".join(self._u_buf), 16)
                if self._high:
                    if 0xDC00 <= code <= 0xDFFF:
                        full = 0x10000 + (self._high - 0xD800) * 0x400 + (code - 0xDC00)
                        if self._mode == "str":
                            self._text_buf.append(chr(full))
                        elif self._mode == "raw":
                            self._text_buf.append(f"\\u{self._high:04x}\\u{code:04x}")
                    else:
                        self._flush_high()
                        self._emit_uchar(code)
                    self._high = 0
                elif 0xD800 <= code <= 0xDBFF:
                    self._high = code
                else:
                    self._emit_uchar(code)
                self._st = "STR"

        elif st == "NUM":
            if ch in "0123456789.eE+-":
                self._out(ch)
            else:
                self._end_val()
                self._after(ch)

        elif st == "LIT":
            if ch.isalpha():
                self._out(ch)
            else:
                self._end_val()
                self._after(ch)

        elif st == "AFTER":
            self._after(ch)
