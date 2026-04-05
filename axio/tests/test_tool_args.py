"""Tests for ToolArgStream."""

from __future__ import annotations

import json
import time

import pytest

from axio.events import ToolFieldDelta, ToolFieldEnd, ToolFieldStart
from axio.tool_args import ToolArgStream, ToolFieldEvent

# ── Helpers ────────────────────────────────────────────────────────────────────


def collect(payload: str, chunk_size: int | None = None) -> dict[str, str]:
    """Feed payload through a fresh stream, return {key: decoded_text}."""
    s = ToolArgStream("c")
    events: list[ToolFieldEvent] = []
    if chunk_size:
        for i in range(0, len(payload), chunk_size):
            events.extend(s.feed(payload[i : i + chunk_size]))
    else:
        events = s.feed(payload)
    buf: dict[str, list[str]] = {}
    for e in events:
        if isinstance(e, ToolFieldStart):
            buf[e.key] = []
        elif isinstance(e, ToolFieldDelta):
            buf[e.key].append(e.text)
    return {k: "".join(v) for k, v in buf.items()}


def S(key: str, idx: int = 0, tid: str = "c1") -> ToolFieldStart:
    return ToolFieldStart(idx, tid, key)


def D(key: str, text: str, idx: int = 0, tid: str = "c1") -> ToolFieldDelta:
    return ToolFieldDelta(idx, tid, key, text)


def E(key: str, idx: int = 0, tid: str = "c1") -> ToolFieldEnd:
    return ToolFieldEnd(idx, tid, key)


def feed(json: str, tid: str = "c1") -> list[ToolFieldEvent]:
    return ToolArgStream(tid).feed(json)


# ── Basic ──────────────────────────────────────────────────────────────────────


class TestBasic:
    def test_single_string(self) -> None:
        assert feed('{"path": "/tmp/foo.py"}') == [
            S("path"),
            D("path", "/tmp/foo.py"),
            E("path"),
        ]

    def test_two_fields(self) -> None:
        assert feed('{"a": "x", "b": "y"}') == [
            S("a"),
            D("a", "x"),
            E("a"),
            S("b"),
            D("b", "y"),
            E("b"),
        ]

    def test_empty_object(self) -> None:
        assert feed("{}") == []

    def test_current_key(self) -> None:
        s = ToolArgStream("c1")
        s.feed('{"path": "/a"}')
        assert s.current_key == "path"

    def test_index_propagated(self) -> None:
        events = ToolArgStream("c1", index=3).feed('{"k": "v"}')
        assert all(e.index == 3 for e in events)


# ── Chunked ────────────────────────────────────────────────────────────────────


class TestChunked:
    def test_split_mid_value(self) -> None:
        s = ToolArgStream("c1")
        e1 = s.feed('{"path":"/tmp/f')
        e2 = s.feed('oo.py"}')
        assert e1 == [S("path"), D("path", "/tmp/f")]
        assert e2 == [D("path", "oo.py"), E("path")]

    def test_split_mid_key(self) -> None:
        s = ToolArgStream("c1")
        e1 = s.feed('{"pa')
        e2 = s.feed('th":"v"}')
        assert e1 == []
        assert e2 == [S("path"), D("path", "v"), E("path")]

    def test_one_char_at_a_time(self) -> None:
        """Char-by-char and batch produce same semantic output (Delta batching may differ)."""

        def summarise(events: list[ToolFieldEvent]) -> dict[str, object]:
            starts = {e.key for e in events if isinstance(e, ToolFieldStart)}
            ends = {e.key for e in events if isinstance(e, ToolFieldEnd)}
            text = {
                key: "".join(e.text for e in events if isinstance(e, ToolFieldDelta) and e.key == key)
                for key in starts
            }
            return {"starts": starts, "ends": ends, "text": text}

        full = ToolArgStream("c1").feed('{"k":"hello"}')
        char = ToolArgStream("c1")
        events: list[ToolFieldEvent] = []
        for ch in '{"k":"hello"}':
            events.extend(char.feed(ch))
        assert summarise(events) == summarise(full)

    def test_empty_chunks(self) -> None:
        s = ToolArgStream("c1")
        s.feed("")
        s.feed("")
        events = s.feed('{"x":"y"}')
        assert events == [S("x"), D("x", "y"), E("x")]


# ── Raw values (non-string) ────────────────────────────────────────────────────


class TestRaw:
    def test_integer(self) -> None:
        events = feed('{"n": 42}')
        assert events == [S("n"), D("n", "42"), E("n")]

    def test_float(self) -> None:
        events = feed('{"x": 3.14}')
        assert events == [S("x"), D("x", "3.14"), E("x")]

    def test_boolean_true(self) -> None:
        events = feed('{"ok": true}')
        assert events == [S("ok"), D("ok", "true"), E("ok")]

    def test_null(self) -> None:
        events = feed('{"v": null}')
        assert events == [S("v"), D("v", "null"), E("v")]

    def test_nested_object(self) -> None:
        events = feed('{"x": {"a": 1}}')
        assert events == [S("x"), D("x", '{"a": 1}'), E("x")]

    def test_array(self) -> None:
        events = feed('{"xs": [1, 2]}')
        assert events == [S("xs"), D("xs", "[1, 2]"), E("xs")]

    def test_string_inside_nested_object(self) -> None:
        # Braces/brackets inside strings within raw values must not confuse depth
        events = feed('{"x": {"k": "a}b"}}')
        assert events == [S("x"), D("x", '{"k": "a}b"}'), E("x")]


# ── Escape sequences ───────────────────────────────────────────────────────────


class TestEscapes:
    def test_simple_escapes(self) -> None:
        events = feed(r'{"s": "a\nb\tc"}')
        assert events == [S("s"), D("s", "a\nb\tc"), E("s")]

    def test_unicode_escape(self) -> None:
        events = feed(r'{"s": "\u0041"}')  # A
        assert events == [S("s"), D("s", "A"), E("s")]

    def test_surrogate_pair(self) -> None:
        events = feed(r'{"s": "\uD83D\uDE00"}')  # 😀
        assert events == [S("s"), D("s", "😀"), E("s")]

    def test_lone_high_surrogate_replaced(self) -> None:
        events = feed('{"s": "\\uD800x"}')
        text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert "\ufffd" in text

    def test_escaped_key(self) -> None:
        events = feed(r'{"k\u0065y": "v"}')  # "key"
        assert events[0] == S("key")

    def test_escape_split_across_chunks(self) -> None:
        s = ToolArgStream("c1")
        e1 = s.feed('{"s": "\\')
        e2 = s.feed('n"}')
        combined = "".join(ev.text for ev in e1 + e2 if isinstance(ev, ToolFieldDelta))
        assert combined == "\n"


# ── Truncated / broken JSON ───────────────────────────────────────────────────


class TestBroken:
    """Parser must never raise; partial output for truncated input is acceptable."""

    @pytest.mark.parametrize(
        "payload",
        [
            "",
            "{",
            '{"',
            '{"k',
            '{"key"',
            '{"key":',
            '{"key": ',
            '{"key": "',
            '{"key": "val',  # truncated mid-string (no closing " or })
            '{"key": "val"',  # missing closing }
            '{"key": 42',  # truncated mid-raw-number
            '{"key": tru',  # truncated mid-literal
            '{"key": nul',
            '{"a": 1, "b":',  # second field truncated after colon
            '{"a": "x", "b": "y',  # second field string truncated
            "[1, 2, 3]",  # not an object
            '"string"',  # not an object
            "42",  # not an object
            "null",
            "{{{{",
            '{"k": "v"} extra',  # trailing garbage — first field already complete
        ],
    )
    def test_no_exception(self, payload: str) -> None:
        s = ToolArgStream("c")
        try:
            s.feed(payload)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"raised {type(exc).__name__}: {exc!r} for payload={payload!r}")

    def test_truncated_string_emits_partial_delta(self) -> None:
        """Truncated stream: Start and partial Delta are emitted, End is not."""
        s = ToolArgStream("c")
        events = s.feed('{"path": "/tmp/f')
        assert any(isinstance(e, ToolFieldStart) for e in events)
        assert any(isinstance(e, ToolFieldDelta) for e in events)
        assert not any(isinstance(e, ToolFieldEnd) for e in events)

    def test_truncated_after_colon_emits_start_only(self) -> None:
        s = ToolArgStream("c")
        events = s.feed('{"key":')
        assert events == [ToolFieldStart(0, "c", "key")]

    def test_truncated_raw_emits_partial(self) -> None:
        s = ToolArgStream("c")
        events = s.feed('{"n": 123')  # no closing }
        starts = [e for e in events if isinstance(e, ToolFieldStart)]
        assert len(starts) == 1 and starts[0].key == "n"
        # no End emitted for incomplete value
        assert not any(isinstance(e, ToolFieldEnd) for e in events)

    def test_resumed_after_truncation(self) -> None:
        """Completing a truncated stream by feeding the rest gives full events."""
        s = ToolArgStream("c")
        s.feed('{"k": "hel')
        events = s.feed('lo"}')
        combined = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert combined == "lo"
        assert any(isinstance(e, ToolFieldEnd) for e in events)

    def test_invalid_escape_sequence(self) -> None:
        """Unknown escape like \\q: must not raise, best-effort passthrough."""
        s = ToolArgStream("c")
        events = s.feed(r'{"s": "\q"}')
        text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert "q" in text  # \q → 'q' (passthrough)

    def test_lone_low_surrogate(self) -> None:
        r"""\\uDC00 with no preceding high surrogate: no crash."""
        s = ToolArgStream("c")
        events = s.feed(r'{"s": "\uDC00x"}')
        text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert "x" in text

    def test_double_high_surrogate(self) -> None:
        r"""\\uD800\\uD800: second high replaces first (no low between them)."""
        s = ToolArgStream("c")
        events = s.feed(r'{"s": "\uD800\uD800x"}')
        text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert "x" in text

    def test_truncated_unicode_escape(self) -> None:
        r"""\\u with fewer than 4 hex digits at end of stream: no crash."""
        for partial in [r"\u", r"\uD", r"\uD8", r"\uD83"]:
            s = ToolArgStream("c")
            s.feed('{"s": "' + partial)  # no closing anything


# ── Round-trip against json.loads ─────────────────────────────────────────────


class TestRoundTrip:
    """The ground truth: whatever json.dumps produces, our parser must decode correctly."""

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "hello",
            "with spaces",
            'has "quotes"',
            "newline\nand\ttab",
            "backslash\\here",
            "unicode 😀 and 日本語",
            "all escapes \n\t\r\b\f",
            "braces { } [ ] inside string",
            "colon : comma , inside string",
            "null bytes \x00 inside",
            "a" * 10_000,
        ],
    )
    def test_string_field(self, s: str) -> None:
        payload = json.dumps({"v": s})
        assert collect(payload)["v"] == s

    @pytest.mark.parametrize(
        "s",
        [
            "",
            "hello",
            'has "quotes"',
            "newline\nand\ttab",
            "a" * 1_000,
        ],
    )
    def test_string_field_chunked(self, s: str) -> None:
        payload = json.dumps({"v": s})
        for chunk_size in (1, 3, 7, 64):
            assert collect(payload, chunk_size)["v"] == s, f"chunk_size={chunk_size}"

    @pytest.mark.parametrize(
        "obj",
        [
            {"n": 0},
            {"n": -1},
            {"n": 42},
            {"n": 1_000_000},
            {"f": 3.14},
            {"f": -2.718},
            {"f": 1e10},
            {"f": 1.5e-3},
            {"b": True},
            {"b": False},
            {"z": None},
        ],
    )
    def test_scalar_raw(self, obj: dict[str, object]) -> None:
        payload = json.dumps(obj)
        key = next(iter(obj))
        raw = collect(payload)[key]
        assert json.loads(raw) == obj[key]

    @pytest.mark.parametrize(
        "obj",
        [
            {"a": []},
            {"a": [1]},
            {"a": [1, 2, 3]},
            {"a": ["x", "y"]},
            {"a": [{"b": 1}, {"c": 2}]},
            {"a": {}},
            {"a": {"b": 1}},
            {"a": {"b": {"c": {"d": 4}}}},
            {"a": [[1, 2], [3, 4]]},
            {"a": {"s": "a}b{c[d]e"}},
        ],
    )
    def test_nested_raw(self, obj: dict[str, object]) -> None:
        payload = json.dumps(obj)
        key = next(iter(obj))
        raw = collect(payload)[key]
        assert json.loads(raw) == obj[key]

    def test_many_fields_all_types(self) -> None:
        obj = {
            "str": "hello world",
            "int": 42,
            "float": 3.14,
            "true": True,
            "false": False,
            "null": None,
            "arr": [1, 2, 3],
            "obj": {"x": 1},
            "empty_str": "",
            "empty_arr": [],
            "empty_obj": {},
        }
        payload = json.dumps(obj)
        got = collect(payload)
        assert got["str"] == "hello world"
        assert json.loads(got["int"]) == 42
        assert json.loads(got["float"]) == 3.14
        assert json.loads(got["true"]) is True
        assert json.loads(got["false"]) is False
        assert json.loads(got["null"]) is None
        assert json.loads(got["arr"]) == [1, 2, 3]
        assert json.loads(got["obj"]) == {"x": 1}
        assert got["empty_str"] == ""
        assert json.loads(got["empty_arr"]) == []
        assert json.loads(got["empty_obj"]) == {}

    def test_split_at_every_position(self) -> None:
        """Chunk boundary at every byte position must give identical result."""
        obj = {"path": "/tmp/foo.py", "content": "line1\nline2\n", "n": 42}
        payload = json.dumps(obj)
        expected = collect(payload)
        for pos in range(1, len(payload)):
            s = ToolArgStream("c")
            events = s.feed(payload[:pos]) + s.feed(payload[pos:])
            got = {}
            buf: dict[str, list[str]] = {}
            for e in events:
                if isinstance(e, ToolFieldStart):
                    buf[e.key] = []
                elif isinstance(e, ToolFieldDelta):
                    buf[e.key].append(e.text)
            got = {k: "".join(v) for k, v in buf.items()}
            assert got == expected, f"mismatch at split pos={pos}"


# ── JSON validity edge cases ───────────────────────────────────────────────────


class TestValidity:
    def test_empty_string_value(self) -> None:
        assert collect('{"k": ""}') == {"k": ""}

    def test_whitespace_heavy(self) -> None:
        assert collect('  {  "k"  :  "v"  }  ') == {"k": "v"}

    def test_whitespace_inside_value_preserved(self) -> None:
        assert collect('{"k": "  spaces  "}') == {"k": "  spaces  "}

    def test_many_fields(self) -> None:
        obj = {f"field_{i}": f"value_{i}" for i in range(50)}
        payload = json.dumps(obj)
        got = collect(payload)
        assert got == {k: v for k, v in obj.items()}

    def test_key_order_preserved(self) -> None:
        """Events must arrive in document order."""
        payload = '{"z": "last", "a": "first", "m": "mid"}'
        s = ToolArgStream("c")
        events = s.feed(payload)
        keys = [e.key for e in events if isinstance(e, ToolFieldStart)]
        assert keys == ["z", "a", "m"]

    def test_start_end_pair_for_every_field(self) -> None:
        obj = {f"f{i}": i for i in range(20)}
        events = ToolArgStream("c").feed(json.dumps(obj))
        starts = [e.key for e in events if isinstance(e, ToolFieldStart)]
        ends = [e.key for e in events if isinstance(e, ToolFieldEnd)]
        assert starts == ends  # same keys, same order

    def test_scientific_notation(self) -> None:
        raw = collect('{"x": 1.5e10}')["x"]
        assert json.loads(raw) == 1.5e10

    def test_negative_zero(self) -> None:
        raw = collect('{"x": -0}')["x"]
        assert json.loads(raw) == 0

    def test_deeply_nested(self) -> None:
        depth = 20
        inner: dict[str, object] = {"leaf": 1}
        for _ in range(depth):
            inner = {"child": inner}
        obj = {"root": inner}
        payload = json.dumps(obj)
        got = json.loads(collect(payload)["root"])
        assert got == inner

    def test_string_with_json_special_chars(self) -> None:
        cases = ['{}[],:"\\ are fine', "null true false 0 1.0"]
        for s in cases:
            assert collect(json.dumps({"v": s}))["v"] == s

    def test_unicode_bmp(self) -> None:
        s = "\u4e2d\u6587"  # 中文
        assert collect(json.dumps({"v": s}))["v"] == s

    def test_unicode_supplementary_plane(self) -> None:
        s = "emoji: 😀🎉🚀"
        assert collect(json.dumps({"v": s}))["v"] == s

    def test_surrogate_pair_split_across_chunks(self) -> None:
        payload = r'{"s": "\uD83D\uDE00"}'  # 😀 as surrogate pair
        assert collect(payload, chunk_size=1)["s"] == "😀"

    def test_escaped_backslash_in_raw(self) -> None:
        # backslash inside a string that's inside a nested object stays raw
        obj = {"x": {"path": "C:\\Users"}}
        payload = json.dumps(obj)
        raw = collect(payload)["x"]
        assert json.loads(raw) == {"path": "C:\\Users"}


# ── Newlines ──────────────────────────────────────────────────────────────────


class TestNewlines:
    r"""JSON encodes newlines as \n, \r, \r\n; also tests literal control chars."""

    def test_lf(self) -> None:
        assert collect(r'{"s": "line1\nline2"}')["s"] == "line1\nline2"

    def test_cr(self) -> None:
        assert collect(r'{"s": "line1\rline2"}')["s"] == "line1\rline2"

    def test_crlf(self) -> None:
        assert collect(r'{"s": "line1\r\nline2"}')["s"] == "line1\r\nline2"

    def test_only_newlines(self) -> None:
        assert collect(r'{"s": "\n\n\n"}')["s"] == "\n\n\n"

    def test_newline_at_chunk_boundary(self) -> None:
        r"""\\n split so backslash is in one chunk, 'n' in the next."""
        s = ToolArgStream("c")
        e1 = s.feed('{"s": "a\\')
        e2 = s.feed('nb"}')
        text = "".join(e.text for e in e1 + e2 if isinstance(e, ToolFieldDelta))
        assert text == "a\nb"

    def test_multiline_string(self) -> None:
        src = "first line\nsecond line\nthird line"
        assert collect(json.dumps({"s": src}))["s"] == src

    def test_mixed_whitespace_escapes(self) -> None:
        r"""All whitespace escapes in one string: \n \r \t \b \f."""
        payload = r'{"s": "\n\r\t\b\f"}'
        assert collect(payload)["s"] == "\n\r\t\b\f"

    def test_escaped_backslash_vs_newline(self) -> None:
        r"""\\\\n is escaped backslash + literal 'n', not a newline."""
        payload = r'{"s": "\\n"}'
        assert collect(payload)["s"] == "\\n"

    def test_escaped_backslash_then_real_newline_escape(self) -> None:
        r"""\\\\\\n → backslash + newline."""
        payload = r'{"s": "\\\n"}'
        assert collect(payload)["s"] == "\\\n"

    def test_newline_inside_nested_raw(self) -> None:
        r"""\\n inside a nested object value stays as \\n in raw output."""
        obj = {"x": {"msg": "line1\nline2"}}
        raw = collect(json.dumps(obj))["x"]
        assert json.loads(raw) == {"msg": "line1\nline2"}


# ── Complex escapes ────────────────────────────────────────────────────────────


class TestComplexEscapes:
    def test_all_simple_escapes_together(self) -> None:
        r"""All six JSON escape sequences: \" \\ \/ \b \f \n \r \t."""
        payload = r'{"s": "\"\\\/\b\f\n\r\t"}'
        got = collect(payload)["s"]
        assert got == '"\\/\b\f\n\r\t'

    def test_escaped_quote_inside_string(self) -> None:
        payload = r'{"s": "say \"hello\""}'
        assert collect(payload)["s"] == 'say "hello"'

    def test_escaped_backslash_chain(self) -> None:
        r"""Four backslashes in JSON (\\\\\\\\) → two backslashes in Python."""
        payload = r'{"s": "\\\\"}'
        assert collect(payload)["s"] == "\\\\"

    @pytest.mark.parametrize(
        "code,expected",
        [
            (r"\u0000", "\x00"),  # null character
            (r"\u001F", "\x1f"),  # control character
            (r"\u007F", "\x7f"),  # DEL
            (r"\u00E9", "\xe9"),  # é
            (r"\u4E2D", "\u4e2d"),  # 中
            (r"\uFFFF", "\uffff"),  # BMP max
        ],
    )
    def test_unicode_codepoints(self, code: str, expected: str) -> None:
        payload = f'{{"s": "{code}"}}'
        assert collect(payload)["s"] == expected

    def test_unicode_escape_split_at_every_position(self) -> None:
        r"""\\u4E2D split 0-5 chars from the backslash."""
        full = r'{"s": "\u4E2D"}'
        for split in range(len(full)):
            s = ToolArgStream("c")
            events = s.feed(full[:split]) + s.feed(full[split:])
            text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
            assert text == "\u4e2d", f"wrong at split={split}"

    def test_surrogate_pair_split_at_every_position(self) -> None:
        r"""\\uD83D\\uDE00 (😀) — chunk boundary at every byte."""
        full = r'{"s": "\uD83D\uDE00"}'
        for split in range(len(full)):
            s = ToolArgStream("c")
            events = s.feed(full[:split]) + s.feed(full[split:])
            text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
            assert text == "😀", f"wrong at split={split}"

    def test_multiple_surrogate_pairs(self) -> None:
        r"""Three emoji as surrogate pairs in sequence."""
        payload = r'{"s": "\uD83D\uDE00\uD83C\uDF89\uD83D\uDE80"}'
        assert collect(payload)["s"] == "😀🎉🚀"

    def test_surrogate_pair_chunked_1_byte(self) -> None:
        r"""Byte-by-byte feed for \\uD83D\\uDE00."""
        full = r'{"s": "\uD83D\uDE00"}'
        s = ToolArgStream("c")
        events: list[ToolFieldEvent] = []
        for ch in full:
            events.extend(s.feed(ch))
        text = "".join(e.text for e in events if isinstance(e, ToolFieldDelta))
        assert text == "😀"

    def test_many_unicode_escapes(self) -> None:
        """1000 unicode escape sequences — correctness and no quadratic cost."""
        chars = "ABCDEFGHIJ"
        escaped = "".join(f"\\u{ord(c):04X}" for c in chars * 100)
        payload = f'{{"s": "{escaped}"}}'
        assert collect(payload)["s"] == chars * 100

    def test_escape_adjacent_to_delimiter(self) -> None:
        r"""Escaped quote immediately before closing brace: {"s": "\""} ."""
        payload = r'{"s": "\""}'
        assert collect(payload)["s"] == '"'

    def test_key_with_escape(self) -> None:
        r"""Key containing \\n and \\t (unusual but valid JSON)."""
        payload = r'{"ke\ny": "v"}'
        events = ToolArgStream("c").feed(payload)
        key = next(e.key for e in events if isinstance(e, ToolFieldStart))
        assert key == "ke\ny"


# ── Benchmark (not a correctness test) ────────────────────────────────────────


class TestPerf:
    """Verify O(1)-per-character behaviour: time should scale linearly."""

    _LIMIT_NS = 2_000  # generous upper bound per character, even on slow CI

    @pytest.mark.parametrize("n", [10_000, 100_000, 1_000_000])
    def test_linear_time_string(self, n: int) -> None:
        payload = '{"data": "' + "x" * n + '"}'
        start = time.perf_counter()
        events = ToolArgStream("c").feed(payload)
        elapsed = time.perf_counter() - start

        assert events[0] == ToolFieldStart(0, "c", "data")
        assert events[-1] == ToolFieldEnd(0, "c", "data")
        assert sum(len(e.text) for e in events if isinstance(e, ToolFieldDelta)) == n

        ns = elapsed / len(payload) * 1e9
        assert ns < self._LIMIT_NS, f"too slow: {ns:.0f} ns/char for n={n}"

    @pytest.mark.parametrize("n", [10_000, 100_000, 1_000_000])
    def test_linear_time_raw(self, n: int) -> None:
        """Raw (non-string) values also O(1) per char."""
        # build {"f0":0,"f1":1,...} with n total chars
        fields = []
        total = 0
        i = 0
        while total < n:
            entry = f'"f{i}":{i}'
            fields.append(entry)
            total += len(entry) + 1  # +1 for comma
            i += 1
        payload = "{" + ",".join(fields) + "}"
        start = time.perf_counter()
        ToolArgStream("c").feed(payload)
        elapsed = time.perf_counter() - start

        ns = elapsed / len(payload) * 1e9
        assert ns < self._LIMIT_NS, f"too slow: {ns:.0f} ns/char for n={n}"

    def test_o1_scaling(self) -> None:
        """time/n must be roughly constant across 3 orders of magnitude (no O(n²))."""
        sizes = [10_000, 100_000, 1_000_000]
        ns_per_char = []
        for n in sizes:
            payload = '{"d": "' + "x" * n + '"}'
            start = time.perf_counter()
            ToolArgStream("c").feed(payload)
            ns_per_char.append((time.perf_counter() - start) / len(payload) * 1e9)

        # ratio between largest and smallest must be < 10× (O(n²) would be ~10000×)
        ratio = max(ns_per_char) / min(ns_per_char)
        assert ratio < 10, (
            f"non-linear scaling detected: {[f'{x:.0f}' for x in ns_per_char]} ns/char (ratio={ratio:.1f}×)"
        )

    @pytest.mark.parametrize("chunk_size", [1, 16, 64, 256])
    def test_chunked_throughput(self, chunk_size: int) -> None:
        """Chunked feeding must stay O(1) per character regardless of chunk size."""
        n = 200_000
        payload = '{"content": "' + "a" * n + '"}'
        start = time.perf_counter()
        collect(payload, chunk_size=chunk_size)
        elapsed = time.perf_counter() - start

        ns = elapsed / len(payload) * 1e9
        assert ns < self._LIMIT_NS, f"too slow at chunk_size={chunk_size}: {ns:.0f} ns/char"

    def test_many_short_fields(self) -> None:
        """100 fields of ~20 chars each."""
        obj = {f"field_{i:03d}": f"value number {i}" for i in range(100)}
        payload = json.dumps(obj)
        start = time.perf_counter()
        got = collect(payload)
        elapsed = time.perf_counter() - start

        assert len(got) == 100
        assert all(got[f"field_{i:03d}"] == f"value number {i}" for i in range(100))
        ns = elapsed / len(payload) * 1e9
        assert ns < self._LIMIT_NS, f"too slow: {ns:.0f} ns/char"

    def test_throughput_report(self) -> None:
        """Print ns/char breakdown for manual inspection (never fails)."""
        results = []
        for label, payload in [
            ("string 500k", '{"content": "' + "a" * 500_000 + '"}'),
            ("raw 500k", "{" + ",".join(f'"f{i}":{i}' for i in range(25_000)) + "}"),
            ("100 fields", json.dumps({f"k{i}": "x" * 100 for i in range(100)})),
        ]:
            start = time.perf_counter()
            ToolArgStream("c").feed(payload)
            elapsed = time.perf_counter() - start
            ns = elapsed / len(payload) * 1e9
            mb_s = len(payload) / elapsed / 1e6
            results.append(f"{label}: {ns:.1f} ns/char  ({mb_s:.1f} MB/s)")

        print("\n  " + "\n  ".join(results))
