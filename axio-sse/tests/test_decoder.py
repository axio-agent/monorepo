"""The format, case by case, and the state machine that reads it."""

import time
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from axio_sse import Decoder, Event

#: The `read` fixture, typed where it is used. Importing this from conftest only resolves
#: when this package is pytest's rootdir, which breaks collection from the repository root.
type Read = Callable[..., Coroutine[Any, Any, list[Event]]]


async def test_the_ordinary_case(read: Read) -> None:
    assert await read(b'data: {"a":1}\n\n') == [Event(data='{"a":1}')]


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r"])
async def test_all_three_terminators(read: Read, ending: str) -> None:
    assert await read(f"data: hello{ending}{ending}") == [Event(data="hello")]


async def test_a_terminator_split_across_chunks(read: Read) -> None:
    assert await read("data: hello\r", "\ndata: world\r\n\r\n") == [Event(data="hello\nworld")]


async def test_data_over_several_lines_is_one_event(read: Read) -> None:
    assert await read(b'data: {"a":\ndata: 1}\n\n') == [Event(data='{"a":\n1}')]


async def test_comments_are_not_events(read: Read) -> None:
    assert await read(b": ping\n\n: ping\n\ndata: real\n\n") == [Event(data="real")]


async def test_the_other_fields_are_carried_and_not_confused_with_data(read: Read) -> None:
    assert await read(b"event: delta\nid: 7\nretry: 500\ndata: x\n\n") == [
        Event(data="x", event="delta", id="7", retry=500)
    ]


async def test_a_field_nobody_defined_is_ignored(read: Read) -> None:
    assert await read(b"weird: thing\ndata: x\n\n") == [Event(data="x")]


async def test_exactly_one_leading_space_comes_off_the_value(read: Read) -> None:
    assert await read(b"data:  two spaces\n\n") == [Event(data=" two spaces")]


async def test_an_id_with_a_null_is_refused(read: Read) -> None:
    assert await read("id: a\0b\ndata: x\n\n") == [Event(data="x", id="")]


async def test_a_stream_that_stops_before_its_blank_line_says_nothing(read: Read) -> None:
    # The format discards what is pending at end of file. Dispatched anyway, a connection cut
    # between a frame and the blank line after it made a truncated turn read as a finished one.
    assert await read(b"data: cut short\n") == []
    assert await read(b"data: cut short") == []


async def test_nothing_at_all_yields_nothing(read: Read) -> None:
    assert await read(b"") == []
    assert await read(b"\n\n\n") == []


@pytest.mark.parametrize("size", [1, 2, 3, 7, 64])
async def test_the_result_does_not_depend_on_where_the_chunks_fall(read: Read, size: int) -> None:
    stream = (
        b": keep-alive\n\n"
        b'data: {"first":\ndata: true}\n\n'
        b"event: named\r\ndata: second\r\n\r\n"
        b"data: \xd0\xb1\xd0\xb0\xd0\xbb\xd0\xba\xd0\xbe\xd0\xbd\n\n"
    )
    assert await read(stream, size=size) == [
        Event(data='{"first":\ntrue}'),
        Event(data="second", event="named"),
        Event(data="балкон"),
    ]


async def test_an_event_far_larger_than_any_line_limit(read: Read) -> None:
    huge = "x" * 300_000
    assert await read(f"data: {huge}\n\n".encode(), size=8192) == [Event(data=huge)]


async def test_a_bare_cr_ends_the_line_and_not_the_event(read: Read) -> None:
    # The held \r is a terminator, not data: appended back into the value it made data == "x\r".
    # The line is complete, the event is not, so end of file discards it.
    assert await read(b"data: x\r") == []
    assert await read(b"data: x\r\r") == [Event(data="x")]


def test_the_decoder_is_the_format_and_needs_no_loop() -> None:
    decoder = Decoder()
    assert decoder.decode(b"data: hel") == []
    assert decoder.decode(b"lo\n\ndata: wor") == [Event(data="hello")]
    assert decoder.decode(b"ld\n\n", final=True) == [Event(data="world")]


def test_a_decoder_forgets_a_half_read_event_when_it_is_reset() -> None:
    decoder = Decoder()
    decoder.decode(b"data: half")
    decoder.reset()
    assert decoder.decode(b"", final=True) == []


# ---------- junk that used to end the turn ----------


def test_a_leading_byte_order_mark_does_not_eat_the_first_event() -> None:
    # Left in, the mark makes the first field name `﻿data`, which is unknown, so the event
    # vanishes with nothing collected. `strict` cannot see it: it is absent, not unknown.
    stream = b'\xef\xbb\xbfdata: {"a":1}\n\ndata: {"b":2}\n\n'
    assert [e.data for e in Decoder().decode(stream, final=True)] == ['{"a":1}', '{"b":2}']


def test_a_byte_order_mark_split_across_chunks_is_still_stripped() -> None:
    decoder = Decoder()
    got = decoder.decode(b"\xef\xbb") + decoder.decode(b"\xbfdata: x\n\n", final=True)
    assert [e.data for e in got] == ["x"]


def test_a_mark_given_as_text_is_stripped_too() -> None:
    # decode() takes bytes or text, and only the byte decoder strips the mark for us.
    assert [e.data for e in Decoder().decode("﻿data: x\n\n", final=True)] == ["x"]


def test_a_mark_inside_a_value_is_left_alone() -> None:
    assert Decoder().decode("data: a﻿b\n\n", final=True) == [Event(data="a﻿b")]


@pytest.mark.parametrize("value", ["²", "٩" * 30, "9" * 5000, "-1", "1.5", ""])
def test_a_retry_value_int_cannot_read_is_ignored_rather_than_fatal(value: str) -> None:
    # str.isdigit() is true for characters int() refuses, and CPython refuses past 4300 digits, so
    # the guard meant to keep junk out raised out of decode() and took the rest of the turn with it.
    got = Decoder().decode(f"retry: {value}\ndata: x\n\n", final=True)
    assert got == [Event(data="x")]


def test_a_retry_value_int_can_read_still_arrives() -> None:
    assert Decoder().decode("retry: 500\ndata: x\n\n", final=True) == [Event(data="x", retry=500)]


def test_an_event_with_a_name_and_no_data_fires_nothing() -> None:
    # The format dispatches on the data buffer, never on the name.
    assert Decoder().decode(b"event: ping\n\n", final=True) == []
    assert Decoder().decode(b"event::\n\n", final=True) == []


def test_a_name_that_fired_nothing_does_not_leak_onto_the_next_event() -> None:
    assert Decoder().decode(b"event: ping\n\ndata: x\n\n", final=True) == [Event(data="x")]


def test_a_retry_on_its_own_sends_nothing() -> None:
    # It sets the stream's reconnection time. The format dispatches on the data buffer alone, so a
    # caller of the public events() was handed an empty event for a directive.
    assert Decoder().decode(b"retry: 500\n\n", final=True) == []


def test_a_retry_beside_data_still_reports_the_value() -> None:
    assert Decoder().decode(b"retry: 500\ndata: x\n\n", final=True) == [Event(data="x", retry=500)]


@pytest.mark.parametrize("value", ["١٠٠", "٩", "１２３"])
def test_a_retry_in_digits_that_are_not_ascii_is_ignored(value: str) -> None:
    # str.isdecimal() is true for these and int() reads them, but the format allows ASCII 0-9 only.
    assert Decoder().decode(f"retry: {value}\ndata: x\n\n", final=True) == [Event(data="x")]


def test_a_switch_from_bytes_to_text_does_not_swallow_a_half_read_character() -> None:
    # The byte decoder was holding the first byte of a two-byte character. A str chunk can never
    # complete it, and dropped it took the character with it and said nothing.
    decoder = Decoder()
    decoder.decode(b"data: ")
    decoder.decode(b"\xc3")  # the first byte of a two-byte character

    made = decoder.decode("x\n\n")

    assert [event.data for event in made] == ["\ufffdx"]


def test_text_after_a_clean_byte_boundary_adds_nothing() -> None:
    decoder = Decoder()
    decoder.decode(b"data: hel")

    made = decoder.decode("lo\n\n")

    assert [event.data for event in made] == ["hello"]


def test_a_partial_bom_does_not_outlive_the_line_it_broke() -> None:
    # A final flush returns nothing for a partial BOM and leaves the byte in the decoder. Held, it
    # prefixed the next byte chunk long after the corruption, breaking a line that was clean.
    decoder = Decoder()
    decoder.decode(b"\xef")
    decoder.decode("")

    made = decoder.decode(b"data: broken\n\ndata: clean\n\n")

    assert [event.data for event in made] == ["clean"]


# ---------- what holding a chunk costs ----------


def _cost(payload: int, size: int = 8192) -> float:
    """The CPU cost of the best of three runs over one ``data:`` line, cut into chunks.

    CPU time, not wall clock: a machine busy with other work inflates the clock and fails the test
    for a reason that has nothing to do with the decoder.
    """
    stream = ("data: " + "x" * payload + "\n\n").encode()
    parts = [stream[at : at + size] for at in range(0, len(stream), size)]
    best = float("inf")
    for _ in range(3):
        decoder = Decoder()
        started = time.process_time()
        for part in parts:
            decoder.decode(part)
        decoder.decode(b"", final=True)
        best = min(best, time.process_time() - started)
    return best


def test_a_big_event_costs_what_its_size_costs_and_not_what_its_square_costs() -> None:
    # Held text was scanned once per line and copied once per chunk, so four times the payload cost
    # sixteen times the time. Sixteen and four are the two answers, and the gate sits between them.
    _cost(1 << 18)  # warm the paths, so the first measured size is not the one that pays for them
    small, large = _cost(1 << 20), _cost(1 << 22)
    assert large / small < 8, f"four times the payload cost {large / small:.1f} times the time"


def test_a_read_buffer_holding_hundreds_of_small_events_gives_back_every_one_of_them() -> None:
    # 64 KB is aiohttp's default read buffer, so hundreds of ordinary deltas arrive in one chunk.
    want = [Event(data=f'{{"i":{n}}}', event="delta") for n in range(2_000)]
    stream = "".join(f'event: delta\ndata: {{"i":{n}}}\n\n' for n in range(2_000)).encode()
    decoder = Decoder()
    got = [event for at in range(0, len(stream), 65536) for event in decoder.decode(stream[at : at + 65536])]
    assert got + decoder.decode(b"", final=True) == want


def test_a_reset_leaves_no_buffer_and_no_offset_into_it() -> None:
    # The buffer keeps read lines and reads past them by offset. Left set against an empty buffer,
    # the scan offset goes negative and str.find then starts near the end of the next stream.
    decoder = Decoder()
    decoder.decode(b"data: first\n\ndata: half")

    decoder.reset()

    assert (decoder._held, decoder._start, decoder._scan, decoder._parts) == ("", 0, 0, [])
    assert decoder.decode(b"data: second\n\n", final=True) == [Event(data="second")]


def _ordinary(size: int, events: int = 8_000) -> float:
    """The CPU cost of the best of three runs over many small events, cut into chunks of ``size``.

    The stream is far larger than the largest chunk size measured, or the biggest chunk holds the
    whole stream and the per-line rescan this guards against never happens.
    """
    stream = (('data: {"i":"' + "x" * 40 + '"}\n\n') * events).encode()
    parts = [stream[at : at + size] for at in range(0, len(stream), size)]
    best = float("inf")
    for _ in range(3):
        decoder = Decoder()
        started = time.process_time()
        for part in parts:
            decoder.decode(part)
        decoder.decode(b"", final=True)
        best = min(best, time.process_time() - started)
    return best


def test_many_small_events_cost_the_same_however_they_are_chunked() -> None:
    # Each line re-scanned the whole remaining buffer, so a bigger read buffer cost more for the
    # same stream: 64 KB chunks were 26 times 1 KB chunks. 64 KB is aiohttp's default.
    _ordinary(8192)  # warm the paths, so the first measured size does not pay for them
    small, large = _ordinary(1024), _ordinary(65536)
    assert large / small < 2, f"a 64x larger read buffer cost {large / small:.1f} times the time"


def test_a_finished_event_is_not_held_a_second_time() -> None:
    # The buffer dropped read lines only when the next terminator arrived, so between two large
    # events it held both: the one already handed back, and the one still arriving.
    body = ("data: " + "x" * (1 << 20) + "\n\n").encode()
    decoder = Decoder()
    made = [event for at in range(0, len(body), 65536) for event in decoder.decode(body[at : at + 65536])]

    assert len(made) == 1
    assert len(decoder._held) < len(made[0].data) // 100


def test_a_turn_cut_before_its_blank_line_is_not_a_finished_turn() -> None:
    # The frame that says the response completed arrived whole, but the event did not. Dispatched,
    # the reader above read a cut connection as a finished answer, which is what the transports'
    # cut-stream check exists to catch.
    cut = b'data: {"type":"response.completed","response":{"status":"completed"}}\n'

    assert Decoder().decode(cut, final=True) == []


def test_the_same_frame_with_its_blank_line_does_arrive() -> None:
    whole = b'data: {"type":"response.completed"}\n\n'

    assert [event.data for event in Decoder().decode(whole, final=True)] == ['{"type":"response.completed"}']


def test_a_mark_in_a_byte_chunk_after_a_text_chunk_is_data() -> None:
    # Only a mark at the very start of the stream is stripped. The byte decoder kept expecting one
    # of its own, so it ate the mark from the first byte chunk that followed a text chunk.
    decoder = Decoder()
    decoder.decode("data: first\n\n")

    made = decoder.decode(b"\xef\xbb\xbfdata: second\n\n")

    assert made == [], "the field name is ﻿data, which no format defines"


@pytest.mark.parametrize("as_bytes", [True, False])
def test_only_the_first_mark_is_stripped(as_bytes: bool) -> None:
    # The second is data, so the field name is ﻿data and the event is dropped. Stripped as well,
    # the same stream read one way as bytes and another as text.
    stream = "﻿﻿data: x\n\n"

    made = Decoder().decode(stream.encode() if as_bytes else stream, final=True)

    assert made == []
