"""The format, case by case, and the state machine that reads it."""

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


async def test_a_stream_that_stops_without_its_last_blank_line_still_says_what_it_had(read: Read) -> None:
    assert await read(b"data: cut short\n") == [Event(data="cut short")]
    assert await read(b"data: cut short") == [Event(data="cut short")]


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


async def test_a_stream_that_ends_on_a_bare_cr_dispatches_on_it(read: Read) -> None:
    # The held \r is a terminator, not data. Appended back into the value it made data == "x\r".
    assert await read(b"data: x\r") == [Event(data="x")]


def test_the_decoder_is_the_format_and_needs_no_loop() -> None:
    decoder = Decoder()
    assert decoder.decode(b"data: hel") == []
    assert decoder.decode(b"lo\n\ndata: wor") == [Event(data="hello")]
    assert decoder.decode(b"ld", final=True) == [Event(data="world")]


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


def test_a_retry_on_its_own_still_arrives_because_it_carries_no_data_by_definition() -> None:
    assert Decoder().decode(b"retry: 500\n\n", final=True) == [Event(retry=500)]
