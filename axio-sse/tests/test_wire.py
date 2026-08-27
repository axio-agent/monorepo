"""Payload shapes: what a declared field reads, and what it refuses."""

from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from axio_sse import Event, Payload, Reader, Wire, on


@dataclass(frozen=True, slots=True)
class Usage(Wire):
    """Nested, and never dispatched to: it has no name of its own."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Item(Wire):
    type: str = ""
    id: str = ""


@dataclass(frozen=True, slots=True)
class Completed(Wire, name="response.completed", also="response.done"):
    usage: Usage = field(default_factory=Usage)
    output: list[Item] = field(default_factory=list)
    status: str = "completed"
    truncated: bool = False
    output_index: int = 0


def test_declared_fields_are_read_and_the_rest_ignored() -> None:
    made = Completed.read(
        Payload(
            {
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "output": [{"type": "function_call", "id": "a"}, {"type": "message", "id": "b"}],
                "status": "incomplete",
                "sequence_number": 12,
            }
        )
    )
    assert made.usage == Usage(input_tokens=7, output_tokens=3)
    assert made.output == [Item(type="function_call", id="a"), Item(type="message", id="b")]
    assert made.status == "incomplete"


def test_a_field_that_did_not_arrive_takes_its_default() -> None:
    assert Completed.read(Payload({})) == Completed()


def test_a_null_or_wrongly_typed_field_takes_its_default_rather_than_landing() -> None:
    # What an optional provider field is. One bad field must not lose the whole event.
    made = Completed.read(Payload({"status": None, "usage": "not an object", "output": {"not": "a list"}}))
    assert made.status == "completed" and made.usage == Usage() and made.output == []


def test_a_flag_is_not_a_count_and_a_count_is_not_a_flag() -> None:
    # bool is an int in Python, so each has to refuse the other.
    assert Completed.read(Payload({"output_index": True})).output_index == 0
    assert Completed.read(Payload({"truncated": 1})).truncated is False
    assert Completed.read(Payload({"truncated": True})).truncated is True


def test_a_shape_can_arrive_under_two_names() -> None:
    assert Completed.names == ("response.completed", "response.done")


def test_a_nested_shape_needs_no_name() -> None:
    assert Usage.names == ()


# ---------- what a reader does with them ----------


class Endpoint(Reader[str]):
    @on(Completed)
    def _completed(self, wire: Completed) -> Iterator[str]:
        # Fields, not string keys: mypy checks this line.
        yield f"{wire.status}:{wire.usage.input_tokens}"

    @on("response.anything.else")
    def _forwarded(self, payload: Payload) -> Iterator[str]:
        # Named rather than shaped, because nothing here reads a field of it.
        yield f"forwarded:{sorted(payload)}"


def test_a_shaped_handler_is_handed_the_shape() -> None:
    got = Endpoint().read(Event(data='{"type":"response.completed","usage":{"input_tokens":9}}'))
    assert got == ["completed:9"]


def test_a_named_handler_is_handed_the_payload() -> None:
    got = Endpoint().read(Event(data='{"type":"response.anything.else","a":1,"b":2}'))
    assert got == ["forwarded:['a', 'b', 'type']"]


def test_a_shape_claims_its_own_names() -> None:
    assert Endpoint.names() == {"response.completed", "response.done", "response.anything.else"}


def test_a_shape_with_no_name_cannot_be_dispatched_to() -> None:
    with pytest.raises(ValueError, match="no name to read under"):
        on(Usage)


def test_repeating_a_shapes_names_beside_it_is_refused() -> None:
    with pytest.raises(ValueError, match="already carries its names"):
        on(Completed, "response.completed")


def test_two_shapes_on_one_method_is_refused() -> None:
    with pytest.raises(ValueError, match="one shape at a time"):
        on(Completed, Item)


def test_something_that_is_not_a_shape_is_refused() -> None:
    class Impostor:
        names = ("x",)

    with pytest.raises(TypeError, match="not a Wire"):
        on(Impostor)  # type: ignore[arg-type]  # mypy refuses it too, which is the point


def test_a_shape_that_is_not_a_dataclass_says_so() -> None:
    class Bare(Wire, name="bare"):
        pass

    with pytest.raises(TypeError, match="not a dataclass"):
        Bare.read(Payload({}))


def test_a_raw_field_receives_the_whole_payload() -> None:
    # For a shape that varies too much to declare whole: the fields worth reading are declared and
    # the rest travels beside them.
    @dataclass(frozen=True, slots=True)
    class Annotation(Wire, name="annotation"):
        title: str = ""
        raw: Payload = field(default_factory=Payload)

    made = Annotation.read(Payload({"title": "The Report", "source": {"url": "https://example.invalid"}}))
    assert made.title == "The Report"
    assert made.raw["source"] == {"url": "https://example.invalid"}


# ---------- what a declared field refuses ----------


def test_a_number_too_large_for_a_float_takes_the_default_and_keeps_the_event() -> None:
    # json.loads makes an unbounded int and float() raises past ~1.8e308. One field the caller
    # cannot represent must not lose every other field beside it.
    @dataclass(frozen=True, slots=True)
    class Scored(Wire, name="scored"):
        text: str = ""
        confidence: float = 0.0

    made = Scored.read(Payload({"text": "kept", "confidence": int("9" * 401)}))
    assert made == Scored(text="kept", confidence=0.0)


def test_every_item_of_a_list_is_read_the_way_a_field_is() -> None:
    # Copied through unchecked, a declared list[Payload] handed the handler plain dicts and the
    # first .string() on one ended the stream.
    @dataclass(frozen=True, slots=True)
    class Listy(Wire, name="listy"):
        notes: list[Payload] = field(default_factory=list)
        tags: list[str] = field(default_factory=list)

    made = Listy.read(Payload({"notes": [{"title": "a"}, "junk", 7], "tags": [1, "ok", None]}))
    assert [type(one).__name__ for one in made.notes] == ["Payload"]
    assert made.notes[0].string("title") == "a"
    assert made.tags == ["ok"]


def test_reading_a_shape_does_not_rederive_its_annotations_every_time() -> None:
    # Annotations do not change, and re-deriving them measured as nine tenths of the cost of
    # reading an event.
    from axio_sse.wire import _hints

    @dataclass(frozen=True, slots=True)
    class Once(Wire, name="once"):
        text: str = ""

    _hints.cache_clear()
    for _ in range(5):
        Once.read(Payload({"text": "x"}))
    assert _hints.cache_info().misses == 1


def test_a_field_declared_as_a_real_union_reads_either_member() -> None:
    # Collapsed to the first member, a field declared `str | int` took its default whenever the
    # provider sent the other one, with no diagnostic.
    @dataclass(frozen=True, slots=True)
    class Either(Wire, name="either"):
        value: str | int = ""

    assert Either.read(Payload({"value": 7})).value == 7
    assert Either.read(Payload({"value": "a"})).value == "a"
    assert Either.read(Payload({"value": None})).value == ""
    assert Either.read(Payload({"value": {"a": 1}})).value == ""


def test_the_optional_case_still_takes_its_default() -> None:
    @dataclass(frozen=True, slots=True)
    class Maybe(Wire, name="maybe"):
        value: str | None = None

    assert Maybe.read(Payload({"value": "x"})).value == "x"
    assert Maybe.read(Payload({"value": 7})).value is None
