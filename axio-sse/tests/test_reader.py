"""One endpoint's vocabulary, as one method per event."""

import logging
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import FrozenInstanceError, dataclass

import pytest

from axio_sse import EVENT_NAME, Event, Payload, Reader, UnknownEvent, on

#: The `stream` fixture, typed where it is used. Importing this from conftest only resolves
#: when this package is pytest's rootdir, which breaks collection from the repository root.
type Stream = Callable[..., AsyncIterator[bytes | str]]


class Watch(Reader[tuple[str, str]]):
    """The vocabulary of a made-up endpoint, as this test's reader reads it."""

    def __init__(self) -> None:
        self.seen = 0

    @on("delta")
    def _delta(self, payload: Payload) -> Iterator[tuple[str, str]]:
        self.seen += 1
        yield ("delta", payload.string("text"))

    @on("done", "finished")
    def _done(self, payload: Payload) -> Iterator[tuple[str, str]]:
        yield ("done", payload.string("why"))

    @on("ping")
    def _expected(self, payload: Payload) -> None:
        """Sent every few seconds and carries nothing. Named so strict stays quiet."""

    @on("split")
    def _split(self, payload: Payload) -> list[tuple[str, str]]:
        return [("delta", part) for part in payload.string("text")]


async def test_each_event_reads_itself(stream: Stream) -> None:
    chunks = stream(
        b'data: {"type":"delta","text":"hel"}\n\n',
        b'data: {"type":"delta","text":"lo"}\n\n',
        b'data: {"type":"done","why":"stop"}\n\n',
    )
    assert [made async for made in Watch().over(chunks)] == [("delta", "hel"), ("delta", "lo"), ("done", "stop")]


async def test_one_method_can_answer_for_several_names() -> None:
    assert Watch().read(Event(data='{"type":"finished"}')) == [("done", "")]


async def test_an_event_can_become_nothing_one_thing_or_many() -> None:
    reader = Watch()
    assert reader.read(Event(data='{"type":"ping"}')) == []
    assert reader.read(Event(data='{"type":"delta","text":"x"}')) == [("delta", "x")]
    assert reader.read(Event(data='{"type":"split","text":"ab"}')) == [("delta", "a"), ("delta", "b")]


async def test_an_unknown_event_is_skipped_when_that_is_the_policy(
    stream: Stream, caplog: pytest.LogCaptureFixture
) -> None:
    chunks = stream(b'data: {"type":"something-new"}\n\ndata: {"type":"delta","text":"x"}\n\n')
    with caplog.at_level(logging.DEBUG, logger="axio.sse"):
        got = [made async for made in Watch().over(chunks)]
    assert got == [("delta", "x")], "an event nobody reads stopped the ones that were read"
    assert "something-new" in caplog.text


async def test_an_unknown_event_raises_when_that_is_the_policy() -> None:
    with pytest.raises(UnknownEvent, match="something-new"):
        Watch().read(Event(data='{"type":"something-new"}'), strict=True)


async def test_an_event_the_reader_refused_on_purpose_is_silent() -> None:
    # The difference between "expected and carries nothing" and "never heard of it".
    Watch().read(Event(data='{"type":"ping"}'), strict=True)


async def test_what_it_reads_can_be_asked() -> None:
    # Why this is a reader rather than a chain of ifs: a chain cannot be asked what it handles, so
    # nothing can hold it against the provider's own list.
    assert Watch.names() == {"delta", "done", "finished", "ping", "split"}


async def test_two_methods_for_one_name_is_refused() -> None:
    with pytest.raises(ValueError, match="reads 'delta' twice"):

        class Twice(Reader[str]):
            @on("delta")
            def _one(self, payload: Payload) -> None: ...

            @on("delta")
            def _two(self, payload: Payload) -> None: ...


async def test_a_name_must_be_a_name() -> None:
    with pytest.raises(ValueError, match="at least one event name"):
        on()
    with pytest.raises(ValueError, match="no name may be empty"):
        on("")


async def test_the_events_own_name_can_be_the_discriminator(stream: Stream) -> None:
    class Named(Reader[str], by=EVENT_NAME):
        @on("ping")
        def _ping(self, payload: Payload) -> Iterator[str]:
            yield payload.string("at")

    assert [made async for made in Named().over(stream(b'event: ping\ndata: {"at":"noon"}\n\n'))] == ["noon"]


async def test_a_subclass_keeps_what_its_parent_reads_and_replaces_only_what_it_claims_again() -> None:
    class Louder(Watch):
        @on("delta")
        def _delta(self, payload: Payload) -> Iterator[tuple[str, str]]:
            yield ("delta", payload.string("text").upper())

    assert Louder.names() == Watch.names()
    assert Louder().read(Event(data='{"type":"delta","text":"x"}')) == [("delta", "X")]


async def test_a_method_overridden_without_the_decorator_is_still_the_one_that_runs() -> None:
    # The table holds the attribute name. Keyed by the function, the parent's method would run and
    # the override would be dead code that nothing reports.
    class Quiet(Watch):
        def _delta(self, payload: Payload) -> Iterator[tuple[str, str]]:
            yield ("delta", "")

    assert Quiet().read(Event(data='{"type":"delta","text":"x"}')) == [("delta", "")]


async def test_the_vocabulary_is_complete_at_the_class_statement() -> None:
    # No method adds a handler afterwards, which is what makes names() answerable without reading
    # every module that imported this one.
    assert [name for name in ("add", "register", "reads", "ignores") if hasattr(Watch, name)] == []


async def test_one_reader_holds_one_turn() -> None:
    reader = Watch()
    reader.read(Event(data='{"type":"delta","text":"x"}'))
    assert reader.seen == 1
    assert Watch().seen == 0, "a second turn started with the first turn's state"


async def test_a_second_discriminator_obeys_the_same_policy() -> None:
    class Blocks(Reader[str]):
        @on("block")
        def _block(self, payload: Payload) -> Iterator[str]:
            match payload.string("kind"):
                case "text":
                    yield payload.string("text")
                case other:
                    self.unknown(other)

    assert Blocks().read(Event(data='{"type":"block","kind":"text","text":"x"}')) == ["x"]
    assert Blocks().read(Event(data='{"type":"block","kind":"new"}')) == []
    with pytest.raises(UnknownEvent, match="'new'"):
        Blocks().read(Event(data='{"type":"block","kind":"new"}'), strict=True)


async def test_strict_belongs_to_the_call_and_not_to_the_reader() -> None:
    # A policy that outlived one call would leave a CI test's strictness set for the next caller.
    reader = Watch()
    with pytest.raises(UnknownEvent):
        reader.read(Event(data='{"type":"new"}'), strict=True)
    assert reader.read(Event(data='{"type":"new"}')) == []


async def test_a_reader_gives_back_what_its_handlers_make(stream: Stream) -> None:
    # Reader[T] with T a union is how one stream yields several types, each narrowable in a match.
    @dataclass(frozen=True, slots=True)
    class Delta:
        text: str

    @dataclass(frozen=True, slots=True)
    class Done:
        why: str

    class Mixed(Reader[Delta | Done]):
        @on("delta")
        def _delta(self, payload: Payload) -> Iterator[Delta]:
            yield Delta(text=payload.string("text"))

        @on("done")
        def _done(self, payload: Payload) -> Iterator[Done]:
            yield Done(why=payload.string("why"))

    chunks = stream(b'data: {"type":"delta","text":"x"}\n\ndata: {"type":"done","why":"stop"}\n\n')
    assert [made async for made in Mixed().over(chunks)] == [Delta(text="x"), Done(why="stop")]


def test_a_reader_may_have_slots_but_not_be_frozen() -> None:
    # A reader is the turn's state, so frozen is a contradiction. Pinned because the rebuilt class
    # reports the refusal as a super() type error, which names nothing that led to it.
    @dataclass(slots=True)
    class Slotted(Reader[str]):
        @on("delta")
        def _delta(self, payload: Payload) -> Iterator[str]:
            yield payload.string("text")

    assert Slotted().read(Event(data='{"type":"delta","text":"x"}')) == ["x"]

    @dataclass(frozen=True)
    class Frozen(Reader[str]):
        @on("delta")
        def _delta(self, payload: Payload) -> Iterator[str]:
            yield payload.string("text")

    with pytest.raises(FrozenInstanceError):
        Frozen().read(Event(data='{"type":"delta","text":"x"}'))


class TestUnmatched:
    """A stream whose vocabulary grows on its own must not need a list that grows with it."""

    class Forwarding(Reader[str]):
        @on("delta")
        def _delta(self, payload: Payload) -> Iterator[str]:
            yield payload.string("text")

        def unmatched(self, name: str, payload: Payload) -> Iterator[str]:
            yield f"forwarded:{name}"

    def test_a_payload_no_method_claims_can_be_forwarded_instead_of_dropped(self) -> None:
        reader = self.Forwarding()
        assert reader.read(Event(data='{"type":"delta","text":"x"}')) == ["x"]
        assert reader.read(Event(data='{"type":"something.brand.new"}')) == ["forwarded:something.brand.new"]

    def test_the_reader_still_names_only_what_it_interprets(self) -> None:
        assert self.Forwarding.names() == {"delta"}

    def test_strict_still_refuses_what_the_reader_does_not_interpret(self) -> None:
        # Forwarding is the running policy; a replay meant to catch new vocabulary still fails.
        with pytest.raises(UnknownEvent, match="something.brand.new"):
            self.Forwarding().read(Event(data='{"type":"something.brand.new"}'), strict=True)

    def test_dropping_is_still_the_default(self) -> None:
        class Quiet(Reader[str]):
            @on("delta")
            def _delta(self, payload: Payload) -> Iterator[str]:
                yield payload.string("text")

        assert Quiet().read(Event(data='{"type":"whatever"}')) == []


async def test_a_redecorated_method_drops_the_name_its_parent_gave_it() -> None:
    # The table is keyed by attribute name so an override without @on still runs. A subclass that
    # decorates the attribute again has redefined it, and keeping the parent's name dispatched the
    # parent's event to a method written for another one.
    class Base(Reader[str]):
        @on("alpha")
        def _handler(self, payload: Payload) -> Iterator[str]:
            yield "base"

    class Child(Base):
        @on("beta")
        def _handler(self, payload: Payload) -> Iterator[str]:
            yield "child"

    assert Base.names() == {"alpha"}
    assert Child.names() == {"beta"}
    assert Child().read(Event(data='{"type":"alpha"}')) == []
    assert Child().read(Event(data='{"type":"beta"}')) == ["child"]


async def test_an_override_without_the_decorator_still_keeps_the_name() -> None:
    class Base(Reader[str]):
        @on("alpha")
        def _handler(self, payload: Payload) -> Iterator[str]:
            yield "base"

    class Quiet(Base):
        def _handler(self, payload: Payload) -> Iterator[str]:
            yield "child"

    assert Quiet.names() == {"alpha"}
    assert Quiet().read(Event(data='{"type":"alpha"}')) == ["child"]
