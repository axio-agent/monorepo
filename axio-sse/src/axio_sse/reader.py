"""What one endpoint sends, as one method per event."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast

from .event import Event, Payload
from .stream import events
from .wire import Wire

log = logging.getLogger("axio.sse")

#: ``by=EVENT_NAME`` dispatches on the format's own ``event:`` field. Any other ``by`` names
#: a key in the payload, and no provider puts a colon in a key.
EVENT_NAME: Final = "event:"


class UnknownEvent(LookupError):
    """A name no method of this reader claims, met while the reader reads strictly."""


#: What a handler returns: what it made, or nothing. One rule for none, one, and many.
type Handled[T] = Iterable[T] | None

type _Handler[R, P, T] = Callable[[R, P], Handled[T]]


def on[R, P, T](*claimed: str | type[Wire]) -> Callable[[_Handler[R, P, T]], _Handler[R, P, T]]:
    """Give a ``Reader`` method the payloads it reads.

    Give it a ``Wire`` shape and the method is handed that shape, its fields read by declared name
    and type. Give it wire names and the method is handed the ``Payload`` itself, which is what a
    method that only forwards an event wants. Declaring a shape for a payload nobody reads a field
    of would be a schema written for nothing.

    Several names on one method is how a stream that sends one thing under two names is written.
    Both stay in the class body, so ``strict`` has nothing to fire on and no second list exists to
    keep in step with the first.
    """
    shapes = [one for one in claimed if isinstance(one, type)]
    if len(shapes) > 1:
        raise ValueError("on() takes one shape, or names; a method reads one shape at a time")
    if shapes and len(claimed) > 1:
        raise ValueError(f"on({shapes[0].__name__}) already carries its names; do not repeat them")

    if shapes:
        shape = shapes[0]
        if not issubclass(shape, Wire):
            raise TypeError(f"{shape.__name__} is not a Wire, so it cannot say what it reads")
        if not shape.names:
            raise ValueError(f"{shape.__name__} has no name to read under — give it name= on the class line")
        names: tuple[str, ...] = shape.names
    else:
        names = tuple(one for one in claimed if isinstance(one, str))
        shape = None
        if not names or not all(names):
            raise ValueError("on() takes at least one event name, and no name may be empty")

    def tag(method: _Handler[R, P, T]) -> _Handler[R, P, T]:
        # setattr, not an assignment: a type checker refuses a new attribute on a Callable.
        setattr(method, "_sse_names", names)
        setattr(method, "_sse_shape", shape)
        return method

    return tag


def _redecorated(klass: type, attribute: str) -> bool:
    """Whether this class gives that attribute names of its own."""
    return bool(getattr(vars(klass).get(attribute), "_sse_names", ()))


class Reader[T]:
    """What one endpoint sends, as one method per event.

        class Messages(Reader[StreamEvent], by=EVENT_NAME):
            @on("content_block_delta")
            def _delta(self, payload: Payload) -> Iterator[StreamEvent]: ...

    One instance reads one stream. The turn's running totals and id maps live on ``self`` instead
    of travelling through a call. A reader used for a second response would carry the first one's
    state into it. Construct one per response. Being that state, a reader must not be frozen.
    ``read`` latches the caller's ``strict`` on ``self``, and ``@dataclass(frozen=True)`` refuses
    that assignment. With ``slots=True`` as well, the refusal comes from inside the rebuilt class
    and says only that ``super()`` got the wrong type. ``@dataclass(slots=True)`` alone is fine.

    A handler returns what the event became — an iterable, or None where the event only moved that
    state. ``by`` names the payload key that holds the event's name, or ``EVENT_NAME`` for the
    format's own ``event:`` field. A subclass that does not give ``by`` inherits it.
    """

    _by: ClassVar[str] = "type"
    #: Wire name to the method name that reads it and the shape it reads it as. Keyed by the
    #: function, a subclass that overrides a handler without repeating ``@on`` never runs.
    _handlers: ClassVar[Mapping[str, tuple[str, type[Wire] | None]]] = MappingProxyType({})
    #: What the running read was asked for, so ``unknown()`` obeys it from inside a handler.
    _strict: bool = False

    def __init_subclass__(cls, *, by: str | None = None, **rest: object) -> None:
        super().__init_subclass__(**rest)
        if by is not None:
            cls._by = by
        found: dict[str, tuple[str, type[Wire] | None]] = {}
        # The whole MRO, base first: a subclass keeps what its parent reads and replaces only the
        # names it claims again.
        for klass in reversed(cls.__mro__):
            here: dict[str, tuple[str, type[Wire] | None]] = {}
            # A method this class decorates again is this class's method, whatever its parent said it
            # read. Kept, the parent's name would still point at the attribute.
            for stale in [n for n, (attribute, _) in found.items() if _redecorated(klass, attribute)]:
                del found[stale]
            for attribute, method in vars(klass).items():
                shape = cast("type[Wire] | None", getattr(method, "_sse_shape", None))
                for name in cast(tuple[str, ...], getattr(method, "_sse_names", ())):
                    if name in here:
                        # Refused rather than settled by definition order, which silently leaves one of
                        # the two methods never called.
                        taken = here[name][0]
                        raise ValueError(f"{klass.__qualname__} reads {name!r} twice: {taken} and {attribute}")
                    here[name] = (attribute, shape)
            found.update(here)
        cls._handlers = MappingProxyType(found)

    @classmethod
    def names(cls) -> frozenset[str]:
        """Every name this reader claims, for a test to hold against the provider's own list."""
        return frozenset(cls._handlers)

    def unknown(self, name: str) -> None:
        """The one policy for a name nothing here reads: DEBUG, or refuse under ``strict``.

        A handler calls it for a second discriminator inside one event, such as a block that names
        the kind of its own chunks. A nested name nobody read then fails the same replay a new
        event fails, instead of disappearing.
        """
        log.debug("%s does not read %r", type(self).__name__, name)
        if self._strict:
            raise UnknownEvent(
                f"{type(self).__name__} does not read {name!r}; it reads {', '.join(sorted(self.names()))}"
            )

    def unmatched(self, name: str, payload: Payload) -> Handled[T]:
        """What a payload no method claims becomes. Nothing, unless a reader says otherwise.

        Override it to forward instead of drop. That is what a stream whose vocabulary grows on its
        own needs. An endpoint that runs tools names an event per tool. That set is a function
        of which tools exist and which were asked for, not of the protocol. Naming them one by one
        makes the reader stale the day a tool is added. It also reports a new tool as news about
        the protocol, when it is news about the tools.

        Name here only what this reader interprets. ``strict`` still refuses anything unnamed. A test can
        therefore hold the interpreted set against the schema, and the reader carries no list it
        cannot keep true.
        """
        return None

    def read(self, event: Event, *, strict: bool = False) -> list[T]:
        """Everything this one event became, empty where it became nothing."""
        payload = event.payload()
        if payload is None:
            return []
        # ``strict`` stays latched for the length of this read, so a nested unknown obeys the same
        # policy as a top-level one.
        self._strict = strict
        name = event.event if self._by == EVENT_NAME else payload.string(self._by)
        claimed = self._handlers.get(name)
        if claimed is None:
            # unknown() first: under strict it raises, so a reader that forwards still fails a replay.
            self.unknown(name)
            made = self.unmatched(name, payload)
            return [] if made is None else list(made)
        attribute, shape = claimed
        handler = cast(Callable[[Any], Handled[T]], getattr(self, attribute))
        made = handler(payload if shape is None else shape.read(payload))
        return [] if made is None else list(made)

    async def over(
        self, chunks: AsyncIterable[bytes | str], *, strict: bool = False, until: str = ""
    ) -> AsyncIterator[T]:
        """Read a whole stream of chunks, yielding what each event became.

        An event that becomes nothing yields nothing, so the outputs do not line up with the
        events.
        """
        async for event in events(chunks, until=until):
            for made in self.read(event, strict=strict):
                yield made
