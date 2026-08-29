"""One event and the JSON object inside it."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("axio.sse")


class Payload(dict[str, Any]):
    """The JSON object inside one event, read by path.

    ``payload.number("message", "usage", "input_tokens")`` walks the path and gives the default
    wherever a step is missing, null, or the wrong type — which is what an optional provider field
    is. It is a ``dict``, so ``payload["x"]``, ``in``, and ``json.dumps`` all still work. The four
    readers exist so a handler carries no ``Any`` and no chain of ``.get({})``.
    """

    __slots__ = ()

    def _at(self, keys: tuple[str, ...]) -> Any:
        found: Any = self
        for key in keys:
            if not isinstance(found, dict):
                return None
            found = found.get(key)
        return found

    def string(self, *keys: str, default: str = "") -> str:
        """The string at this path, or the default where the provider sent none."""
        found = self._at(keys)
        return found if isinstance(found, str) else default

    def number(self, *keys: str, default: int = 0) -> int:
        """The whole number at this path, or the default where the provider sent none."""
        found = self._at(keys)
        # bool is an int in Python. A true/false field must not read here as 1 or 0.
        return found if isinstance(found, int) and not isinstance(found, bool) else default

    def obj(self, *keys: str) -> Payload:
        """The object at this path, empty where there is none, so a path can be walked in steps."""
        found = self._at(keys)
        return Payload(found) if isinstance(found, dict) else Payload()

    def objs(self, *keys: str) -> list[Payload]:
        """Every object in the list at this path. A missing list reads as no objects."""
        found = self._at(keys)
        if not isinstance(found, list):
            return []
        return [Payload(one) for one in found if isinstance(one, dict)]


class MalformedPayload(ValueError):
    """An event that carried data no reader can act on.

    Raised rather than skipped: the stream said this event mattered, and there is no way to
    continue reading it that does not report a partial turn as a whole one.
    """


@dataclass(frozen=True, slots=True)
class Event:
    """One dispatched event, with the four fields the format defines."""

    data: str = ""
    #: What the ``event:`` field carried, empty where the stream sent none.
    event: str = ""
    #: The stream position for a client that reconnects, not an id of this event.
    id: str = ""
    retry: int | None = None

    @property
    def name(self) -> str:
        """The event's type. An unnamed event is of type ``message``, which the format defines.

        Dispatched on the raw field instead, an ``@on("message")`` handler never runs for the
        ordinary unnamed event, and a strict read rejects it as unknown.
        """
        return self.event or "message"

    def payload(self) -> Payload | None:
        """This event's JSON object, or None where the event carries no data at all.

        Data that will not parse raises. Skipped instead, a text or tool-call event whose JSON
        arrived corrupt was dropped, the completion event after it still reported success, and the
        caller got a short answer or half a tool's arguments with nothing saying anything was lost.

        A sentinel such as ``[DONE]`` is data too, and reaches here as junk. Name it in ``until``,
        which ends the stream before it is read.
        """
        if not self.data:
            return None
        try:
            got = json.loads(self.data)
        except json.JSONDecodeError as exc:
            log.error("payload is not JSON: %.80s", self.data)
            raise MalformedPayload(f"event {self.name!r} carries data that is not JSON: {exc}") from exc
        if not isinstance(got, dict):
            log.error("payload is not an object: %.80s", self.data)
            raise MalformedPayload(f"event {self.name!r} carries {type(got).__name__} and not an object")
        return Payload(got)
