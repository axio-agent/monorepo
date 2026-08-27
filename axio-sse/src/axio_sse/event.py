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


@dataclass(frozen=True, slots=True)
class Event:
    """One dispatched event, with the four fields the format defines."""

    data: str = ""
    #: Empty means unnamed, which the format reads as "message".
    event: str = ""
    #: The stream position for a client that reconnects, not an id of this event.
    id: str = ""
    retry: int | None = None

    def payload(self) -> Payload | None:
        """This event's JSON object, or None when it carries none.

        Junk is skipped rather than raised: one unreadable event must not end a turn that was
        working. It is a warning only where the data opens as an object. Anything else is a
        sentinel the caller did not name in ``until``, and one warning per turn is noise.
        """
        if not self.data:
            return None
        try:
            got = json.loads(self.data)
        except json.JSONDecodeError:
            log.log(
                logging.WARNING if self.data.startswith("{") else logging.DEBUG,
                "payload is not JSON, skipping: %.80s",
                self.data,
            )
            return None
        if not isinstance(got, dict):
            log.warning("payload is not an object, skipping: %.80s", self.data)
            return None
        return Payload(got)
