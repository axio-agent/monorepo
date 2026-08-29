"""When an HTTP attempt is worth repeating, and how long to wait before repeating it.

One rule for every transport. Written out per transport they drifted apart: one retried only 429,
500 and 503, so the 502 and 504 a proxy returns in front of a slow streaming endpoint failed the
turn there while the same failure was retried everywhere else. The same transport ignored
``Retry-After``, which is how a rate-limited provider says when to come back.

This module imports no HTTP client. ``retry_delay`` takes anything with a ``headers`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import Protocol

__all__ = ["HasHeaders", "is_retryable", "retry_delay"]


class HasHeaders(Protocol):
    """An HTTP response, as far as the retry rules are concerned."""

    @property
    def headers(self) -> object: ...


def is_retryable(status: int) -> bool:
    """Whether a provider's HTTP status says the same request may yet succeed.

    ``429`` is rate limiting. Anything from ``500`` up is the server calling the failure its own,
    which includes the ``502`` and ``504`` a gateway returns while the model is slow to answer.
    """
    return status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR


#: The longest wait any provider gets to ask for. ``Retry-After`` is theirs to set, and a header
#: reading ``inf`` or naming a date in 2099 otherwise stops the turn for ever.
_LONGEST_WAIT = 300.0


def retry_delay(resp: HasHeaders | None, attempt: int, *, base: float = 2.0) -> float:
    """Seconds to wait before the attempt after ``attempt``, which counts from 1.

    The provider's ``Retry-After`` is preferred, because it is the only figure that knows when the
    rate limit lifts. RFC 9110 allows it to be a count of seconds or an HTTP-date, and both are
    read: a date left unread meant retrying sooner than the server asked. Without a usable header
    the wait doubles each attempt, starting at ``base``.
    """
    headers = getattr(resp, "headers", None)
    if isinstance(headers, Mapping):
        raw = str(headers.get("Retry-After", "")).strip()
        try:
            return _bounded(float(raw))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            wait = (when - datetime.now(UTC)).total_seconds()
            # A date in the past says nothing about when the limit lifts: the server's clock is
            # ahead of ours, or the date elapsed on the way here. Clamped to zero it removed the
            # backoff altogether, and the loop retried a rate limit as fast as it could send.
            if wait > 0:
                return _bounded(wait)
    return _bounded(base * (2 ** (attempt - 1)))


def _bounded(seconds: float) -> float:
    """A wait this process can actually sit through, and never a negative one."""
    if seconds != seconds:  # NaN, which float() reads from a header saying "nan"
        return 0.0
    return min(max(0.0, seconds), _LONGEST_WAIT)
