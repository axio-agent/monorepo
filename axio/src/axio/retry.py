"""When an HTTP attempt is worth repeating, and how long to wait before repeating it.

One rule for every transport. Written out per transport they drifted apart: one retried only 429,
500 and 503, so the 502 and 504 a proxy returns in front of a slow streaming endpoint failed the
turn there while the same failure was retried everywhere else. The same transport ignored
``Retry-After``, which is how a rate-limited provider says when to come back.

This module imports no HTTP client. ``retry_delay`` takes anything with a ``headers`` mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def retry_delay(resp: HasHeaders | None, attempt: int, *, base: float = 2.0) -> float:
    """Seconds to wait before the attempt after ``attempt``, which counts from 1.

    The provider's ``Retry-After`` is preferred, because it is the only figure that knows when the
    rate limit lifts. Without one the wait doubles each attempt, starting at ``base``.
    """
    headers = getattr(resp, "headers", None)
    if isinstance(headers, Mapping):
        try:
            return max(0.0, float(headers["Retry-After"]))
        except (KeyError, TypeError, ValueError):
            pass
    return float(base * (2 ** (attempt - 1)))
