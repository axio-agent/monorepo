"""Tests for axio.retry: which HTTP attempts repeat, and how long they wait first."""

from __future__ import annotations

from typing import Any

import pytest

from axio.retry import is_retryable, retry_delay


class _Response:
    """Only what the retry rules read of a response."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class TestWhichStatusesRepeat:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
    def test_rate_limits_and_server_faults_repeat(self, status: int) -> None:
        # 502 and 504 are what a gateway returns while the model is slow. One transport left them
        # out and failed the turn where every other transport retried it.
        assert is_retryable(status)

    @pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 422])
    def test_nothing_the_caller_got_wrong_repeats(self, status: int) -> None:
        assert not is_retryable(status)


class TestHowLongToWait:
    def test_the_provider_header_wins(self) -> None:
        # It is the only figure that knows when the rate limit lifts.
        assert retry_delay(_Response({"Retry-After": "5"}), 1) == 5.0

    def test_the_wait_doubles_without_one(self) -> None:
        assert [retry_delay(None, n, base=2.0) for n in (1, 2, 3)] == [2.0, 4.0, 8.0]

    def test_a_header_that_is_not_a_number_falls_back(self) -> None:
        # Some gateways send an HTTP-date, which float() refuses.
        assert retry_delay(_Response({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}), 1, base=1.0) == 1.0

    def test_a_negative_header_never_asks_to_wait_backwards(self) -> None:
        assert retry_delay(_Response({"Retry-After": "-30"}), 1) == 0.0

    def test_a_response_without_headers_falls_back(self) -> None:
        assert retry_delay(object(), 2, base=1.0) == 2.0  # type: ignore[arg-type]

    def test_the_caller_can_pass_anything_mapping_like(self) -> None:
        headers: Any = {"Retry-After": "0.5"}
        assert retry_delay(_Response(headers), 1) == 0.5
