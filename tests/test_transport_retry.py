"""Tests for OpenAITransport retry logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from axio.events import StreamEvent, TextDelta
from axio.exceptions import StreamError

from axio_transport_openai import OpenAITransport


def _make_sse_bytes(text: str = "hello") -> bytes:
    """Build minimal SSE byte payload that _parse_sse can handle."""
    import json

    lines = []
    chunk = {
        "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": None}],
    }
    lines.append(f"data: {json.dumps(chunk)}\n")
    done_chunk = {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    lines.append(f"data: {json.dumps(done_chunk)}\n")
    lines.append("data: [DONE]\n")
    return "\n".join(lines).encode()


def _mock_response(status: int, body: str = "error") -> AsyncMock:
    """Create a mock aiohttp response with given status."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    return resp


def _mock_sse_response(text: str = "hello") -> AsyncMock:
    """Create a mock aiohttp response that streams SSE data."""
    resp = AsyncMock()
    resp.status = 200
    data = _make_sse_bytes(text)

    content = AsyncMock()

    async def iter_any() -> AsyncIterator[bytes]:
        yield data

    content.iter_any = iter_any
    resp.content = content
    return resp


class _FakeContextManager:
    """Wraps a mock response to work as an async context manager."""

    def __init__(self, resp: AsyncMock) -> None:
        self.resp = resp

    async def __aenter__(self) -> AsyncMock:
        return self.resp

    async def __aexit__(self, *args: Any) -> None:
        pass


def _make_transport(**kwargs: Any) -> OpenAITransport:
    """Create an OpenAITransport with a mock session and fast retries."""
    kwargs.setdefault("retry_base_delay", 0.0)
    transport = OpenAITransport(
        api_key="test-key",
        **kwargs,
    )
    transport.session = MagicMock(spec=aiohttp.ClientSession)
    return transport


async def _collect_events(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


async def test_retry_on_5xx() -> None:
    transport = _make_transport()
    error_resp = _mock_response(500, "Internal Server Error")
    ok_resp = _mock_sse_response("recovered")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(error_resp), _FakeContextManager(ok_resp)],
    )

    events = await _collect_events(transport.stream([], [], ""))
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) == 1
    assert text_events[0].delta == "recovered"


async def test_retry_on_429() -> None:
    transport = _make_transport()
    rate_limit_resp = _mock_response(429, "Rate limited")
    ok_resp = _mock_sse_response("ok")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(rate_limit_resp), _FakeContextManager(ok_resp)],
    )

    events = await _collect_events(transport.stream([], [], ""))
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) == 1
    assert text_events[0].delta == "ok"


async def test_no_retry_on_4xx() -> None:
    transport = _make_transport()
    client_error_resp = _mock_response(400, "Bad Request")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(client_error_resp)],
    )

    with pytest.raises(StreamError, match="400"):
        await _collect_events(transport.stream([], [], ""))

    assert transport.session.post.call_count == 1  # type: ignore[union-attr]


async def test_no_retry_on_401() -> None:
    transport = _make_transport()
    auth_error_resp = _mock_response(401, "Unauthorized")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(auth_error_resp)],
    )

    with pytest.raises(StreamError, match="401"):
        await _collect_events(transport.stream([], [], ""))

    assert transport.session.post.call_count == 1  # type: ignore[union-attr]


async def test_max_retries_exhausted() -> None:
    transport = _make_transport(max_retries=3)
    error_resps = [_FakeContextManager(_mock_response(500, "down")) for _ in range(3)]
    transport.session.post = MagicMock(side_effect=error_resps)  # type: ignore[union-attr,method-assign]

    with pytest.raises(StreamError, match="500"):
        await _collect_events(transport.stream([], [], ""))

    assert transport.session.post.call_count == 3  # type: ignore[union-attr]


async def test_retry_on_connection_error() -> None:
    transport = _make_transport()
    ok_resp = _mock_sse_response("reconnected")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[
            aiohttp.ClientError("Connection reset"),
            _FakeContextManager(ok_resp),
        ],
    )

    events = await _collect_events(transport.stream([], [], ""))
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) == 1
    assert text_events[0].delta == "reconnected"


async def test_retry_on_520() -> None:
    transport = _make_transport()
    error_resp = _mock_response(520, "Web server returned an unknown error")
    ok_resp = _mock_sse_response("ok")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(error_resp), _FakeContextManager(ok_resp)],
    )

    events = await _collect_events(transport.stream([], [], ""))
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) == 1
    assert text_events[0].delta == "ok"


async def test_success_on_first_attempt() -> None:
    transport = _make_transport()
    ok_resp = _mock_sse_response("first try")
    transport.session.post = MagicMock(  # type: ignore[union-attr,method-assign]
        side_effect=[_FakeContextManager(ok_resp)],
    )

    events = await _collect_events(transport.stream([], [], ""))
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_events) == 1
    assert text_events[0].delta == "first try"
    assert transport.session.post.call_count == 1  # type: ignore[union-attr]


async def test_exponential_backoff_delays() -> None:
    transport = _make_transport(max_retries=3, retry_base_delay=1.0)
    error_resps = [_FakeContextManager(_mock_response(503, "unavailable")) for _ in range(3)]
    transport.session.post = MagicMock(side_effect=error_resps)  # type: ignore[union-attr,method-assign]

    with patch("axio_transport_openai.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(StreamError):
            await _collect_events(transport.stream([], [], ""))

        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)
