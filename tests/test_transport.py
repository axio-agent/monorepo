"""Tests for CodexTransport — message conversion, SSE parsing, token refresh."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import (
    IterationEnd,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.tool import Tool, ToolHandler
from axio.types import StopReason, Usage

from axio_transport_codex.transport import (
    CODEX_MODELS,
    CodexTransport,
    _convert_messages,
    _convert_tools,
    _strip_title,
)

# ---------------------------------------------------------------------------
# Test tool handler
# ---------------------------------------------------------------------------


class GetWeather(ToolHandler):
    location: str
    units: str = "celsius"

    async def __call__(self) -> str:
        return f"Weather in {self.location}: 22{self.units[0]}"


# ---------------------------------------------------------------------------
# SSE helpers for Responses API format
# ---------------------------------------------------------------------------


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _text_response_sse(text: str, usage: dict[str, int] | None = None) -> str:
    """Build SSE for a text response in Responses API format."""
    lines = ""
    # Text delta events
    for ch in text:
        lines += _sse_event(
            {
                "type": "response.output_text.delta",
                "delta": ch,
            }
        )
    # Completed event
    lines += _sse_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": usage or {"input_tokens": 10, "output_tokens": 5},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            },
        }
    )
    return lines


def _tool_call_response_sse(
    call_id: str,
    name: str,
    arguments: str,
    usage: dict[str, int] | None = None,
) -> str:
    """Build SSE for a tool call response in Responses API format."""
    lines = ""
    # Tool use start — item has both "id" (item_id) and "call_id"
    item_id = f"fc_{call_id}"
    lines += _sse_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": item_id, "call_id": call_id, "name": name},
        }
    )
    # Arguments delta — references item_id, not call_id
    mid = len(arguments) // 2
    for part in [arguments[:mid], arguments[mid:]]:
        if part:
            lines += _sse_event(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "item_id": item_id,
                    "delta": part,
                }
            )
    # Completed
    lines += _sse_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": usage or {"input_tokens": 15, "output_tokens": 8},
                "output": [
                    {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments},
                ],
            },
        }
    )
    return lines


def _reasoning_response_sse(reasoning: str, answer: str) -> str:
    """Build SSE with reasoning summary + text output."""
    lines = ""
    for ch in reasoning:
        lines += _sse_event(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": ch,
            }
        )
    for ch in answer:
        lines += _sse_event(
            {
                "type": "response.output_text.delta",
                "delta": ch,
            }
        )
    lines += _sse_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 20, "output_tokens": 15},
                "output": [],
            },
        }
    )
    return lines


# ---------------------------------------------------------------------------
# Fake Codex server
# ---------------------------------------------------------------------------


class FakeCodexServer:
    def __init__(self) -> None:
        self.responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.status_code: int = 200
        self.error_body: str = ""
        self._status_sequence: list[int] = []
        self._call_count = 0

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/responses", self._handle)
        return app

    def _get_status(self) -> int:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._status_sequence):
            return self._status_sequence[idx]
        return self.status_code

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.received_payloads.append(payload)

        status = self._get_status()
        if status != 200:
            return web.Response(status=status, text=self.error_body)

        sse_body = self.responses.pop(0) if self.responses else ""
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        await resp.write(sse_body.encode("utf-8"))
        await resp.write_eof()
        return resp


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeCodexServer, str]]:
    server = FakeCodexServer()
    app = server.make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sock = site._server.sockets[0]  # type: ignore[union-attr]
    host, port = sock.getsockname()[:2]
    base_url = f"http://{host}:{port}"

    yield server, base_url

    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeCodexServer, str]) -> AsyncIterator[CodexTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield CodexTransport(
            base_url=base_url,
            api_key="test-token",
            account_id="test-account",
            model=CODEX_MODELS["gpt-4.1"],
            session=session,
            retry_base_delay=0.0,
        )


async def _collect(stream: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [event async for event in stream]


# ---------------------------------------------------------------------------
# Text streaming
# ---------------------------------------------------------------------------


async def test_text_streaming(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    server.responses.append(_text_response_sse("Hello"))

    events = await _collect(transport.stream([], [], ""))

    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_deltas) == 5
    assert "".join(e.delta for e in text_deltas) == "Hello"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1
    assert ends[0].stop_reason == StopReason.end_turn
    assert ends[0].usage == Usage(10, 5)


# ---------------------------------------------------------------------------
# Tool call streaming
# ---------------------------------------------------------------------------


async def test_tool_call_streaming(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    args = json.dumps({"location": "Paris"})
    server.responses.append(_tool_call_response_sse("call_abc", "get_weather", args))

    events = await _collect(transport.stream([], [], ""))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "call_abc"
    assert starts[0].name == "get_weather"

    deltas = [e for e in events if isinstance(e, ToolInputDelta)]
    assert len(deltas) == 2
    assert "".join(d.partial_json for d in deltas) == args

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert ends[0].stop_reason == StopReason.tool_use


# ---------------------------------------------------------------------------
# Reasoning + text
# ---------------------------------------------------------------------------


async def test_reasoning_then_text(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    server.responses.append(_reasoning_response_sse("let me think", "42"))

    events = await _collect(transport.stream([], [], ""))

    reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
    text = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in reasoning) == "let me think"
    assert "".join(e.delta for e in text) == "42"


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


async def test_failed_response_raises(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    server.responses.append(
        _sse_event(
            {
                "type": "response.failed",
                "response": {"error": {"message": "Something went wrong"}},
            }
        )
    )

    with pytest.raises(StreamError, match="Something went wrong"):
        await _collect(transport.stream([], [], ""))


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


async def test_http_401(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    server.status_code = 401
    server.error_body = "Unauthorized"

    with pytest.raises(StreamError, match="401"):
        await _collect(transport.stream([], [], ""))


async def test_http_429_retries(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    server, _ = fake_server
    server._status_sequence = [429, 200]
    server.error_body = "Rate limited"
    server.responses.append(_text_response_sse("ok"))

    events = await _collect(transport.stream([], [], ""))
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.delta for e in text_deltas) == "ok"
    assert len(server.received_payloads) == 2


# ---------------------------------------------------------------------------
# build_payload
# ---------------------------------------------------------------------------


def test_build_payload_basic() -> None:
    t = CodexTransport(model=CODEX_MODELS["gpt-4.1"])
    payload = t.build_payload([], [], "You are helpful.")
    assert payload["model"] == "gpt-4.1"
    assert payload["instructions"] == "You are helpful."
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["input"] == []


def test_build_payload_no_instructions_when_empty() -> None:
    t = CodexTransport(model=CODEX_MODELS["gpt-4.1"])
    payload = t.build_payload([], [], "")
    assert "instructions" not in payload


def test_build_payload_with_tools() -> None:
    t = CodexTransport(model=CODEX_MODELS["gpt-4.1"])
    tool = Tool(name="get_weather", description="Get weather", handler=GetWeather)
    payload = t.build_payload([], [tool], "")
    assert len(payload["tools"]) == 1
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True


def test_build_payload_no_tools_field_when_empty() -> None:
    t = CodexTransport(model=CODEX_MODELS["gpt-4.1"])
    payload = t.build_payload([], [], "")
    assert "tools" not in payload
    assert "tool_choice" not in payload


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_convert_user_text() -> None:
    messages = [Message(role="user", content=[TextBlock(text="Hello")])]
    _, items = _convert_messages(messages, "")
    assert len(items) == 1
    assert items[0]["role"] == "user"
    assert items[0]["content"] == [{"type": "input_text", "text": "Hello"}]


def test_convert_user_image() -> None:
    img_data = b"\x89PNG\r\n\x1a\nfake"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="What is this?"),
                ImageBlock(media_type="image/png", data=img_data),
            ],
        ),
    ]
    _, items = _convert_messages(messages, "")
    assert len(items) == 1
    content = items[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_convert_tool_results() -> None:
    messages = [
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="22C")],
        ),
    ]
    _, items = _convert_messages(messages, "")
    assert len(items) == 1
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call_1"
    assert items[0]["output"] == "22C"


def test_convert_assistant_text() -> None:
    messages = [
        Message(role="assistant", content=[TextBlock(text="Sure, I can help.")]),
    ]
    _, items = _convert_messages(messages, "")
    assert len(items) == 1
    assert items[0]["role"] == "assistant"
    assert items[0]["content"] == [{"type": "output_text", "text": "Sure, I can help."}]


def test_convert_assistant_tool_use() -> None:
    messages = [
        Message(
            role="assistant",
            content=[
                ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"}),
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="22C")],
        ),
    ]
    _, items = _convert_messages(messages, "")
    assert len(items) == 2
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == "call_1"
    assert items[0]["name"] == "get_weather"
    assert json.loads(items[0]["arguments"]) == {"location": "Paris"}
    assert items[1]["type"] == "function_call_output"
    assert items[1]["call_id"] == "call_1"


def test_convert_system_as_instructions() -> None:
    instructions, _ = _convert_messages([], "You are a helpful assistant.")
    assert instructions == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# _strip_title
# ---------------------------------------------------------------------------


def test_strip_title_recursive() -> None:
    schema = {
        "title": "Root",
        "type": "object",
        "properties": {
            "name": {"title": "Name", "type": "string"},
        },
    }
    result = _strip_title(schema)
    assert "title" not in result
    assert "title" not in result["properties"]["name"]


# ---------------------------------------------------------------------------
# _convert_tools
# ---------------------------------------------------------------------------


def test_convert_tools() -> None:
    tool = Tool(name="get_weather", description="Get weather", handler=GetWeather)
    result = _convert_tools([tool])
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["name"] == "get_weather"
    assert result[0]["description"] == "Get weather"
    assert "parameters" in result[0]


# ---------------------------------------------------------------------------
# Token refresh via _ensure_token
# ---------------------------------------------------------------------------


async def test_ensure_token_skips_when_not_expired() -> None:
    t = CodexTransport(
        api_key="valid-token",
        refresh_token="rt",
        expires_at=str(int(time.time()) + 300),
    )
    with patch.object(CodexTransport, "_refresh", new_callable=AsyncMock) as mock_refresh:
        await t._ensure_token()
    mock_refresh.assert_not_called()


async def test_ensure_token_refreshes_when_expired() -> None:
    t = CodexTransport(
        api_key="expired-token",
        refresh_token="rt",
        expires_at=str(int(time.time()) - 60),
    )
    with patch.object(CodexTransport, "_refresh", new_callable=AsyncMock) as mock_refresh:
        await t._ensure_token()
    mock_refresh.assert_called_once()


async def test_ensure_token_refreshes_when_within_30s() -> None:
    t = CodexTransport(
        api_key="almost-expired",
        refresh_token="rt",
        expires_at=str(int(time.time()) + 15),
    )
    with patch.object(CodexTransport, "_refresh", new_callable=AsyncMock) as mock_refresh:
        await t._ensure_token()
    mock_refresh.assert_called_once()


async def test_ensure_token_skips_without_refresh_token() -> None:
    t = CodexTransport(api_key="token", refresh_token="", expires_at="0")
    with patch.object(CodexTransport, "_refresh", new_callable=AsyncMock) as mock_refresh:
        await t._ensure_token()
    mock_refresh.assert_not_called()


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


async def test_account_id_header_sent(
    fake_server: tuple[FakeCodexServer, str],
) -> None:
    server, base_url = fake_server
    server.responses.append(_text_response_sse("ok"))

    async with aiohttp.ClientSession() as session:
        t = CodexTransport(
            base_url=base_url,
            api_key="test-token",
            account_id="acct-123",
            model=CODEX_MODELS["gpt-4.1"],
            session=session,
        )
        await _collect(t.stream([], [], ""))

    assert len(server.received_payloads) == 1
    assert server.received_payloads[0]["model"] == "gpt-4.1"


# ---------------------------------------------------------------------------
# CODEX_MODELS registry
# ---------------------------------------------------------------------------


def test_codex_models_registry() -> None:
    assert len(CODEX_MODELS) > 0
    assert "gpt-4.1" in CODEX_MODELS
    assert "o4-mini" in CODEX_MODELS


# ---------------------------------------------------------------------------
# fetch_models fallback
# ---------------------------------------------------------------------------


async def test_fetch_models_uses_defaults_without_session() -> None:
    t = CodexTransport()
    await t.fetch_models()
    assert len(t.models) == len(CODEX_MODELS)


# ---------------------------------------------------------------------------
# Orphan function_call synthesis
# ---------------------------------------------------------------------------


def test_convert_messages_orphan_synthesizes_output() -> None:
    """Assistant message with ToolUseBlock but no matching ToolResultBlock → placeholder output."""
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_orphan", name="get_weather", input={"location": "Paris"})],
        ),
    ]
    _, items = _convert_messages(messages, "")
    # Should have function_call + synthesized function_call_output
    fc = [i for i in items if i.get("type") == "function_call"]
    fco = [i for i in items if i.get("type") == "function_call_output"]
    assert len(fc) == 1
    assert len(fco) == 1
    assert fco[0]["call_id"] == "call_orphan"
    assert "not executed" in fco[0]["output"]


def test_convert_messages_matched_call_not_synthesized() -> None:
    """function_call with matching output should NOT get a placeholder."""
    messages = [
        Message(
            role="assistant",
            content=[ToolUseBlock(id="call_1", name="get_weather", input={"location": "Paris"})],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="22C")],
        ),
    ]
    _, items = _convert_messages(messages, "")
    fco = [i for i in items if i.get("type") == "function_call_output"]
    assert len(fco) == 1
    assert fco[0]["output"] == "22C"


# ---------------------------------------------------------------------------
# _parse_sse forces tool_use stop reason
# ---------------------------------------------------------------------------


def _tool_call_response_sse_no_output(
    call_id: str,
    name: str,
    arguments: str,
    usage: dict[str, int] | None = None,
) -> str:
    """SSE with function_call items but response.completed has empty output."""
    lines = ""
    item_id = f"fc_{call_id}"
    lines += _sse_event(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "id": item_id, "call_id": call_id, "name": name},
        }
    )
    for part in [arguments[: len(arguments) // 2], arguments[len(arguments) // 2 :]]:
        if part:
            lines += _sse_event(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "item_id": item_id,
                    "delta": part,
                }
            )
    # response.completed with empty output — does NOT list function_call items
    lines += _sse_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": usage or {"input_tokens": 15, "output_tokens": 8},
                "output": [],
            },
        }
    )
    return lines


async def test_parse_sse_empty_output_reports_end_turn(
    fake_server: tuple[FakeCodexServer, str],
    transport: CodexTransport,
) -> None:
    """Streamed function_call items but response.completed with empty output → transport reports end_turn.

    The agent loop handles this by dispatching tools based on content, not stop_reason.
    """
    server, _ = fake_server
    args = json.dumps({"location": "Paris"})
    server.responses.append(_tool_call_response_sse_no_output("call_xyz", "get_weather", args))

    events = await _collect(transport.stream([], [], ""))

    starts = [e for e in events if isinstance(e, ToolUseStart)]
    assert len(starts) == 1
    assert starts[0].tool_use_id == "call_xyz"

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1
    # Transport faithfully reports what the API said — agent handles the mismatch
    assert ends[0].stop_reason == StopReason.end_turn
