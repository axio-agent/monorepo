"""Tests for AnthropicTransport — KV cache and rate-limit handling."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio.blocks import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock
from axio.events import IterationEnd, ReasoningDelta, StreamEvent, TextDelta, ToolInputDelta, ToolUseStart
from axio.exceptions import StreamError
from axio.messages import Message
from axio.tool import Tool, ToolHandler
from axio.types import StopReason

from axio_transport_anthropic import ANTHROPIC_MODELS, AnthropicTransport, _convert_messages


class GetWeather(ToolHandler):
    location: str

    async def __call__(self) -> str:
        return f"Weather in {self.location}"


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _text_sse(text: str, input_tokens: int = 10, output_tokens: int = 5) -> str:
    parts = _sse("message_start", {"message": {"usage": {"input_tokens": input_tokens}}})
    parts += _sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
    for ch in text:
        parts += _sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": ch}})
    parts += _sse("content_block_stop", {"index": 0})
    parts += _sse("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": output_tokens}})
    parts += _sse("message_stop", {})
    return parts


def _tool_call_sse(call_id: str, name: str, arguments: str) -> str:
    parts = _sse("message_start", {"message": {"usage": {"input_tokens": 15}}})
    parts += _sse(
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": call_id, "name": name}},
    )
    mid = len(arguments) // 2
    for chunk in [arguments[:mid], arguments[mid:]]:
        if chunk:
            parts += _sse(
                "content_block_delta",
                {"index": 0, "delta": {"type": "input_json_delta", "partial_json": chunk}},
            )
    parts += _sse("content_block_stop", {"index": 0})
    parts += _sse("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}})
    parts += _sse("message_stop", {})
    return parts


def _thinking_sse(thinking: str) -> str:
    parts = _sse("message_start", {"message": {"usage": {"input_tokens": 10}}})
    parts += _sse("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}})
    parts += _sse("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": thinking}})
    parts += _sse("content_block_stop", {"index": 0})
    parts += _sse("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}})
    parts += _sse("message_stop", {})
    return parts


# ---------------------------------------------------------------------------
# Fake server
# ---------------------------------------------------------------------------


class FakeAnthropicServer:
    def __init__(self) -> None:
        self.responses: list[str] = []
        self.received_payloads: list[dict[str, Any]] = []
        self.status_code: int = 200
        self.error_body: str = ""
        self.retry_after: str | None = None
        self._status_sequence: list[int] = []
        self._call_count: int = 0

    def _next_status(self) -> int:
        idx = self._call_count
        self._call_count += 1
        return self._status_sequence[idx] if idx < len(self._status_sequence) else self.status_code

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/messages", self._handle)
        return app

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.received_payloads.append(await request.json())
        status = self._next_status()
        if status != 200:
            headers: dict[str, str] = {}
            if self.retry_after is not None:
                headers["Retry-After"] = self.retry_after
            return web.Response(status=status, text=self.error_body, headers=headers)
        sse_body = self.responses.pop(0) if self.responses else ""
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(sse_body.encode())
        await resp.write_eof()
        return resp


@pytest.fixture
async def fake_server() -> AsyncIterator[tuple[FakeAnthropicServer, str]]:
    server = FakeAnthropicServer()
    runner = web.AppRunner(server.make_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = site._server.sockets[0].getsockname()[:2]  # type: ignore[union-attr]
    yield server, f"http://{host}:{port}"
    await runner.cleanup()


@pytest.fixture
async def transport(fake_server: tuple[FakeAnthropicServer, str]) -> AsyncIterator[AnthropicTransport]:
    _, base_url = fake_server
    async with aiohttp.ClientSession() as session:
        yield AnthropicTransport(
            base_url=base_url,
            api_key="test-key",
            model=ANTHROPIC_MODELS["claude-sonnet-4-6"],
            session=session,
            retry_base_delay=0.0,
        )


async def _collect(it: AsyncIterator[StreamEvent]) -> list[StreamEvent]:
    return [e async for e in it]


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class TestKVCache:
    def test_system_is_array_with_cache_control(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        payload = t.build_payload([], [], "You are helpful.")
        assert isinstance(payload["system"], list)
        block = payload["system"][0]
        assert block == {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}

    def test_no_system_key_when_empty(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        payload = t.build_payload([], [], "")
        assert "system" not in payload

    def test_last_tool_has_cache_control(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        tool_a = Tool(name="tool_a", description="A", handler=GetWeather)
        tool_b = Tool(name="tool_b", description="B", handler=GetWeather)
        payload = t.build_payload([], [tool_a, tool_b], "")
        tools = payload["tools"]
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_single_tool_has_cache_control(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        tool = Tool(name="my_tool", description="desc", handler=GetWeather)
        payload = t.build_payload([], [tool], "")
        assert payload["tools"][0]["cache_control"] == {"type": "ephemeral"}

    def test_no_tools_key_when_empty(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        payload = t.build_payload([], [], "")
        assert "tools" not in payload

    async def test_cache_control_sent_to_server(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.responses.append(_text_sse("ok"))
        tool = Tool(name="my_tool", description="desc", handler=GetWeather)
        await _collect(transport.stream([], [tool], "Be helpful."))
        payload = server.received_payloads[0]
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------


class TestRateLimits:
    async def test_429_retries_and_succeeds(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server._status_sequence = [429, 200]
        server.error_body = "Rate limited"
        server.responses.append(_text_sse("ok"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            events = await _collect(transport.stream([], [], ""))
        text = [e for e in events if isinstance(e, TextDelta)]
        assert "".join(e.delta for e in text) == "ok"
        assert len(server.received_payloads) == 2

    async def test_529_retries_and_succeeds(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server._status_sequence = [529, 200]
        server.error_body = "Overloaded"
        server.responses.append(_text_sse("ok"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            events = await _collect(transport.stream([], [], ""))
        text = [e for e in events if isinstance(e, TextDelta)]
        assert "".join(e.delta for e in text) == "ok"
        assert len(server.received_payloads) == 2

    async def test_retry_after_header_is_used(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server._status_sequence = [429, 200]
        server.error_body = "Rate limited"
        server.retry_after = "7"
        server.responses.append(_text_sse("ok"))
        sleep_calls: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("asyncio.sleep", capture_sleep):
            await _collect(transport.stream([], [], ""))

        assert sleep_calls == [7.0]

    async def test_all_retries_exhausted_raises(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.status_code = 429
        server.error_body = "Always limited"
        t = AnthropicTransport(
            base_url=transport.base_url,
            api_key="key",
            model=ANTHROPIC_MODELS["claude-sonnet-4-6"],
            session=transport.session,
            max_retries=3,
            retry_base_delay=0.0,
        )
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(StreamError, match="429"):
                await _collect(t.stream([], [], ""))
        assert len(server.received_payloads) == 3

    async def test_401_does_not_retry(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.status_code = 401
        server.error_body = "Unauthorized"
        with pytest.raises(StreamError, match="401"):
            await _collect(transport.stream([], [], ""))
        assert len(server.received_payloads) == 1


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


class TestMessageConversion:
    def test_user_text_block(self) -> None:
        msgs = [Message(role="user", content=[TextBlock(text="hello")])]
        result = _convert_messages(msgs)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_user_image_block(self) -> None:
        raw = b"\x89PNG\r\n"
        msgs = [Message(role="user", content=[ImageBlock(media_type="image/png", data=raw)])]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert block["source"]["data"] == base64.b64encode(raw).decode("ascii")

    def test_user_tool_result_string(self) -> None:
        msgs = [Message(role="user", content=[ToolResultBlock(tool_use_id="id1", content="done")])]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block == {"type": "tool_result", "tool_use_id": "id1", "content": "done"}

    def test_user_tool_result_list_text(self) -> None:
        msgs = [
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="id2", content=[TextBlock(text="ok")])],
            )
        ]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block["content"] == [{"type": "text", "text": "ok"}]

    def test_user_tool_result_list_image(self) -> None:
        raw = b"\xff\xd8"
        msgs = [
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="id3", content=[ImageBlock(media_type="image/jpeg", data=raw)])],
            )
        ]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block["content"][0]["type"] == "image"
        assert block["content"][0]["source"]["data"] == base64.b64encode(raw).decode("ascii")

    def test_user_tool_result_is_error(self) -> None:
        msgs = [Message(role="user", content=[ToolResultBlock(tool_use_id="id4", content="boom", is_error=True)])]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block["is_error"] is True

    def test_assistant_text_block(self) -> None:
        msgs = [Message(role="assistant", content=[TextBlock(text="hi")])]
        result = _convert_messages(msgs)
        assert result == [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]

    def test_assistant_tool_use_block(self) -> None:
        msgs = [
            Message(
                role="assistant",
                content=[ToolUseBlock(id="tu1", name="get_weather", input={"location": "NYC"})],
            )
        ]
        result = _convert_messages(msgs)
        block = result[0]["content"][0]
        assert block == {"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {"location": "NYC"}}

    def test_empty_content_skipped(self) -> None:
        msgs = [Message(role="user", content=[])]
        result = _convert_messages(msgs)
        assert result == []


# ---------------------------------------------------------------------------
# SSE streaming — tool calls and reasoning
# ---------------------------------------------------------------------------


class TestSSEStreaming:
    async def test_tool_call_events(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.responses.append(_tool_call_sse("call-1", "get_weather", '{"location":"NYC"}'))
        events = await _collect(transport.stream([], [], ""))
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        deltas = [e for e in events if isinstance(e, ToolInputDelta)]
        assert len(starts) == 1
        assert starts[0].tool_use_id == "call-1"
        assert starts[0].name == "get_weather"
        assert "".join(d.partial_json for d in deltas) == '{"location":"NYC"}'
        assert all(d.tool_use_id == "call-1" for d in deltas)

    async def test_reasoning_delta(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.responses.append(_thinking_sse("let me think"))
        events = await _collect(transport.stream([], [], ""))
        reasoning = [e for e in events if isinstance(e, ReasoningDelta)]
        assert len(reasoning) == 1
        assert reasoning[0].delta == "let me think"

    async def test_iteration_end_carries_usage_and_stop_reason(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.responses.append(_text_sse("hi", input_tokens=42, output_tokens=7))
        events = await _collect(transport.stream([], [], ""))
        ends = [e for e in events if isinstance(e, IterationEnd)]
        assert len(ends) == 1
        assert ends[0].usage.input_tokens == 42
        assert ends[0].usage.output_tokens == 7
        assert ends[0].stop_reason == StopReason.end_turn

    async def test_tool_use_stop_reason(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        server.responses.append(_tool_call_sse("c1", "fn", "{}"))
        events = await _collect(transport.stream([], [], ""))
        ends = [e for e in events if isinstance(e, IterationEnd)]
        assert ends[0].stop_reason == StopReason.tool_use

    async def test_unknown_stop_reason_maps_to_error(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        server, _ = fake_server
        sse = _sse("message_start", {"message": {"usage": {"input_tokens": 1}}})
        sse += _sse("message_delta", {"delta": {"stop_reason": "future_reason"}, "usage": {"output_tokens": 1}})
        sse += _sse("message_stop", {})
        server.responses.append(sse)
        events = await _collect(transport.stream([], [], ""))
        ends = [e for e in events if isinstance(e, IterationEnd)]
        assert ends[0].stop_reason == StopReason.error


# ---------------------------------------------------------------------------
# Connection error retry
# ---------------------------------------------------------------------------


class TestConnectionError:
    async def test_client_error_retries(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        assert transport.session is not None
        call_count = 0
        original_post = transport.session.post

        def patched_post(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise aiohttp.ClientConnectionError("network down")
            return original_post(*args, **kwargs)

        fake_server[0].responses.append(_text_sse("ok"))
        with patch.object(transport.session, "post", side_effect=patched_post):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                events = await _collect(transport.stream([], [], ""))
        text = [e for e in events if isinstance(e, TextDelta)]
        assert "".join(e.delta for e in text) == "ok"


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_roundtrip(self) -> None:
        t = AnthropicTransport(
            base_url="https://example.com",
            api_key="sk-test",
            model=ANTHROPIC_MODELS["claude-sonnet-4-6"],
        )
        d = t.to_dict()
        assert d["base_url"] == "https://example.com"
        assert d["api_key"] == "sk-test"
        assert any(m["id"] == "claude-sonnet-4-6" for m in d["models"])

    def test_from_dict_restores_transport(self) -> None:
        original = AnthropicTransport(
            base_url="https://example.com",
            api_key="sk-orig",
            model=ANTHROPIC_MODELS["claude-haiku-4-5-20251001"],
        )
        d = original.to_dict()
        restored = AnthropicTransport.from_dict(d)
        assert restored.base_url == "https://example.com"
        assert restored.api_key == "sk-orig"
        assert any(m.id == "claude-haiku-4-5-20251001" for m in restored.models.values())
