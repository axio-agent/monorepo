"""Tests for AnthropicTransport - KV cache and rate-limit handling."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
from aiohttp import web
from axio.agent import Agent
from axio.blocks import (
    ImageBlock,
    ProviderBlock,
    ReasoningBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from axio.context import MemoryContextStore
from axio.events import (
    BlockEnd,
    Citation,
    IterationEnd,
    IterationStart,
    ProviderEvent,
    ProviderOutput,
    ReasoningDelta,
    ReasoningSignature,
    Refusal,
    StreamEvent,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
)
from axio.exceptions import StreamError
from axio.messages import Message
from axio.testing import StubTransport, assert_stream_contract, make_echo_tool
from axio.tool import Tool
from axio.types import StopReason, Usage
from axio_sse import Event, UnknownEvent

from axio_transport_anthropic import ANTHROPIC_MODELS, AnthropicTransport, Messages, _convert_messages


async def get_weather(location: str) -> str:
    return f"Weather in {location}"


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
    """Every event the stream produced, checked against what any transport must produce.

    Asserted here rather than in one test, so every stream this package drives is held to it.
    """
    made = [e async for e in it]
    assert_stream_contract(made)
    return made


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

    def test_system_message_in_history_appended_to_system(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        messages = [Message(role="system", content=[TextBlock(text="wrap up now")])]
        payload = t.build_payload(messages, [], "")
        assert payload["system"] == [{"type": "text", "text": "wrap up now"}]

    def test_system_message_in_history_combined_with_prompt(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        messages = [Message(role="system", content=[TextBlock(text="wrap up now")])]
        payload = t.build_payload(messages, [], "You are helpful.")
        assert len(payload["system"]) == 2
        assert payload["system"][0] == {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        }
        assert payload["system"][1] == {"type": "text", "text": "wrap up now"}

    def test_system_message_not_in_messages_array(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        messages = [Message(role="system", content=[TextBlock(text="wrap up")])]
        payload = t.build_payload(messages, [], "")
        assert all(m.get("role") != "system" for m in payload["messages"])

    def test_last_tool_has_cache_control(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        tool_a: Tool[Any] = Tool(name="tool_a", description="A", handler=get_weather)
        tool_b: Tool[Any] = Tool(name="tool_b", description="B", handler=get_weather)
        payload = t.build_payload([], [tool_a, tool_b], "")
        tools = payload["tools"]
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == {"type": "ephemeral"}

    def test_single_tool_has_cache_control(self) -> None:
        t = AnthropicTransport(model=ANTHROPIC_MODELS["claude-sonnet-4-6"])
        tool: Tool[Any] = Tool(name="my_tool", description="desc", handler=get_weather)
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
        tool: Tool[Any] = Tool(name="my_tool", description="desc", handler=get_weather)
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
# SSE streaming - tool calls and reasoning
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

    async def test_an_unknown_stop_reason_reads_as_unknown(
        self,
        fake_server: tuple[FakeAnthropicServer, str],
        transport: AnthropicTransport,
    ) -> None:
        # Every other answer claims something the API did not say. Raising also discards content
        # the caller has already read.
        server, _ = fake_server
        sse = _sse("message_start", {"message": {"usage": {"input_tokens": 1}}})
        sse += _sse("message_delta", {"delta": {"stop_reason": "future_reason"}, "usage": {"output_tokens": 1}})
        sse += _sse("message_stop", {})
        server.responses.append(sse)

        events = await _collect(transport.stream([], [], ""))

        assert [e.stop_reason for e in events if isinstance(e, IterationEnd)] == [StopReason.unknown]

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


class TestUsageAccounting:
    async def test_cached_input_is_counted_and_not_lost(
        self, fake_server: tuple[FakeAnthropicServer, str], transport: AnthropicTransport
    ) -> None:
        """input_tokens counts only past the last cache breakpoint, so the cache has to be added.

        This transport sets cache_control itself, so before the counts were added back a cached
        100k prompt was reported as the handful of tokens after the breakpoint.
        """
        server, _ = fake_server
        sse = _sse(
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": 50,
                        "cache_read_input_tokens": 100_000,
                        "cache_creation_input_tokens": 148,
                    }
                }
            },
        )
        sse += _sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})
        sse += _sse("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}})
        sse += _sse("content_block_stop", {"index": 0})
        sse += _sse(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 503, "output_tokens_details": {"thinking_tokens": 200}},
            },
        )
        sse += _sse("message_stop", {})
        server.responses = [sse]

        events = await _collect(transport.stream([Message(role="user", content=[TextBlock(text="hi")])], [], ""))
        end = [e for e in events if isinstance(e, IterationEnd)][0]

        assert end.usage.input_tokens == 100_198, "the cached tokens were dropped from the input total"
        assert end.usage.cache_read_tokens == 100_000
        assert end.usage.cache_write_tokens == 148
        assert end.usage.uncached_input_tokens == 50
        assert end.usage.output_tokens == 503, "thinking is already inside output_tokens and must not be added"
        assert end.usage.reasoning_tokens == 200


class TestTheWholeVocabulary:
    """Every event the Messages API publishes, and what each one now becomes."""

    def test_the_reader_names_every_published_event(self) -> None:
        # The published list, from the streaming reference. A ninth name is the API's news.
        assert Messages.names() == {
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
            "ping",
            "error",
        }

    def test_an_event_outside_the_published_list_is_refused_under_strict(self) -> None:
        with pytest.raises(UnknownEvent, match="message_paused"):
            Messages().read(Event(data="{}", event="message_paused"), strict=True)

    def test_a_delta_kind_nobody_reads_obeys_the_same_policy(self) -> None:
        # The block names the delta, so the delta type is a second vocabulary inside one event.
        event = Event(data=json.dumps({"index": 0, "delta": {"type": "something_delta"}}), event="content_block_delta")
        assert Messages().read(event) == []
        with pytest.raises(UnknownEvent, match="something_delta"):
            Messages().read(event, strict=True)

    def test_a_thinking_signature_reaches_the_caller(self) -> None:
        # The API refuses a returned thinking block whose signature is missing, so a turn that
        # dropped it could not be replayed.
        reader = Messages()
        made = reader.read(
            Event(
                data=json.dumps({"index": 0, "delta": {"type": "signature_delta", "signature": "ErUBCkYIBRgC"}}),
                event="content_block_delta",
            )
        )
        # Named beside the proof, so a session that later changes transport does not send an
        # Anthropic signature to a provider that never issued one.
        assert made == [ReasoningSignature(index=0, signature="ErUBCkYIBRgC", provider="anthropic")]

    def test_a_redacted_thinking_block_carries_its_proof_and_no_text(self) -> None:
        made = Messages().read(
            Event(
                data=json.dumps({"index": 1, "content_block": {"type": "redacted_thinking", "data": "EroBCkYI"}}),
                event="content_block_start",
            )
        )
        assert made == [ReasoningSignature(index=1, signature="EroBCkYI", redacted=True, provider="anthropic")]

    def test_a_citation_carries_the_unit_its_span_is_counted_in(self) -> None:
        # Only char_location counts characters; page_location counts pages. Compared across units
        # the offsets mean nothing, so the unit travels with them.
        made = Messages().read(
            Event(
                data=json.dumps(
                    {
                        "index": 0,
                        "delta": {
                            "type": "citations_delta",
                            "citation": {
                                "type": "page_location",
                                "cited_text": "as reported",
                                "document_title": "The Report",
                                "document_index": 3,
                            },
                        },
                    }
                ),
                event="content_block_delta",
            )
        )
        citation = made[0]
        assert isinstance(citation, Citation)
        assert (citation.cited_text, citation.title, citation.source_id) == ("as reported", "The Report", "3")
        assert citation.unit == "page"

    def test_a_server_side_tool_block_is_forwarded_rather_than_dropped(self) -> None:
        made = Messages().read(
            Event(
                data=json.dumps(
                    {"index": 2, "content_block": {"type": "web_search_tool_result", "content": [{"title": "a"}]}}
                ),
                event="content_block_start",
            )
        )
        forwarded = made[0]
        assert isinstance(forwarded, ProviderEvent)
        assert (forwarded.provider, forwarded.kind, forwarded.index) == ("anthropic", "web_search_tool_result", 2)
        assert forwarded.data["content"] == [{"title": "a"}]

    def test_a_block_that_ends_says_so(self) -> None:
        assert Messages().read(Event(data='{"index": 4}', event="content_block_stop")) == [BlockEnd(index=4)]

    def test_the_model_that_served_the_turn_is_reported(self) -> None:
        made = Messages().read(
            Event(data=json.dumps({"message": {"id": "msg_1", "model": "claude-sonnet-4-6"}}), event="message_start")
        )
        assert made == [IterationStart(iteration=0, id="msg_1", model="claude-sonnet-4-6")]

    @pytest.mark.parametrize(
        ("published", "expected"),
        [
            ("end_turn", StopReason.end_turn),
            ("stop_sequence", StopReason.end_turn),
            ("tool_use", StopReason.tool_use),
            ("max_tokens", StopReason.max_tokens),
            ("refusal", StopReason.refusal),
            ("pause_turn", StopReason.pause_turn),
            ("model_context_window_exceeded", StopReason.context_window_exceeded),
        ],
    )
    def test_every_published_stop_reason_is_mapped(self, published: str, expected: StopReason) -> None:
        # A reason left out was read as an error, which ends a run the provider expected to resume.
        reader = Messages()
        reader.read(Event(data=json.dumps({"delta": {"stop_reason": published}}), event="message_delta"))
        assert reader.finished().stop_reason == expected


class TestRefusalAndCumulativeUsage:
    def test_a_decline_reaches_the_caller_with_the_policy_that_triggered_it(self) -> None:
        """A decline is a successful response with `content: []`.

        With nothing emitted for it the turn arrived as an empty answer, and the run ended on a
        RuntimeError that named no reason a user could act on.
        """
        reader = Messages()
        made = reader.read(
            Event(
                data=json.dumps(
                    {
                        "delta": {
                            "stop_reason": "refusal",
                            "stop_details": {
                                "type": "refusal",
                                "category": "cyber",
                                "explanation": "This request was declined because it could enable cyber harm.",
                            },
                        },
                        "usage": {"output_tokens": 0},
                    }
                ),
                event="message_delta",
            )
        )
        refusal = made[0]
        assert isinstance(refusal, Refusal)
        assert refusal.category == "cyber"
        assert refusal.text.startswith("This request was declined")
        assert reader.finished().stop_reason == StopReason.refusal

    def test_a_decline_with_no_named_category_still_arrives(self) -> None:
        # Both fields are null where the decline maps to no category. That null is permanent, not a
        # placeholder, so it must not stop the event being emitted.
        made = Messages().read(Event(data=json.dumps({"delta": {"stop_reason": "refusal"}}), event="message_delta"))
        assert made == [Refusal(index=0, text="", category=None, raw={})]

    def test_an_ordinary_turn_emits_no_refusal(self) -> None:
        made = Messages().read(Event(data=json.dumps({"delta": {"stop_reason": "end_turn"}}), event="message_delta"))
        assert made == []

    def test_the_cumulative_input_count_is_read_and_not_left_at_the_opening_one(self) -> None:
        """message_delta usage is cumulative in every field, not only the output ones.

        Read back from message_start alone, a turn that ran a server-side tool reported a fraction
        of what it was billed for.
        """
        reader = Messages()
        reader.read(
            Event(
                data=json.dumps({"message": {"usage": {"input_tokens": 2679, "cache_read_input_tokens": 0}}}),
                event="message_start",
            )
        )
        reader.read(
            Event(
                data=json.dumps(
                    {
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"input_tokens": 10682, "output_tokens": 510},
                    }
                ),
                event="message_delta",
            )
        )
        usage = reader.finished().usage
        assert usage.input_tokens == 10682, "8003 billed input tokens went unreported"
        assert usage.output_tokens == 510

    def test_a_plain_turn_that_repeats_no_input_count_keeps_the_opening_one(self) -> None:
        # Ordinary turns omit the input fields from message_delta, so the guard has to be presence
        # and not a blind overwrite.
        reader = Messages()
        reader.read(
            Event(
                data=json.dumps({"message": {"usage": {"input_tokens": 40, "cache_read_input_tokens": 100}}}),
                event="message_start",
            )
        )
        reader.read(
            Event(
                data=json.dumps({"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
                event="message_delta",
            )
        )
        usage = reader.finished().usage
        assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (140, 100, 7)


class TestReasoningIsReplayed:
    def test_a_signed_thinking_block_goes_back_with_its_signature(self) -> None:
        # With extended thinking on, a turn that thought and then called a tool is refused unless
        # its thinking comes back with the signature the API issued for it.
        messages = [
            Message(
                role="assistant",
                content=[
                    ReasoningBlock(text="weighing it", signature="ErUBCkYIBRgC"),
                    TextBlock(text="the answer"),
                ],
            )
        ]
        parts = _convert_messages(messages)[0]["content"]
        assert parts[0] == {"type": "thinking", "thinking": "weighing it", "signature": "ErUBCkYIBRgC"}
        assert parts[1] == {"type": "text", "text": "the answer"}

    def test_a_redacted_block_goes_back_as_the_proof_it_is(self) -> None:
        messages = [Message(role="assistant", content=[ReasoningBlock(signature="EroBCkYI", redacted=True)])]
        assert _convert_messages(messages)[0]["content"] == [{"type": "redacted_thinking", "data": "EroBCkYI"}]

    def test_an_unsigned_block_is_dropped_because_the_api_would_refuse_it(self) -> None:
        messages = [Message(role="assistant", content=[ReasoningBlock(text="unsigned"), TextBlock(text="answer")])]
        assert _convert_messages(messages)[0]["content"] == [{"type": "text", "text": "answer"}]


def test_a_stream_that_ends_without_a_stop_reason_raises() -> None:
    """Every turn ends on a message_delta carrying a stop_reason.

    Without one the connection was cut. Reported as an ending, the partial text is stored as the
    model's answer, which is what the other three transports now refuse to do.
    """
    reader = Messages()
    reader.read(
        Event(
            data=json.dumps({"index": 0, "delta": {"type": "text_delta", "text": "half"}}), event="content_block_delta"
        )
    )
    with pytest.raises(StreamError, match="without a stop_reason"):
        reader.finished()


def test_a_turn_that_ended_properly_still_finishes() -> None:
    reader = Messages()
    reader.read(Event(data=json.dumps({"delta": {"stop_reason": "end_turn"}}), event="message_delta"))
    assert reader.finished().stop_reason == StopReason.end_turn


class TestAProofFromAnotherProviderIsNotSentHere:
    """The value means something only to the protocol that made it.

    Anthropic sends it as a thinking signature, Google as thoughtSignature, Responses as
    encrypted_content. Read out of the same field by whichever converter runs, a session that
    changed transport put one provider's opaque data in another's protocol slot.
    """

    @staticmethod
    def _thinking(block: ReasoningBlock) -> list[dict[str, Any]]:
        messages = [Message(role="assistant", content=[block, TextBlock(text="done")])]
        parts = _convert_messages(messages)[0]["content"]
        return [p for p in parts if p["type"] in ("thinking", "redacted_thinking")]

    def test_a_google_proof_is_left_out(self) -> None:
        assert self._thinking(ReasoningBlock(text="hm", signature="sig", provider="google")) == []

    def test_a_redacted_block_from_elsewhere_is_left_out_too(self) -> None:
        assert self._thinking(ReasoningBlock(signature="sig", redacted=True, provider="openai")) == []

    def test_this_provider_s_own_proof_still_travels(self) -> None:
        assert self._thinking(ReasoningBlock(text="hm", signature="sig", provider="anthropic")) == [
            {"type": "thinking", "thinking": "hm", "signature": "sig"}
        ]

    def test_a_proof_with_no_provider_recorded_still_travels(self) -> None:
        # A turn stored before anyone recorded one. Dropped, an existing session loses proofs that
        # are valid and the API refuses the replay.
        assert self._thinking(ReasoningBlock(text="hm", signature="sig")) == [
            {"type": "thinking", "thinking": "hm", "signature": "sig"}
        ]


async def test_a_refused_call_reaches_the_next_request_with_its_reason() -> None:
    """What the agent stored for a call it would not run has to survive the conversion.

    The point of keeping the call is that the next request can tell it from a turn that called
    nothing. Stored and then dropped here, the model is back to guessing.
    """
    transport = StubTransport(
        [
            [
                ToolUseStart(index=0, tool_use_id="call_1", name="echo"),
                ToolInputDelta(index=0, tool_use_id="call_1", partial_json='{"msg":"x"}'),
                IterationEnd(1, StopReason.max_tokens, Usage(1, 1)),
            ]
        ]
    )
    context = MemoryContextStore()
    async for _ in Agent(system="", tools=[make_echo_tool()], transport=transport).run_stream("hi", context):
        pass

    request = _convert_messages(await context.get_history())

    calls = [p for m in request for p in m["content"] if p.get("type") == "tool_use"]
    results = [p for m in request for p in m["content"] if p.get("type") == "tool_result"]
    assert [p["id"] for p in calls] == ["call_1"]
    assert [p["tool_use_id"] for p in results] == ["call_1"], "the API refuses a call nothing answered"
    assert results[0]["is_error"] is True
    assert "max_tokens" in results[0]["content"], "the reason is what the model reads to know why"


class TestABlockTheApiRanItselfSurvivesTheTurn:
    """This API keeps no copy of the turn: the whole message list goes back on every request.

    A block from a tool the API ran — a web search, a code execution, an MCP call — was forwarded
    as news and never stored, so the next request did not carry what the model had answered from.
    """

    @staticmethod
    def _read(reader: Messages, name: str, **payload: Any) -> list[StreamEvent]:
        return reader.read(Event(data=json.dumps(payload), event=name))

    def test_a_result_block_is_kept_whole(self) -> None:
        block = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{"type": "web_search_result", "title": "axio", "url": "https://example.test"}],
        }
        reader = Messages()

        self._read(reader, "content_block_start", index=0, content_block=block)
        made = self._read(reader, "content_block_stop", index=0)

        assert made[0] == ProviderOutput(provider="anthropic", kind="web_search_tool_result", data=block, index=0)

    def test_a_server_call_waits_for_the_arguments_it_streams(self) -> None:
        # The block opens with an empty `input` that the deltas fill. Stored from the opening
        # payload, the call goes back with no arguments and the API refuses the result after it.
        reader = Messages()
        start = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}

        self._read(reader, "content_block_start", index=0, content_block=start)
        self._read(reader, "content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"q"'})
        self._read(reader, "content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": ':"a"}'})
        made = self._read(reader, "content_block_stop", index=0)

        assert isinstance(made[0], ProviderOutput)
        assert made[0].data["input"] == {"q": "a"}
        assert made[0].id == "srvtoolu_1"

    def test_arguments_cut_short_drop_the_block_rather_than_replay_half(self) -> None:
        reader = Messages()
        start = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}

        self._read(reader, "content_block_start", index=0, content_block=start)
        self._read(reader, "content_block_delta", index=0, delta={"type": "input_json_delta", "partial_json": '{"q"'})
        made = self._read(reader, "content_block_stop", index=0)

        assert not [e for e in made if isinstance(e, ProviderOutput)]

    def test_it_replays_exactly_as_it_arrived(self) -> None:
        raw = {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []}
        block = ProviderBlock(provider="anthropic", kind="web_search_tool_result", data=raw)

        parts = _convert_messages([Message(role="assistant", content=[block])])[0]["content"]

        assert parts == [raw]

    def test_a_block_from_another_provider_is_not_sent_here(self) -> None:
        block = ProviderBlock(provider="openai", kind="web_search_call", data={"type": "web_search_call"})
        messages = [Message(role="assistant", content=[block, TextBlock(text="hi")])]

        parts = _convert_messages(messages)[0]["content"]

        assert parts == [{"type": "text", "text": "hi"}]


class TestASavedAnthropicSessionResumesAsItWasSaved:
    """The same round-trip holes the OpenAI transport had, in the transport beside it."""

    def test_the_selected_model_comes_back(self) -> None:
        saved = AnthropicTransport(model=ANTHROPIC_MODELS["claude-opus-4-6"]).to_dict()

        assert AnthropicTransport.from_dict(saved).model.id == "claude-opus-4-6"

    def test_a_model_named_by_a_partial_config_is_found_in_the_class_registry(self) -> None:
        # Handed an empty registry, a hand-written settings dict named a model and lost it.
        restored = AnthropicTransport.from_dict({"name": "x", "model": "claude-opus-4-6"})

        assert restored.model.id == "claude-opus-4-6"

    def test_the_retry_policy_comes_back(self) -> None:
        saved = AnthropicTransport(max_retries=2, retry_base_delay=0.5).to_dict()

        restored = AnthropicTransport.from_dict(saved)

        assert (restored.max_retries, restored.retry_base_delay) == (2, 0.5)

    def test_the_sampling_settings_come_back(self) -> None:
        # `from_dict` read all four and `to_dict` wrote none of them.
        saved = AnthropicTransport(temperature=0.3, top_p=0.9, top_k=40, thinking_budget=2048).to_dict()

        restored = AnthropicTransport.from_dict(saved)

        assert (restored.temperature, restored.top_p, restored.top_k, restored.thinking_budget) == (
            0.3,
            0.9,
            40,
            2048,
        )

    def test_an_empty_credential_saved_on_purpose_is_not_filled_in_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-someone-elses")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.example/v1")
        saved = AnthropicTransport(api_key="", base_url="").to_dict()

        restored = AnthropicTransport.from_dict(saved)

        assert (restored.api_key, restored.base_url) == ("", "")

    def test_a_partial_config_still_takes_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-mine")

        assert AnthropicTransport.from_dict({"name": "x"}).api_key == "sk-mine"

    def test_a_number_that_will_not_read_takes_the_default(self) -> None:
        saved = AnthropicTransport(max_retries=2).to_dict()
        saved["max_retries"] = "soon"

        assert AnthropicTransport.from_dict(saved).max_retries == AnthropicTransport().max_retries
